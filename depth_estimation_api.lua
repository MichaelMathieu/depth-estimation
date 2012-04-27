package.path = "./?.lua;../?.lua;/home/myrhev/local/share/torch/lua/?.lua;/home/myrhev/local/share/torch/lua/?/init.lua;/home/myrhev/local/lib/torch/?.lua;/home/myrhev/local/lib/torch/?/init.lua"
package.cpath = "./?.so;/home/myrhev/local/lib/torch/?.so;/home/myrhev/local/lib/torch/loadall.so"

require 'torch'
require 'opticalflow_model'
require 'opticalflow_model_io'
require 'openmp'
require 'image_camera'
require 'motion_correction'

openmp.setDefaultNumThreads(2)

local input_model = 'model'
local camera_idx = 0

local loaded = loadModel(input_model, true, true)
local model = loaded.model
local filter = loaded.filter
local geometry = loaded.geometry

local cam = ImageCamera
cam:init(geometry, camera_idx)

local last_filtered = nil
local last_im = cam:getNextFrame()

collectgarbage('stop')


function nextFrameDepth()
   local im = cam:getNextFrame()

   local output = torch.Tensor(180, 320):zero()
   --if last_filtered then
   warped_im = motion_correction(last_im, im)
   --warped_im = im
   --[[
   last_filtered = filter:forward(last_im):clone()
   local filtered = filter:forward(im)
   local input = prepareInput(geometry, last_filtered, filtered)
   local moutput = model:forward(input)
   output = processOutput(geometry, moutput, true).full
   output = output[1]*10.
   output = torch.FloatTensor():resize(output:size()):copy(output)
   --]]
   last_im2 = last_im
   last_im = im
   --output = torch.FloatTensor():resize(filtered:size(2), filtered:size(3)):copy(filtered[1])
   --output = torch.FloatTensor():resize(im:size(2), im:size(3)):copy(im[1])*100
   output = output:contiguous()
   --im2 = torch.FloatTensor(warped_im[1]:size()):copy(warped_im[1])
   return warped_im, im, last_im2, output
end