package.path = "./?.lua;../?.lua;/home/myrhev/local/share/torch/lua/?.lua;/home/myrhev/local/share/torch/lua/?/init.lua;/home/myrhev/local/lib/torch/?.lua;/home/myrhev/local/lib/torch/?/init.lua"
package.cpath = "./?.so;/home/myrhev/local/lib/torch/?.so;/home/myrhev/local/lib/torch/loadall.so"

require 'torch'
require 'opticalflow_model'
require 'opticalflow_model_io'
require 'openmp'
require 'image_camera'
require 'image_loader'
require 'sfm2'

openmp.setDefaultNumThreads(2)

local input_model = 'model'
local camera_idx = 1

local loaded = loadModel(input_model, true, true)
local model = loaded.model
local filter = loaded.filter
local geometry = loaded.geometry
geometry.prefilter = true
local K = torch.Tensor(3,3)
K[1][1] = 293.824707
K[1][2] = 0.
K[1][3] = 310.435730
K[2][1] = 0.
K[2][2] = 300.631012
K[2][3] = 251.624924
K[3][1] = 0.
K[3][2] = 0.
K[3][3] = 1.
local distP = torch.Tensor(5)
distP[1] = -0.37994
distP[2] = 0.212737
distP[3] = 0.003098
distP[4] = 0.00087
distP[5] = -0.069770
local Kf = torch.FloatTensor(K:size()):copy(K)

local Khalf = Kf:clone():mul(0.5)
Khalf[3][3] = 1.0

--local cam = ImageCamera
--cam:init(geometry, camera_idx)
local cam = ImageLoader
cam:init(geometry, 'data2/ardrone1', 10, 1)

local last_filtered = nil
local last_im = cam:getNextFrame()
last_im = sfm2.undistortImage(last_im, K, distP)
last_im_scaled = image.scale(last_im, geometry.wImg, geometry.hImg)
last_filtered = filter:forward(last_im_scaled):clone()

collectgarbage('stop')

function nextFrameDepth()
   local im = cam:getNextFrame()
   im = sfm2.undistortImage(im, K, distP)
   local R,T = sfm2.getEgoMotion(last_im, im, Kf, 400)
   im_scaled = image.scale(im, geometry.wImg, geometry.hImg)
   last_filtered, mask = sfm2.removeEgoMotion(last_filtered, Khalf, R)

   dbg_last_im = last_im_scaled
   dbg_last_warped, mask = sfm2.removeEgoMotion(last_im_scaled, Khalf, R)
   
   local filtered = filter:forward(im_scaled)
   
   local input = prepareInput(geometry, last_filtered, filtered)
   local moutput = model:forward(input)
   output = processOutput(geometry, moutput, true).full
   output = output[1]*10.
   output = torch.FloatTensor():resize(output:size()):copy(output)


   last_im = im
   last_im_scaled = im_scaled
   last_filtered = filtered
   --output = torch.FloatTensor():resize(filtered:size(2), filtered:size(3)):copy(filtered[1])
   --output = torch.FloatTensor():resize(im:size(2), im:size(3)):copy(im[1])*100
   output = output:contiguous()
   --im2 = torch.FloatTensor(warped_im[1]:size()):copy(warped_im[1])
   return im_scaled, dbg_last_im, dbg_last_warped, output
end