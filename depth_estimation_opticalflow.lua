require 'torch'
require 'xlua'
require 'opticalflow_model'
require 'openmp'
require 'sys'
require 'download_model'
require 'image_loader'
require 'score_opticalflow'

torch.manualSeed(1)

op = xlua.OptionParser('%prog [options]')
-- general
op:option{'-nt', '--num-threads', action='store', dest='nThreads', default=2,
	  help='Number of threads used'}
-- input
op:option{'-i', '--input-model', action='store', dest='input_model',
	  help='Trained convnet, this option isn\'t used if -dldir is used'}
op:option{'-dldir', '--download-dir', action='store', dest='download_dir', default=nil,
	  help='scp command to the models folder (eg. mfm352@access.cims.nyu.edu:/depth-estimation/models)'}
op:option{'-rd', '--root-directory', action='store', dest='root_directory',
	  default='./data', help='Root dataset directory'}
op:option{'-fi', '--first-image', action='store', dest='first_image', default=0,
	  help='Index of first image used'}
op:option{'-d', '--delta', action='store', dest='delta', default=1,
	  help='Delta between two consecutive frames'}
-- output
op:option{'-do', '--display-output', action='store_true', dest='display_output', default=false,
	  help='Display the computed output'}
op:option{'-o', '--output-dir', action='store', dest='output_dir', default=nil,
	  help='Directory to store processed images'}

opt=op:parse()
opt.nThreads = tonumber(opt.nThreads)
opt.first_image = tonumber(opt.first_image)
opt.delta = tonumber(opt.delta)

openmp.setDefaultNumThreads(opt.nThreads)
if opt.download_dir ~= nil then
   opt.input_model = downloadModel(opt.download_dir)
   if opt.input_model == nil then
      os.exit(0)
   end
end

local loaded = loadModel(opt.input_model, true, true)
local model = loaded.model
local filter = loaded.filter
local geometry = loaded.geometry

local output_window
local timer

ImageLoader:init(geometry, opt.root_directory..'/images', opt.first_image, opt.delta)
local loader = ImageLoader

local last_frame = loader:getNextFrame()
local last_im = filter:forward(last_frame):clone()
local i = 0
while true do
   local frame = loader:getNextFrame()
   timer = torch.Timer()
   local im = filter:forward(frame):clone()
   if im == nil then
      break
   end
   local input
   if geometry.multiscale then
      input = {}
      for i = 1,#geometry.ratios do
	 input[i] = {last_im[i], im[i]}
      end
   else
      input = prepareInput(geometry, last_im, im)
   end
   local moutput = model:forward(input)
   local output = processOutput(geometry, moutput, true)
   print(timer:time())
   if opt.display_output then
      output_window = image.display{image=output.full, win=output_window}
   end
   if opt.output_dir then
      local ps = postProcessImage(output.full, 3)
      local ts = flow2hsv(geometry, ps)
      local im2 = torch.Tensor(3, 2*ts:size(2), 2*ts:size(3)):zero()
      local gthsv = flow2hsv(geometry, loader:getCurrentGT())
      im2:sub(1,im2:size(1),            1, ts:size(2),            1,ts:size(3)):copy(last_frame)
      im2:sub(1,im2:size(1),            1, ts:size(2), ts:size(3)+1,im2:size(3)):copy(frame)
      im2:sub(1, ts:size(1), ts:size(2)+1,im2:size(2),            1, ts:size(3)):copy(ts)  
      im2:sub(1,im2:size(1), ts:size(2)+1,im2:size(2), ts:size(3)+1,im2:size(3)):copy(gthsv)
      image.save(string.format('%s/%09d.png', opt.output_dir, i), im2)
   end
   last_im = im
   last_frame = frame
   i = i+1
end