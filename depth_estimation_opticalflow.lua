require 'torch'
require 'xlua'
require 'opticalflow_model'
require 'openmp'
require 'sys'
require 'download_model'
require 'image_loader'

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

local loaded = loadModel(opt.input_model, true)
local model = loaded.model
local geometry = loaded.geometry

local output_window
local timer

ImageLoader:init(geometry, opt.root_directory..'/images', opt.first_image, opt.delta)
local loader = ImageLoader

local last_im = loader:getNextFrame()
while true do
   local im = loader:getNextFrame()
   if im == nil then
      break
   end
   timer = torch.Timer()
   local input = prepareInput(geometry, last_im, im)
   local moutput = model:forward(input)
   local output = processOutput(geometry, moutput)
   print(timer:time())
   if opt.display_output then
      output_window = image.display{image=output.full, win=output_window}
   end
   last_im = im
end