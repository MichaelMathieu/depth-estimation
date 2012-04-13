require 'score_opticalflow'
require 'xlua'
require 'common'

op = xlua.OptionParser('%prog [options]')
op:option{'-i', '--input', action='store', dest='input', default=nil,
	  help='Epochs directory'}
op:option{'-imd', '--input-models-dir', action='store', dest='input_models_dir', default=nil,
	  help='Instead of computing the curves for a single network, finds all the results in the folder. Overrides option -i'}
op:option{'-fi', '--first-image', action='store', dest='first_image', default=0,
	  help='Index of first image used'}
op:option{'-d', '--delta', action='store', dest='delta', default=1,
	  help='Delta between two consecutive frames'}
op:option{'-ni', '--num-input-images', action='store', dest='num_input_images',
	  default=10, help='Number of annotated images used'}
op:option{'-s', '--save', action='store', dest='save', default = nil,
	  help = 'If a folder is specified, then the scores are saved inside'}
op:option{'-l', '--load', action='store', dest='load', default = nil,
	  help = 'If a path is specified, then the scores are loaded from the path'}
opt = op:parse()
opt.first_image = tonumber(opt.first_image)
opt.delta = tonumber(opt.delta)
opt.num_input_images = tonumber(opt.num_input_images)
if opt.save then
   if opt.save:sub(-1) ~= '/' then opt.save = opt.save..'/' end
end

local inputs = {opt.input}
if opt.input_models_dir then
   inputs = lsR(opt.input_models_dir,
		function(a) return false end,
		function(a) local s = split(strip(a,'/'),'/') return s[#s] ~= 'old' end,
		function(a)
		   local s = split(strip(a,'/'),'/')
		   return s[#s-1]:sub(1,1)=='r'
		end)
end

local scores = {}
if not opt.load then
   geometry = {}
   geometry.wImg = 320
   geometry.hImg = 180
   geometry.maxhGT = 16
   geometry.maxwGT = 16
   geometry.hKernelGT = 16
   geometry.wKernelGT = 16

   learning = {}
   learning.num_images = opt.num_input_images
   learning.first_image = opt.first_image
   learning.delta = opt.delta
   learning.groundtruth = 'cross-correlation'
   
   local raw_data = loadDataOpticalFlow(geometry, learning, 'data/')
   
   for iInput = 1,#inputs do
      scores[iInput] = getLearningScores(inputs[iInput], raw_data, 'full', 100)
   end
else
   scores[1] = torch.load(opt.load)
end

if opt.save then
   for iInput = 1,#inputs do
      local name = table.concat(split(inputs[iInput], '/'))
      torch.save(opt.save .. name, scores[iInput])
   end
else
   for iInput = 1,#scores do
      getLearningCurve(scores[iInput])
   end
end
