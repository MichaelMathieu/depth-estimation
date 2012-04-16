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
	  help = 'If a folder is specified, then the scores are loaded from the folder'}
op:option{'-xmax', '--x-max', action='store', dest='xmax', default=nil,
	  help = 'Crop to xmax'}
op:option{'-p', '--plot', action='store_true', dest='plot', default = false,
	  help = 'Plot curves'}
op:option{'-fix', '--fix', action='store_true', dest='fix_old', default=false,
	  help='Fix old files'}
op:option{'-rd', '--root-directory', action='store', dest='root_directory',
	  default='./data', help='Root dataset directory'}
opt = op:parse()
opt.first_image = tonumber(opt.first_image)
opt.delta = tonumber(opt.delta)
opt.num_input_images = tonumber(opt.num_input_images)
opt.xmax = tonumber(opt.xmax)
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
   
   local raw_data = loadDataOpticalFlow(geometry, learning, opt.root_directory)
   
   for iInput = 1,#inputs do
      local name = table.concat(split(inputs[iInput], '/'))
      print(name)
      scores[iInput] = {name, getLearningScores(inputs[iInput], raw_data, 'full',
						100, opt.fix_old)}
      if opt.save then
	 torch.save(opt.save .. scores[iInput][1], scores[iInput][2])
      end
   end
else
   local paths = lsR(opt.load,
		     function(a) return true end,
		     function(a) return false end,
		     function(a) return false end)
   for i = 1,#paths do
      scores[i] = {paths[i], torch.load(paths[i])}
   end
end

if opt.plot then
   if opt.xmax then
      for i = 1,#scores do
	 if #scores[i][2] > opt.xmax then
	    local tmp = {}
	    for j = 1,opt.xmax do
	       tmp[j] = scores[i][2][j]
	    end
	    scores[i][2] = tmp
	 end
      end
   end
   getLearningCurve(scores)
end
