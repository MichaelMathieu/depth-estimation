require 'score_opticalflow'
require 'xlua'

op = xlua.OptionParser('%prog [options]')
op:option{'-fi', '--first-image', action='store', dest='first_image', default=0,
	  help='Index of first image used'}
op:option{'-d', '--delta', action='store', dest='delta', default=1,
	  help='Delta between two consecutive frames'}
op:option{'-ni', '--num-input-images', action='store', dest='num_input_images',
	  default=10, help='Number of annotated images used'}
op:option{'-s', '--save', action='store', dest='save', default = nil,
	  help = 'If a path is specified, then the scores are saved'}
opt = op:parse()
opt.first_image = tonumber(opt.first_image)
opt.delta = tonumber(opt.delta)
opt.num_input_images = tonumber(opt.num_input_images)

geometry = {}
geometry.wImg = 320
geometry.hImg = 180
geometry.maxhGT = 16
geometry.maxwGT = 16
geometry.hKernelGT = 16
geometry.wKernelGT = 16

local raw_data = loadDataOpticalFlow(geometry, 'data/', opt.num_input_images,
				     opt.first_image, opt.delta, false)

local scores = getLearningScores('models_test', raw_data, 'full', 100)

if opt.save then
   torch.save(opt.save, scores)
else
   getLearningCurve(scores)
end
