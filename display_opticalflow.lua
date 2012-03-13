require 'torch'
require 'xlua'
require 'sys'
require 'image'
require 'nnx'
require 'opticalflow_model'

op = xlua.OptionParser('%prog [options]')
op:option{'-i1', '--input-image1', action='store', dest='input_image1',
	  help='First image for the optical flow'}
op:option{'-i2', '--input-image2', action='store', dest='input_image2',
	  help='Second image for the optical flow'}
op:option{'-i', '--input-model', action='store', dest='input_model',
	  help='Trained convnet'}
op:option{'-nt', '--num-threads', action='store', dest='nThreads', default=2,
	  help='Number of threads used'}

opt=op:parse()
opt.nThreads = tonumber(opt.nThreads)

torch.manualSeed(1)

if opt.nThreads > 1 then
   require 'openmp'
   openmp.setDefaultNumThreads(opt.nThreads)
end


local loaded = torch.load(opt.input_model)
local geometry = loaded[2]
local model = getModel(geometry, true)
local parameters = model:getParameters()
parameters:copy(loaded[1])

image1 = image.scale(image.loadJPG(opt.input_image1), geometry.wImg, geometry.hImg)
image2 = image.scale(image.loadJPG(opt.input_image2), geometry.wImg, geometry.hImg)

input = {
   image1:narrow(2, math.ceil(geometry.maxh/2), image1:size(2)-geometry.maxh+1)
         :narrow(3, math.ceil(geometry.maxw/2), image2:size(3)-geometry.maxw+1),
   image2
      }

output = model:forward(input)
output = processOutput(geometry, output)

image.display{image=output.x}

