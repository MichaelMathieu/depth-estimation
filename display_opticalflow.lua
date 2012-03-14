require 'torch'
require 'xlua'
require 'sys'
require 'image'
require 'nnx'
require 'opticalflow_model'
require 'groundtruth_opticalflow'

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

function flow2hsv(geometry, flow)
   local todisplay = torch.Tensor(3, flow:size(2), flow:size(3))
   for i = 1,flow:size(2) do
      for j = 1,flow:size(3) do
	 local y, x = onebased2centered(geometry, flow[1][i][j], flow[2][i][j])
	 local ang = math.atan2(y, x)
	 local norm = math.sqrt(flow[1][i][j]*flow[1][i][j]+flow[2][i][j]*flow[2][i][j])
	 todisplay[1][i][j] = ang/(math.pi*2.0)
	 todisplay[2][i][j] = norm/math.max(geometry.maxh, geometry.maxw)
	 todisplay[3][i][j] = 1.0
      end
   end
   return image.hsv2rgb(todisplay)
end

function displayResult(geometry, output, gt, init_value)
   init_value = init_value or 0
   if (type(output) == 'table') or (output:size():size(1) == 3) then
      local outputWithBorder = torch.Tensor(2, gt:size(2), gt:size(3)):zero()
      if type(output) == 'table' then
	 for i = 1,2 do
	    outputWithBorder:sub(i, i,
				 math.ceil(geometry.hKernel/2),
				 output[1]:size(1)+math.ceil(geometry.hKernel/2)-1,
				 math.ceil(geometry.wKernel/2),
				 output[1]:size(2)+math.ceil(geometry.wKernel/2)-1):copy(output[i])
	 end
      else
	 outputWithBorder:sub(1, 2,
			      math.ceil(geometry.hKernel/2),
			      output:size(2)+math.ceil(geometry.hKernel/2)-1,
			      math.ceil(geometry.wKernel/2),
			      output:size(3)+math.ceil(geometry.wKernel/2)-1):copy(output)
      end
      image.display{flow2hsv(geometry, outputWithBorder), flow2hsv(geometry, gt)}
   else
      local outputWithBorder = torch.Tensor(gt:size()):fill(init_value)
      outputWithBorder:sub(math.ceil(geometry.hKernel/2),
			   output:size(1)+math.ceil(geometry.hKernel/2)-1,
			   math.ceil(geometry.wKernel/2),
			   output:size(2)+math.ceil(geometry.wKernel/2)-1):copy(output)
      image.display{outputWithBorder, gt}
   end
end

local loaded = torch.load(opt.input_model)
local geometry = loaded[2]
local model = getModel(geometry, true)
local parameters = model:getParameters()
parameters:copy(loaded[1])

local delta = tonumber(opt.input_image2) - tonumber(opt.input_image1)
local image1 = loadImageOpticalFlow(geometry, 'data/', opt.input_image1, nil, nil)
local image2,gt = loadImageOpticalFlow(geometry, 'data/', opt.input_image2,
				       opt.input_image1, delta)

local input = {image1, image2}

local output = model:forward(input)
output = processOutput(geometry, output)

local inity, initx = centered2onebased(geometry, 0, 0)
displayResult(geometry, output.y, gt[1], inity)
displayResult(geometry, output.x, gt[2], initx)
displayResult(geometry, {output.y, output.x}, gt)