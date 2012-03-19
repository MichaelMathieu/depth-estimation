require 'torch'
require 'xlua'
require 'sys'
require 'image'
require 'nnx'
require 'opticalflow_model'
require 'groundtruth_opticalflow'
require 'openmp'

torch.manualSeed(1)

op = xlua.OptionParser('%prog [options]')
op:option{'-i1', '--input-image1', action='store', dest='input_image1',
	  help='First image for the optical flow'}
op:option{'-i2', '--input-image2', action='store', dest='input_image2',
	  help='Second image for the optical flow'}
op:option{'-i', '--input-model', action='store', dest='input_model',
	  help='Trained convnet'}
op:option{'-nt', '--num-threads', action='store', dest='nThreads', default=2,
	  help='Number of threads used'}
op:option{'-p', '--post-process-win-size', action='store', dest='post_process_winsize',
	  default=1,
	  help='Basic output post-processing window size (1 disables post-processing)'}

opt=op:parse()
opt.nThreads = tonumber(opt.nThreads)
opt.post_process_winsize = tonumber(opt.post_process_winsize)

openmp.setDefaultNumThreads(opt.nThreads)

function flow2hsv(geometry, flow)
   local todisplay = torch.Tensor(3, flow:size(2), flow:size(3))
   for i = 1,flow:size(2) do
      for j = 1,flow:size(3) do
	 local y, x = onebased2centered(geometry, flow[1][i][j], flow[2][i][j])
	 local ang = math.atan2(y, x)
	 local norm = math.sqrt(x*x+y*y)
	 todisplay[1][i][j] = ang/(math.pi*2.0)
	 todisplay[2][i][j] = 1.0
	 todisplay[3][i][j] = norm/math.max(geometry.maxh/2, geometry.maxw/2)
      end
   end
   return image.hsv2rgb(todisplay)
end

function displayResult(geometry, output, gt, init_value)
   init_value = init_value or 0
   local outputWithBorder
   if output:size():size(1) == 3 then
      outputWithBorder = torch.Tensor(3, gt:size(2), gt:size(3)):zero()
      outputWithBorder:sub(1,3, math.ceil(geometry.hKernel/2),
			   output:size(2)+math.ceil(geometry.hKernel/2)-1,
			   math.ceil(geometry.wKernel/2),
			   output:size(3)+math.ceil(geometry.wKernel/2)-1):copy(output)
   else
      outputWithBorder = torch.Tensor(gt:size()):fill(init_value)
      outputWithBorder:sub(math.ceil(geometry.hKernel/2),
			   output:size(1)+math.ceil(geometry.hKernel/2)-1,
			   math.ceil(geometry.wKernel/2),
			   output:size(2)+math.ceil(geometry.wKernel/2)-1):copy(output)
   end
   image.display{outputWithBorder, gt}
end

geometry, model = loadModel(opt.input_model, true)

local delta = tonumber(opt.input_image2) - tonumber(opt.input_image1)
local image1 = loadImageOpticalFlow(geometry, 'data/', opt.input_image1, nil, nil)
local image2,gt = loadImageOpticalFlow(geometry, 'data/', opt.input_image2,
				       opt.input_image1, delta)
local input = {image1, image2}
image.display(input)

t = torch.Timer()
print(geometry)
input = prepareInput(geometry, input[1], input[2])
local output = model:forward(input)
print(t:time())
output = processOutput(geometry, output)

local output2 = torch.Tensor(2, output.x:size(1), output.x:size(2)):zero()

if opt.post_process_winsize ~= 1 then
   output2 = postProcessImage({output.y, output.x}, opt.post_process_winsize)
else
   output2[1] = output.y
   output2[2] = output.x
end

local gt2 = gt:sub(1,2,
		   math.ceil(geometry.hKernel/2),
		   output.x:size(1)+math.ceil(geometry.hKernel/2)-1,
		   math.ceil(geometry.wKernel/2),
		   output.x:size(2)+math.ceil(geometry.wKernel/2)-1)

local diff = (output2 - gt2):abs()
diff = diff[1]+diff[2]
local nGood = 0
local nNear = 0
local nBad = 0
for i = 1,diff:size(1) do
   for j = 1,diff:size(2) do
      if diff[i][j] == 0 then
	 nGood = nGood + 1
      elseif diff[i][j] == 1 then
	 nNear = nNear + 1
      else
	 nBad = nBad + 1
      end
   end
end
print('nGood=' .. nGood .. ' nNear=' .. nNear .. ' nBad=' .. nBad)
print(100.*nGood/(nGood+nNear+nBad) .. '% accurate, ' .. 100.*(nGood+nNear)/(nGood+nNear+nBad) .. '% almost accurate')

local inity, initx = centered2onebased(geometry, 0, 0)
displayResult(geometry, output2[1], gt[1], inity)
displayResult(geometry, output2[2], gt[2], initx)
print("--")
local hsv = flow2hsv(geometry, output2)
local gthsv = flow2hsv(geometry, gt)
displayResult(geometry, hsv, gthsv)