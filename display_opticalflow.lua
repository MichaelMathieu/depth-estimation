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
op:option{'-p', '--post-process-win-size', action='store', dest='post_process_winsize',
	  default=1,
	  help='Basic output post-processing window size (1 disables post-processing)'}

opt=op:parse()
opt.nThreads = tonumber(opt.nThreads)
opt.post_process_winsize = tonumber(opt.post_process_winsize)

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
   if (type(output) == 'table') or (output:size():size(1) == 3) then
      local outputWithBorder = torch.Tensor(2, gt:size(2), gt:size(3)):zero()
   else
      local outputWithBorder = torch.Tensor(gt:size()):fill(init_value)
      outputWithBorder:sub(math.ceil(geometry.hKernel/2),
			   output:size(1)+math.ceil(geometry.hKernel/2)-1,
			   math.ceil(geometry.wKernel/2),
			   output:size(2)+math.ceil(geometry.wKernel/2)-1):copy(output)
      image.display{image={outputWithBorder, gt}, min=1, max=17}
   end
end

geometry, model = loadModel(opt.input_model, true)

local delta = tonumber(opt.input_image2) - tonumber(opt.input_image1)
local image1 = loadImageOpticalFlow(geometry, 'data/', opt.input_image1, nil, nil)
local image2,gt = loadImageOpticalFlow(geometry, 'data/', opt.input_image2,
				       opt.input_image1, delta)
image.display{image1, image2}
local input = {image1, image2}

t = torch.Timer()
local output = model:forward(input)
print(t:time())
output = processOutput(geometry, output)

local output2 = torch.Tensor(2, output.x:size(1), output.x:size(2)):zero()

if opt.post_process_winsize ~= 1 then
   local winsize = opt.post_process_winsize
   local winsizeh1 = math.ceil(winsize/2)-1
   local winsizeh2 = math.floor(winsize/2)
   local win = torch.Tensor(2,winsize,winsize)
   for i = 1+winsizeh1,output2:size(2)-winsizeh2 do
      for j = 1+winsizeh1,output2:size(3)-winsizeh2 do
	 win[1] = (output.y:sub(i-winsizeh1, i+winsizeh2, j-winsizeh1, j+winsizeh2)+0.5):floor()
	 win[2] = (output.x:sub(i-winsizeh1, i+winsizeh2, j-winsizeh1, j+winsizeh2)+0.5):floor()
	 local win2 = win:reshape(2, winsize*winsize)
	 win2 = win2:sort(2)
	 local t = 1
	 local tbest = 1
	 local nbest = 1
	 for k = 2,9 do
	    if (win2:select(2,k) - win2:select(2,t)):abs():sum(1)[1] < 0.5 then
	       if k-t > nbest then
		  nbest = k-t
		  tbest = t
	       end
	    else
	       t = k
	    end
	 end
	 output2[1][i][j] = win2[1][tbest]
	 output2[2][i][j] = win2[2][tbest]
      end
   end
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
displayResult(geometry, output2, gt)