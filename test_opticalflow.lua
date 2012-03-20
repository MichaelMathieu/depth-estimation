require 'torch'
require 'xlua'
require 'sys'
require 'image'
require 'nnx'
require 'opticalflow_model'
require 'groundtruth_opticalflow'

torch.manualSeed(1)

op = xlua.OptionParser('%prog [options]')
op:option{'-i', '--input-model', action='store', dest='input_model',
	  help='Trained convnet'}
op:option{'-nt', '--num-threads', action='store', dest='nThreads', default=2,
	  help='Number of threads used'}
op:option{'-p', '--post-process-win-size', action='store', dest='post_process_winsize',
	  default=1,
	  help='Basic output post-processing window size (1 disables post-processing)'}
-- input (images)
op:option{'-fi', '--first-image', action='store', dest='first_image', default=0,
	  help='Index of first image used'}
op:option{'-d', '--delta', action='store', dest='delta', default=2,
	  help='Delta between two consecutive frames'}
op:option{'-ni', '--num-input-images', action='store', dest='num_input_images',
	  default=10, help='Number of annotated images used'}

opt=op:parse()
opt.nThreads = tonumber(opt.nThreads)
opt.post_process_winsize = tonumber(opt.post_process_winsize)

opt.first_image = tonumber(opt.first_image)
opt.delta = tonumber(opt.delta)
opt.num_input_images = tonumber(opt.num_input_images)

require 'openmp'
openmp.setDefaultNumThreads(opt.nThreads)

geometry, model = loadModel(opt.input_model, true)

local imagenames = {}
for i = 1,opt.num_input_images do
   imagenames[i] = string.format('%09d', opt.first_image+(i-1)*opt.delta)
end
images = {}
gt = {}
local images = {}
images[1] = loadImageOpticalFlow(geometry, 'data/', imagenames[1], nil, nil)
for i = 2,#imagenames do
   images[i],gt[i-1] = loadImageOpticalFlow(geometry, 'data/', imagenames[i],
					    imagenames[i-1], opt.delta)
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
   return image.hsl2rgb(todisplay)
end

for i = 2,#images do
   local input = prepareInput(geometry, images[i-1], images[i])
   local output = model:forward(input)
   output = processOutput(geometry, output)
   if opt.post_process_winsize ~= 1 then
      output = postProcessImage(output.full, opt.post_process_winsize)
   else
      output = output.full
   end


   
   local im = flow2hsv(geometry, output)
   local gt = flow2hsv(geometry, gt[i-1])
   local im2 = torch.Tensor(im:size(1), 2*im:size(2), 2*im:size(3))
   im2:sub(1,im2:size(1),            1, im:size(2),            1, im:size(3)):copy(images[i-1])
   im2:sub(1,im2:size(1),            1, im:size(2), im:size(3)+1,im2:size(3)):copy(images[i])
   im2:sub(1,im2:size(1), im:size(2)+1,im2:size(2),            1, im:size(3)):copy(im)  
   im2:sub(1,im2:size(1), im:size(2)+1,im2:size(2), im:size(3)+1,im2:size(3)):copy(gt)
   image.save(string.format('dump/%09d.png', i-1), im2)
   --a = image.display{image=flow2hsv(geometry, output.full), win=a, min=0, max=1}
   --a = image.display{image=output.full, win=a, min=1, max=17}
end