require 'torch'
require 'xlua'
require 'sys'
require 'image'
require 'nnx'
require 'opticalflow_model'
require 'opticalflow_model_io'
require 'groundtruth_opticalflow'
require 'openmp'
require 'score_opticalflow'
require 'download_model'

torch.manualSeed(1)

op = xlua.OptionParser('%prog [options]')
op:option{'-i1', '--input-image1', action='store', dest='input_image1',
	  help='First image for the optical flow'}
op:option{'-i2', '--input-image2', action='store', dest='input_image2',
	  help='Second image for the optical flow'}
op:option{'-nt', '--num-threads', action='store', dest='nThreads', default=2,
	  help='Number of threads used'}
op:option{'-p', '--post-process-win-size', action='store', dest='post_process_winsize',
	  default=1,
	  help='Basic output post-processing window size (1 disables post-processing)'}
op:option{'-i', '--input-model', action='store', dest='input_model',
	  help='Trained convnet, this option isn\'t used if -dldir is used'}
op:option{'-dldir', '--download-dir', action='store', dest='download_dir', default=nil,
	  help='scp command to the models folder (eg. mfm352@access.cims.nyu.edu:/depth-estimation/models)'}
op:option{'-rd', '--root-directory', action='store', dest='root_directory',
	  default='./data/', help='Root dataset directory'}
op:option{'-lg', '--liu-grountruth', action='store_true', dest='use_liu_groundtruth',
	  default=false, help='Use Liu groundtruth'}

opt=op:parse()
opt.nThreads = tonumber(opt.nThreads)
opt.post_process_winsize = tonumber(opt.post_process_winsize)
local groundtruth
if not opt.use_liu_groundtruth then
   groundtruth = 'cross-correlation'
else
   groundtruth = 'liu'
end

openmp.setDefaultNumThreads(opt.nThreads)

if opt.download_dir ~= nil then
   opt.input_model = downloadModel(opt.download_dir)
   if opt.input_model == nil then
      os.exit(0)
   end
end

loaded = loadModel(opt.input_model, true)
--loaded = loadModel(opt.input_model, false)
model = loaded.model
geometry = loaded.geometry
kernels = loaded.getKernels(geometry, model)

for i = 1,#kernels do
   if kernels[i]:size(2) > 5 then
      image.display{image=kernels[i], zoom=4, padding=2}
   else
      image.display{image=kernels[i], zoom=8, padding=2}
   end
end

local delta = tonumber(opt.input_image2) - tonumber(opt.input_image1)
local image1, image2, gt
if geometry.motion_correction then
   image1 = loadRectifiedImageOpticalFlow(geometry, opt.root_directory, opt.input_image1,
					  nil, nil)
   _,gt,image2,_ = loadRectifiedImageOpticalFlow(geometry, opt.root_directory,
						 opt.input_image2, opt.input_image1, delta)
else
   image1 = loadImageOpticalFlow(geometry, opt.root_directory, opt.input_image1, nil, nil)
   image2,gt = loadImageOpticalFlow(geometry, opt.root_directory, opt.input_image2,
				    opt.input_image1, delta, groundtruth)
end
local input = {image1, image2}
image.display(input)

t = torch.Timer()
input = prepareInput(geometry, input[1], input[2])

local output = model:forward(input)
--local output = torch.Tensor(112, input[1]:size(2), input[1]:size(3))
--for i = 1,output:size(2) do
--   xlua.progress(i, output:size(2))
--   for j = 1,output:size(3) do
--      model:focus(j, i)
--      output[{{},i,j}] = model:forward(input):squeeze()
--   end
--end
print(t:time())
output = processOutput(geometry, output)

local output2 = torch.Tensor(2, output.x:size(1), output.x:size(2)):zero()

if opt.post_process_winsize ~= 1 then
   output2 = postProcessImage(output.full, opt.post_process_winsize)
else
   output2 = output.full
end

nGood, nNear, nBad, d, meanDst, stdDst = evalOpticalflow(geometry, output2, gt)
print('nGood=' .. nGood .. ' nNear=' .. nNear .. ' nBad=' .. nBad)
print(100.*nGood/(nGood+nNear+nBad) .. '% accurate, ' .. 100.*(nGood+nNear)/(nGood+nNear+nBad) .. '% almost accurate')
print('meanDst=' .. meanDst .. ' (std=' .. stdDst .. ')')

local inity, initx = centered2onebased(geometry, 0, 0)
image.display{output2[1], gt[1]}
image.display{output2[2], gt[2]}
print("--")
local hsv = flow2hsv(geometry, output2)
local gthsv = flow2hsv(geometry, gt)
image.display{hsv, gthsv}
local diff=(output2-gt):abs():sum(1):squeeze():sub(math.ceil(geometry.hPatch2/2),
						   geometry.hImg-math.ceil(geometry.hPatch2/2),
						   math.ceil(geometry.wPatch2/2),
						   geometry.wImg-math.ceil(geometry.wPatch2/2))
local errs = torch.Tensor(diff:size(1), 3*diff:size(2)):fill(0)
errs:sub(1,diff:size(1), 1               , diff:size(2)  ):copy(diff:ge(1))
errs:sub(1,diff:size(1), diff:size(2)+1  , 2*diff:size(2)):copy(diff:ge(2))
errs:sub(1,diff:size(1), 2*diff:size(2)+1, 3*diff:size(2)):copy(diff:ge(3))
image.display(errs)
image.display{image=diff}