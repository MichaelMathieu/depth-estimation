require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
require 'image'
require 'groundtruth_opticalflow'
require 'opticalflow_model_io'

require 'xlua'
require 'download_model'

op = xlua.OptionParser('%prog [options]')
op:option{'-i', '--input-model', action='store', dest='input_model', default=nil,
	  help='Trained convnet, this option isn\'t used if -dldir is used'}
op:option{'-dldir', '--download-dir', action='store', dest='download_dir', default=nil,
	  help='scp command to the models folder (eg. mfm352@access.cims.nyu.edu:depth-estimation/models)'}
--op:option{'-rd', '--root-directory', action='store', dest='root_directory',
--	  default='./data/', help='Root dataset directory'}
opt=op:parse()

if opt.input_model then input_model = opt.input_model end
if opt.download_dir then input_model = downloadModel(opt.download_dir) end

local correction = {}
correction.motion_correction = 'sfm'
correction.wImg = 640
correction.hImg = 480
correction.bad_image_threshold = 0.2
correction.K = torch.FloatTensor(3,3):zero()
correction.K[1][1] = 293.824707
correction.K[2][2] = 310.435730
correction.K[1][3] = 300.631012
correction.K[2][3] = 251.624924
correction.K[3][3] = 1.0
correction.distP = torch.FloatTensor(5)
correction.distP[1] = -0.379940
correction.distP[2] = 0.212737
correction.distP[3] = 0.003098
correction.distP[4] = 0.000870
correction.distP[5] = -0.069770

input_model = input_model or 'model'
local loaded = loadModel(input_model, true, false)
local filter = loaded.filter
local model = loaded.model
model.modules[5] = nn.SoftMax()
local geometry = loaded.geometry
geometry.prefilter = false
geometry.training_mode = false
geometry.motion_correction = correction.motion_correction

local learning = {}
learning.delta = 1
learning.groundtruth = 'liu'

function scoreOpticalFlow(flow, confs, gt)
   local y = (flow[1] - gt[1]):cmul(confs)
   local x = (flow[2] - gt[2]):cmul(confs)
   local nConfs = confs:sum()
   local nErrs = ((x:abs() + y:abs()):gt(1)):sum()
   --local nYErrs = (y:abs():gt(0)):sum()
   --print(nConfs, nErrs)
   local colorflowx = torch.Tensor(3, flow:size(2), flow:size(3)):zero()
   local colorflowy = torch.Tensor(3, flow:size(2), flow:size(3)):zero()
   colorflowx[1]:copy(flow[1])
   colorflowy[1]:copy(flow[2])
   colorflowx[2] = -confs+1
   colorflowy[2] = -confs+1
   --image.display{image={y,x}, legend="diff"}
   --image.display{image=gt, legend="gt"}
   --image.display{image={colorflowy, colorflowx}, legend="output"}
   return nConfs, nErrs
end

local last_im, warped_im, warped_mask, im, gt = loadRectifiedImageOpticalFlow2(
   correction, geometry, learning, 'data/ardrone1/', '000000015', '000000014')

local input = prepareInput(geometry, warped_im, im)
local moutput = model:forward(input)
for k = 0,10,0.1 do
   local poutput = processOutput(geometry, moutput, true, k)
   poutput.full[1]:cmul(poutput.full_confidences)
   poutput.full[2]:cmul(poutput.full_confidences)
   local total, errs = scoreOpticalFlow(poutput.full, poutput.full_confidences, gt)
   print(k, total, errs/total)
end