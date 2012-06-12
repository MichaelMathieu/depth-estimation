require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
package.path = "./?.lua;../?.lua;" .. package.path
package.cpath = "./?.so;../?.so;" .. package.cpath
require 'xlua'
require 'sys'
require 'nn'
require 'optim'
require 'image'
require 'openmp'
require 'radial_opticalflow_data'
require 'radial_opticalflow_network'
require 'radial_opticalflow_filtering'
require 'radial_opticalflow_polar'
require 'radial_opticalflow_display'
require 'draw'

torch.manualSeed(1)

op = xlua.OptionParser('%prog [options]')
-- general
op:option{'-nt', '--num-threads', action='store', dest='nThreads', default=2,
	  help='Number of threads used'}

-- network
op:option{'-net', '--network', action='store', dest='network_struct',
	  default="{{3,1,17,5},{5,17,1,10}}", help='Network structure'}
op:option{'-hw', '--h-window', action='store', dest='hWin',
	  default=15, help='Height of the search window'}
op:option{'-wi', '--w-intern', action='store', dest='w_input',
	  default=200, help='Width of the intern representation'}
op:option{'-hi', '--h-intern', action='store', dest='h_input',
	  default=200, help='Height of the intern representation'}

-- learning
op:option{'-n', '--n-train-set', action='store', dest='n_train_set', default=2000,
	  help='Number of patches in one single epoch'}
op:option{'-e', '--num-epochs', action='store', dest='n_epochs', default=10,
	  help='Number of epochs'}
op:option{'-r', '--learning-rate', action='store', dest='learning_rate',
          default=5e-3, help='Learning rate'}
op:option{'-lrd', '--learning-rate-decay', action='store', dest='learning_rate_decay',
          default=5e-7, help='Learning rate decay'}
op:option{'-wd', '--weight-decay', action='store', dest='weight_decay',
	  default=0, help='Weight decay'}
op:option{'-crit', '--criterion', action='store', dest='criterion',
	  default='NLL', help='Learning criterion (NLL | MSE)'}

-- input
op:option{'-rd', '--root-directory', action='store', dest='root_directory',
	  default='data/no-risk/part1/', help='Root dataset directory'}
op:option{'-d', '--delta', action='store', dest='delta', default=1,
	  help='Delta between two consecutive frames'}
op:option{'-cal', '--caligration', dest='calibration_file', default='rectified_gopro.cal',
	  action='store', help='Calibration parameters file'}

-- output 
op:option{'-omd', '--output-model-dir', action='store', dest='output_models_dir',
	  default = 'models', help='Output model directory'}
op:option{'-ev', '--evaluate', action='store_true', dest='evaluate',
	  default=false, help='Evaluate the network on the first image'}

opt = op:parse()
opt.nTherads = tonumber(opt.nThreads)
opt.n_train_set = tonumber(opt.n_train_set)
opt.n_images_test_set = tonumber(opt.n_images_test_set)
opt.n_epochs = tonumber(opt.n_epochs)
opt.learning_rate = tonumber(opt.learning_rate)
opt.learning_rate_decay = tonumber(opt.learning_rate_decay)
opt.weight_decay = tonumber(opt.weight_decay)
opt.hWin = tonumber(opt.hWin)
opt.w_input = tonumber(opt.w_input)
opt.h_input = tonumber(opt.h_input)
opt.delta = tonumber(opt.delta)
if opt.root_directory:sub(-1) ~= '/' then opt.root_directory = opt.root_directory .. '/' end

openmp.setDefaultNumThreads(opt.nThreads)

local calibrationp = torch.load(opt.calibration_file)

local networkp = {}
networkp.wImg = 320
networkp.hImg = round(networkp.wImg*calibrationp.hImg/calibrationp.wImg)
networkp.wInput = opt.w_input
networkp.hInput = opt.h_input
networkp.layers = loadstring("return "..opt.network_struct)()
networkp.hWin = opt.hWin
networkp.wKernel = 1
networkp.hKernel = 1
for i = 1,#networkp.layers do
   if type(networkp.layers) == 'table' then
      networkp.hKernel = networkp.hKernel + networkp.layers[i][2]-1
      networkp.wKernel = networkp.wKernel + networkp.layers[i][3]-1
   end
end

local learningp = {}
learningp.delta = opt.delta
learningp.rate = opt.learning_rate
learningp.rate_decay = opt.learning_rate_decay
learningp.weight_decay = opt.weight_decay
learningp.n_train_set = opt.n_train_set
learningp.criterion = opt.criterion

local groundtruthp = {}
groundtruthp.wGT = networkp.wImg
groundtruthp.hGT = networkp.hImg
groundtruthp.delta = learningp.delta
--[[
groundtruthp.type = 'cross-correlation'
groundtruthp.params = {hWin = 17, wWin = 17,
 		       hKer = 17, wKer = 17}
--]]
groundtruthp.type = 'liu'
groundtruthp.params = {alpha = 0.005,
		       ratio = 0.75,
		       minWidth = 60,
		       nOFPIters = 10, nIFPIters = 5,
		       nCGIters = 40}

local optim_config = {learningRate = learningp.rate,
		      weightDecay = learningp.weight_decay,
		      momentum = 0,
		      learningRateDecay = learningp.rate_decay}

local raw_data = load_data(opt.root_directory, networkp, groundtruthp,
			   learningp, calibrationp)

local network = getTrainerNetwork(networkp)

local parameters, gradParameters = network:getParameters()
local criterion
if learningp.criterion == 'NLL' then
   criterion = nn.ClassNLLCriterion()
elseif learningp.criterion == 'MSE' then
   criterion = nn.MSECriterion()
else
   error('Unknown criterion ' .. learningp.criterion)
end

win_kers = {}
local function evaluate(raw_data, network, i)
   testnetwork = getTesterNetwork(networkp)
   copyWeights(network, testnetwork)
   displayWeights(network, win_kers)

   local maskf = raw_data.polar_prev_images_masks[i]
   local mask = torch.LongTensor(maskf:size(1), maskf:size(2)):copy(maskf)
   time = torch.Timer()
   local test = testnetwork:forward({raw_data.polar_prev_images[i],
				     raw_data.polar_images[i]})
				     --mask})
				     
   print(time:time()['real'])
   _,test = test:min(3)
   test:add(-1)
   test = torch.Tensor(test:squeeze():size()):copy(test)*getRMax(networkp, raw_data.e2[i])/networkp.hInput
   local p2cmask = getP2CMaskOF(networkp, raw_data.e2[i])
   win_test = image.display{image=test, win=win_test}
   local h = test:size(1)
   local w = test:size(2)
   test:sub(h,h,1,w):zero()
   win_testcart = image.display{image=cartesian2polar(test, p2cmask), win=win_testcart,
				min = 0, max = raw_data.groundtruth[i]:max()}
   win_gt = image.display{image=raw_data.groundtruth[i], win=win_gt}
   local imcpy = raw_data.images[i]:clone()
   draw.point(imcpy, raw_data.e2[i][1], raw_data.e2[i][2], 3, 1, 0, 0)
   win_imgs = image.display{image={raw_data.prev_images[i], imcpy}, win=win_imgs}
   win_polimgs = image.display{image={raw_data.polar_prev_images[i], raw_data.polar_images[i]},
			       win=win_polimgs}
   local gt = raw_data.polar_groundtruth[i]:clone()
   local gtmask = raw_data.polar_groundtruth_masks[i]
   gt:cmul(gtmask)
   local colored_gt = torch.Tensor(3, gt:size(1), gt:size(2))
   colored_gt[1]:copy(gt)
   colored_gt[2]:copy(gt)
   colored_gt[3]:copy(gt + (-gtmask+1)*gt:max())
   
   win_polgt = image.display{image=colored_gt, win=win_polgt}
end

for iEpoch = 1,opt.n_epochs do
   print('Epoch ' .. iEpoch .. '/' .. opt.n_epochs)
   local train_set = generate_training_patches(raw_data, networkp, learningp)
   local nGood = 0
   local nBad = 0
   local threshold = 0
   --if iEpoch > 1 then
   --threshold = 0.1
--end

   if opt.evaluate then
      evaluate(raw_data, network, 1)
   end
   
   for iTrainSet = 1,train_set:size() do
      modProgress(iTrainSet, train_set:size(), 100)
      
      local input = train_set:getPatch(iTrainSet)
      local target = round(train_set.flow[iTrainSet]+1)
      
      local function feval(x)
	 if x ~= parameters then
	    parameters:copy(x)
	 end
	 gradParameters:zero()
	 local output = network:forward(input)
	 local idx, good = filterOutputTrainer(output, threshold)
	 local err = 0
	 if good or true then
	    local df_do
	    if learningp.criterion == 'NLL' then
	       err = criterion:forward(output, target)
	       df_do = criterion:backward(output, target)
	    elseif learningp.criterion == 'MSE' then
	       local target_idx = torch.Tensor(1):fill(idx)
	       local target_crit = torch.Tensor(1):fill(target)
	       err = criterion:forward(target_idx, target_crit)
	       df_do = criterion:backward(target_idx, target_crit)
	    end
	    network:backward(input, df_do)
	    if idx == target then
	       nGood = nGood + 1
	    else
	       nBad = nBad + 1
	    end
	 end
	 return err, gradParameters
      end
      
      optim.sgd(feval, parameters, optim_config)
      
   end
   print(nGood, nBad)
   collectgarbage()
   saveNetwork('models', iEpoch, networkp, network)
end