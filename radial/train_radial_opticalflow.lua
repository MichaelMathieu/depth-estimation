require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
package.path = "./?.lua;../?.lua;" .. package.path
package.cpath = "./?.so;../?.so;" .. package.cpath
require 'xlua'
require 'optim'
require 'sys'
require 'openmp'
require 'radial_opticalflow_data'
require 'image'
require 'radial_opticalflow_network'
require 'nn'

torch.manualSeed(1)

op = xlua.OptionParser('%prog [options]')
-- general
op:option{'-nt', '--num-threads', action='store', dest='nThreads', default=2,
	  help='Number of threads used'}

-- network

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
op:option{'-fi', '--first-image', action='store', dest='first_image', default=0,
	  help='Index of first image used'}
op:option{'-d', '--delta', action='store', dest='delta', default=1,
	  help='Delta between two consecutive frames'}
op:option{'-ni', '--num-input-images', action='store', dest='num_input_images',
	  default=10, help='Number of annotated images used'}
op:option{'-cal', '--caligration', dest='calibration_file', default='rectified_gopro.cal',
	  action='store', help='Calibration parameters file'}

-- output 
op:option{'-omd', '--output-model-dir', action='store', dest='output_models_dir',
	  default = 'models', help='Output model directory'}

opt = op:parse()
opt.nTherads = tonumber(opt.nThreads)
opt.n_train_set = tonumber(opt.n_train_set)
opt.n_images_test_set = tonumber(opt.n_images_test_set)
opt.n_epochs = tonumber(opt.n_epochs)
opt.learning_rate = tonumber(opt.learning_rate)
opt.learning_rate_decay = tonumber(opt.learning_rate_decay)
opt.weight_decay = tonumber(opt.weight_decay)
opt.first_image = tonumber(opt.first_image)
opt.delta = tonumber(opt.delta)
opt.num_input_images = tonumber(opt.num_input_images)
if opt.root_directory:sub(-1) ~= '/' then opt.root_directory = opt.root_directory .. '/' end

openmp.setDefaultNumThreads(opt.nThreads)

local calibrationp = torch.load(opt.calibration_file)

local networkp = {}
networkp.wImg = 320
networkp.hImg = round(networkp.wImg*calibrationp.hImg/calibrationp.wImg)
--networkp.hImg = 320
networkp.wInput = 200
networkp.hInput = 200
networkp.layers = {{3,1,7,5}, {5, 17, 1, 10}}
networkp.hWin = 16
networkp.wKernel = 7
networkp.hKernel = 17

local learningp = {}
learningp.first_image = opt.first_image
learningp.delta = opt.delta
learningp.n_images = opt.num_input_images
learningp.rate = opt.learning_rate
learningp.rate_decay = opt.learning_rate_decay
learningp.weight_decay = opt.weight_decay
learningp.n_train_set = opt.n_train_set
learningp.criterion = opt.criterion

local groundtruthp = {}
groundtruthp.type = 'cross-correlation'
groundtruthp.wGT = networkp.wImg
groundtruthp.hGT = networkp.hImg
--groundtruthp.hGT = 180
groundtruthp.delta = learningp.delta
groundtruthp.params = {hWin = 17, wWin = 17,
 		       hKer = 17, wKer = 17}

local optim_config = {learningRate = learningp.rate,
		      weightDecay = learningp.weight_decay,
		      momentum = 0,
		      learningRateDecay = learningp.rate_decay}

local raw_data = load_training_raw_data(opt.root_directory, networkp, groundtruthp,
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
   local testlayers = testnetwork.modules[1].modules[2].modules
   local layers = network.modules[1].modules[2].modules
   for i = 1,#layers do
      if layers[i].weight then
	 testlayers[i].weight:copy(layers[i].weight)
	 w = layers[i].weight
	 if w:size(2) ~= 3 then
	    w = w:reshape(w:size(1)*w:size(2), w:size(3), w:size(4))
	 end
	 win_kers[i] = image.display{image=w,padding=2,zoom=4,win=win_kers[i]}
      end
      if layers[i].bias then
	 testlayers[i].bias:copy(layers[i].bias)
      end
   end
   time = torch.Timer()
   local test = testnetwork:forward({raw_data.polar_prev_images[i],
				     raw_data.polar_images[i]})
   print(time:time()['real'])
   _,test = test:min(3)
   test = test-1
   test = torch.Tensor(test:squeeze():size()):copy(test)*160/networkp.hInput
   local p2cmask = getP2CMask(test:size(2), test:size(1),
			      networkp.wImg-33/200*160, networkp.hImg-33/200*160,
			      699/4, 238/4, 160)
   win_test = image.display{image=test, win=win_test, min=0, max=12}
   local h = test:size(1)
   local w = test:size(2)
   test:sub(h,h,1,w):zero()
   win_testcart = image.display{image=cartesian2polar(test, p2cmask), win=win_testcart}
   win_gt = image.display{image=raw_data.groundtruth[i], win=win_gt}
end

for iEpoch = 1,opt.n_epochs do
   print('Epoch ' .. iEpoch .. '/' .. opt.n_epochs)
   local train_set = generate_training_patches(raw_data, networkp, learningp)
   local nGood = 0
   local nBad = 0

   --evaluate(raw_data, network, 3)
   
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
	 local _, idx = output:max(1)
	 local err, df_do
	 if learningp.criterion == 'NLL' then
	    err = criterion:forward(output, target)
	    df_do = criterion:backward(output, target)
	 elseif learningp.criterion == 'MSE' then
	    local target_idx = torch.Tensor(1):copy(idx)
	    local target_crit = torch.Tensor(1):fill(target)
	    err = criterion:forward(target_idx, target_crit)
	    df_do = criterion:backward(target_idx, target_crit)
	 end
	 network:backward(input, df_do)
	 idx = idx:squeeze()
	 if idx == target then
	    nGood = nGood + 1
	 else
	    nBad = nBad + 1
	 end
	 return err, gradParameters
      end
      
      optim.sgd(feval, parameters, optim_config)
      
   end
   print(nGood, nBad)
   collectgarbage()
end