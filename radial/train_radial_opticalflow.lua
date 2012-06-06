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
networkp.wInput = 320
networkp.hInput = round(networkp.wInput*calibrationp.hImg/calibrationp.wImg)
networkp.layers = {{3,5,5,4}, 'tanh', {4,11,11,10}}

local learningp = {}
learningp.first_image = opt.first_image
learningp.delta = opt.delta
learningp.n_images = opt.num_input_images
learningp.rate = opt.learning_rate
learningp.rate_decay = opt.learning_rate_decay
learningp.weight_decay = opt.weight_decay
learningp.n_train_set = opt.n_train_set

local groundtruthp = {}
groundtruthp.type = 'liu'
groundtruthp.wGT = networkp.wInput
groundtruthp.hGT = networkp.hInput
groundtruthp.delta = learningp.delta
--groundtruthp.params = {hWin = 16, wWin = 16,
-- 		       hKer = 16, wKer = 16}

optim_config = {learningRate = learningp.rate,
		weightDecay = learningp.weight_decay,
		momentum = 0,
		learningRateDecay = learningp.rate_decay}

local raw_data = load_training_raw_data(opt.root_directory, networkp, groundtruthp,
					learningp, calibrationp)

--[[
for iEpoch = 1,opt.n_epochs do
      print('Epoch ' .. iEpoch .. '/' .. opt.n_epochs)
      local train_set = generateDataOpticalFlow(..., raw_data)
      
      for iTrainSet = 1,train_Set:size() do
	 
	 --]]