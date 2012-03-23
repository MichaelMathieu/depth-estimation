require 'torch'
require 'xlua'
require 'nnx'
require 'image'
require 'optim'
require 'load_data'
require 'groundtruth_opticalflow'
require 'opticalflow_model'
require 'sys'
require 'openmp'

torch.manualSeed(1)

op = xlua.OptionParser('%prog [options]')
-- general
op:option{'-nt', '--num-threads', action='store', dest='nThreads', default=2,
	  help='Number of threads used'}
-- network
op:option{'-nf', '--n-features', action='store', dest='n_features',
          default=10, help='Number of features in the first layer'}
op:option{'-k1s', '--kernel1-size', action='store', dest='kernel1_size',
	  default=5, help='Kernel 1 size, if ns == two_layers'}
op:option{'-k2s', '--kernel2-size', action='store', dest='kernel2_size',
	  default=16, help='Kernel 2 size'}
op:option{'-ws', '--window-size', action='store', dest='win_size',
	  default=17, help='Window size (maxh)'}
op:option{'-ns', '-network-structure', action='store', dest='network_structure',
	  default='one_layer', help='Network structure (one_layer | two_layers)'}
op:option{'-s2', '--layer-two-size', action='store', dest='layer_two_size', default=8,
	  help='Second (hidden) layer size, if ns == two_layers'}
op:option{'-s2c', '--layer-two-connections', action='store', dest='layer_two_connections',
	  default=4, help='Number of connectons between layers 1 and 2'}
op:option{'-l2', '--l2-pooling', action='store_true', dest='l2_pooling', default=false,
	  help='L2 pooling'}
op:option{'-ms', '--multiscale', action='store_true', dest='multiscale', default=false,
	  help='Use multiscale (experimental)'}
-- learning
op:option{'-n', '--n-train-set', action='store', dest='n_train_set', default=2000,
	  help='Number of patches in the training set'}
op:option{'-m', '--n-test-set', action='store', dest='n_test_set', default=1000,
	  help='Number of patches in the test set'}
op:option{'-e', '--num-epochs', action='store', dest='n_epochs', default=10,
	  help='Number of epochs'}
op:option{'-r', '--learning-rate', action='store', dest='learning_rate',
          default=5e-3, help='Learning rate'}
op:option{'-lrd', '--learning-rate-decay', action='store', dest='learning_rate_decay',
          default=5e-7, help='Learning rate decay'}
op:option{'-st', '--soft-targets', action='store_true', dest='soft_targets', default=false,
	  help='Enable soft targets (targets are gaussians centered on groundtruth)'}
op:option{'-s', '--sampling-method', action='store', dest='sampling_method',
	  default='uniform_position', help='Sampling method. uniform_position | uniform_flow'}
op:option{'-wd', '--weight-decay', action='store', dest='weight_decay',
	  default=0, help='Weight decay'}
-- input
op:option{'-rd', '--root-directory', action='store', dest='root_directory',
	  default='./data', help='Root dataset directory'}
op:option{'-fi', '--first-image', action='store', dest='first_image', default=0,
	  help='Index of first image used'}
op:option{'-d', '--delta', action='store', dest='delta', default=2,
	  help='Delta between two consecutive frames'}
op:option{'-ni', '--num-input-images', action='store', dest='num_input_images',
	  default=10, help='Number of annotated images used'}
op:option{'-mc', '--motion-correction', action='store_true', dest='motion_correction', default=false,
     help='Eliminate panning, tilting and rotation camera movements'}

opt=op:parse()
opt.nThreads = tonumber(opt.nThreads)

opt.n_features = tonumber(opt.n_features)
opt.kernel2_size = tonumber(opt.kernel2_size)
opt.kernel1_size = tonumber(opt.kernel1_size)
opt.win_size = tonumber(opt.win_size)
opt.layer_two_size = tonumber(opt.layer_two_size)
opt.layer_two_connections = tonumber(opt.layer_two_connections)

opt.n_train_set = tonumber(opt.n_train_set)
opt.n_test_set = tonumber(opt.n_test_set)
opt.n_epochs = tonumber(opt.n_epochs)
opt.learning_rate = tonumber(opt.learning_rate)
opt.learning_rate_decay = tonumber(opt.learning_rate_decay)
opt.weight_decay = tonumber(opt.weight_decay)

opt.first_image = tonumber(opt.first_image)
opt.delta = tonumber(opt.delta)
opt.num_input_images = tonumber(opt.num_input_images)

openmp.setDefaultNumThreads(opt.nThreads)

local geometry = {}
geometry.wImg = 320
geometry.hImg = 180
geometry.maxw = opt.win_size
geometry.maxh = opt.win_size
geometry.wKernelGT = 16
geometry.hKernelGT = 16
if opt.network_structure == 'two_layers' then
   geometry.wKernel1 = opt.kernel1_size
   geometry.hKernel1 = opt.kernel1_size
   geometry.wKernel2 = opt.kernel2_size
   geometry.hKernel2 = opt.kernel2_size
   geometry.wKernel = geometry.wKernel1 + geometry.wKernel2 - 1
   geometry.hKernel = geometry.hKernel1 + geometry.hKernel2 - 1
elseif opt.network_structure == 'one_layer' then
   geometry.wKernel = opt.kernel1_size
   geometry.hKernel = opt.kernel1_size
end
geometry.wPatch1 = geometry.wKernel
geometry.hPatch1 = geometry.hKernel
geometry.wPatch2 = geometry.maxw + geometry.wKernel - 1
geometry.hPatch2 = geometry.maxh + geometry.hKernel - 1
geometry.nChannelsIn = 3
geometry.nFeatures = opt.n_features
geometry.soft_targets = opt.soft_targets --todo should be in learning
geometry.features = opt.network_structure --todo change that name
geometry.layerTwoSize = opt.layer_two_size
geometry.layerTwoConnections = opt.layer_two_connections
geometry.L2Pooling = opt.l2_pooling
geometry.multiscale = opt.multiscale
geometry.ratios = {1,2} --todo

local learning = {}
learning.rate = opt.learning_rate
learning.rate_decay = opt.learning_rate_decay
learning.weight_decay = opt.weight_decay
learning.sampling_method = opt.sampling_method

local summary = describeModel(geometry, learning, opt.num_input_images,
			      opt.first_image, opt.delta)

local model
if geometry.multiscale then
   model = getModelFovea(geometry, false)
else
   model = getModel(geometry, false)
end
local parameters, gradParameters = model:getParameters()

local criterion
if geometry.soft_targets then
   criterion = nn.DistNLLCriterion()
   criterion.inputAsADistance = true
   criterion.targetIsProbability = true
else
   criterion = nn.ClassNLLCriterion()
end

print('Loading images...')
local raw_data = loadDataOpticalFlow(geometry, 'data/', opt.num_input_images,
				     opt.first_image, opt.delta, opt.motion_correction)
print('Generating training set...')
local trainData = generateDataOpticalFlow(geometry, raw_data, opt.n_train_set,
					  learning.sampling_method, opt.motion_correction)
print('Generating test set...')
local testData = generateDataOpticalFlow(geometry, raw_data, opt.n_test_set,
					 learning.sampling_method, opt.motion_correction)

saveModel('model_of_', geometry, learning, parameters, opt.num_input_images,
	  opt.first_image, opt.delta, 0)

for iEpoch = 1,opt.n_epochs do
   print('Epoch ' .. iEpoch .. ' over ' .. opt.n_epochs)
   print(summary)

   local nGood = 0
   local nBad = 0
   local meanErr = 0.

   for t = 1,testData:size() do
      modProgress(t, testData:size(), 100)

      local input, target, targetCrit
      if not geometry.multiscale then
	 local sample = testData[t]
	 input = prepareInput(geometry, sample[1][1], sample[1][2])
	 targetCrit, target = prepareTarget(geometry, sample[2])
      else
	 local sample = testData:getElemFovea(t)
	 input = sample[1][1]
	 model:focus(sample[1][2][1], sample[1][2][2])
	 targetCrit, target = prepareTarget(geometry, sample[2])
      end
      
      local output = model:forward(input):squeeze()
      local err = criterion:forward(output, targetCrit)
      
      meanErr = meanErr + err
      local outputp = processOutput(geometry, output)
      if outputp.index == target then
	 nGood = nGood + 1
      else
	 nBad = nBad + 1
      end
   end

   meanErr = meanErr / (testData:size())
   print('test: nGood = ' .. nGood .. ' nBad = ' .. nBad .. ' (' .. 100.0*nGood/(nGood+nBad) .. '%) meanErr = ' .. meanErr)

   nGood = 0
   nBad = 0
   meanErr = 0
   
   for t = 1,trainData:size() do
      modProgress(t, trainData:size(), 100)
      local sample = trainData[t]
      local input = prepareInput(geometry, sample[1][1], sample[1][2])
      local targetCrit, target = prepareTarget(geometry, sample[2])
      
      local feval = function(x)
		       if x ~= parameters then
			  parameters:copy(x)
		       end
		       gradParameters:zero()
		       
		       local output = model:forward(input):squeeze()
		       local err = criterion:forward(output, targetCrit)
		       local df_do = criterion:backward(output, targetCrit)
		       model:backward(input, df_do)
		       
		       meanErr = meanErr + err
		       local outputp = processOutput(geometry, output)
		       if outputp.index == target then
			  nGood = nGood + 1
		       else
			  nBad = nBad + 1
		       end

		       return err, gradParameters
		    end

      config = {learningRate = learning.rate,
		weightDecay = learning.weight_decay,
		momentum = 0,
		learningRateDecay = learning.rate_decay}
      optim.sgd(feval, parameters, config)
   end
      
   meanErr = meanErr / (trainData:size())
   print('train: nGood = ' .. nGood .. ' nBad = ' .. nBad .. ' (' .. 100.0*nGood/(nGood+nBad) .. '%) meanErr = ' .. meanErr)

   saveModel('model_of_', geometry, learning, parameters, opt.num_input_images,
	     opt.first_image, opt.delta, iEpoch)

end