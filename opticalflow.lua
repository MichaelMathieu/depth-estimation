require 'torch'
require 'xlua'
require 'nnx'
require 'image'
require 'optim'
require 'load_data'
require 'groundtruth_opticalflow'
require 'opticalflow_model'
require 'opticalflow_model_io'
require 'sys'
require 'openmp'
require 'score_opticalflow'

torch.manualSeed(1)

op = xlua.OptionParser('%prog [options]')
-- general
op:option{'-nt', '--num-threads', action='store', dest='nThreads', default=2,
	  help='Number of threads used'}

-- network
op:option{'-nf', '--n-features', action='store', dest='n_features',
          default=10, help='Number of features used for the matching'}
op:option{'-k1s', '--kernel1-size', action='store', dest='kernel1_size',
	  default=5, help='Kernel 1 size'}
op:option{'-k2s', '--kernel2-size', action='store', dest='kernel2_size',
	  default=16, help='Kernel 2 size'}
op:option{'-k3s', '--kernel3-size', action='store', dest='kernel3_size',
	  default=16, help='Kernel 3 size'}
op:option{'-ws', '--window-size', action='store', dest='win_size',
	  default=16, help='Window size maxw (and maxh)'}
op:option{'-nl', '-num-layers', action='store', dest='num_layers',
	  default=2, help='Number of layers in the network (1 2 or 3)'}
op:option{'-s2', '--layer-two-size', action='store', dest='layer_two_size', default=8,
	  help='Second layer size, if nl >= 2'}
op:option{'-s2c', '--layer-two-connections', action='store', dest='layer_two_connections',
	  default=4, help='Number of connectons between layers 1 and 2'}
op:option{'-s3', '--layer-three-size', action='store', dest='layer_three_size', default=8,
	  help='Third layer size, if nl >= 3'}
op:option{'-s3c', '--layer-three-connections', action='store', dest='layer_three_connections',
	  default=4, help='Number of connectons between layers 2 and 3'}
op:option{'-l2', '--l2-pooling', action='store_true', dest='l2_pooling', default=false,
	  help='L2 pooling (experimental)'}
op:option{'-ms', '--multiscale', action='store', dest='multiscale', default=0,
	  help='Number of scales used (0 disables multiscale)'}
op:option{'-sf', '--share-filters', action='store_true', dest='share_filters', default=false,
	  help='Share multiscale filters'}
op:option{'-lw', '--load-weights', action='store', dest='load_weights', default = nil,
	  help = 'Load weights from previously trained model'}
op:option{'-mstw', '--multiscale-trainable-weights', action='store_true',
	  dest='ms_trainable_weights', default = false,
	  help='Allow the weights of CascadingAddTable to be trained'}

-- learning
op:option{'-n', '--n-train-set', action='store', dest='n_train_set', default=2000,
	  help='Number of patches in the training set'}
op:option{'-m', '--n-test-set', action='store', dest='n_test_set', default=1000,
	  help='Number of patches in the test set'}
op:option{'-mni', '--n-images-test-set', action='store', dest='n_images_test_set', default=2,
	  help='Number of full images to compute epoch score'}
op:option{'-e', '--num-epochs', action='store', dest='n_epochs', default=10,
	  help='Number of epochs'}
op:option{'-r', '--learning-rate', action='store', dest='learning_rate',
          default=5e-3, help='Learning rate'}
op:option{'-lrd', '--learning-rate-decay', action='store', dest='learning_rate_decay',
          default=5e-7, help='Learning rate decay'}
op:option{'-wd', '--weight-decay', action='store', dest='weight_decay',
	  default=0, help='Weight decay'}
op:option{'-rn', '--renew-train-set', action='store_true', dest='renew_train_set',
	  default=false, help='Renew train set at each epoch'}
op:option{'-st', '--soft-targets', action='store_true', dest='soft_targets',
	  default=false, help='Targets are gaussians'}

-- input
op:option{'-rd', '--root-directory', action='store', dest='root_directory',
	  default='data/', help='Root dataset directory'}
op:option{'-fi', '--first-image', action='store', dest='first_image', default=0,
	  help='Index of first image used'}
op:option{'-d', '--delta', action='store', dest='delta', default=1,
	  help='Delta between two consecutive frames'}
op:option{'-ni', '--num-input-images', action='store', dest='num_input_images',
	  default=10, help='Number of annotated images used'}
op:option{'-mc', '--motion-correction', action='store_true', dest='motion_correction',
	  default=false, help='Eliminate panning, tilting and rotation camera movements'}
op:option{'-lg', '--liu-grountruth', action='store_true', dest='use_liu_groundtruth',
	  default=false, help='Use Liu groundtruth'}
op:option{'-nci', '--n-channels-in', action='store', dest='n_channels_in',
	  default=3, help='Number of channels of the input images'}

-- output 
op:option{'-omd', '--output-model-dir', action='store', dest='output_models_dir',
	  default = 'models', help='Output model directory'}

opt=op:parse()
opt.nThreads = tonumber(opt.nThreads)

opt.multiscale = tonumber(opt.multiscale)

opt.n_train_set = tonumber(opt.n_train_set)
opt.n_test_set = tonumber(opt.n_test_set)
opt.n_images_test_set = tonumber(opt.n_images_test_set)
opt.n_epochs = tonumber(opt.n_epochs)
opt.learning_rate = tonumber(opt.learning_rate)
opt.learning_rate_decay = tonumber(opt.learning_rate_decay)
opt.weight_decay = tonumber(opt.weight_decay)

openmp.setDefaultNumThreads(opt.nThreads)

local geometry = {}
geometry.wImg = 320
geometry.hImg = 180
geometry.maxwGT = tonumber(opt.win_size)
geometry.maxhGT = tonumber(opt.win_size)
geometry.wKernelGT = 16
geometry.hKernelGT = 16
geometry.layers = {}
if tonumber(opt.num_layers) == 1 then
   geometry.layers[1] = {tonumber(opt.n_channels_in), tonumber(opt.kernel1_size),
			 tonumber(opt.kernel1_size), tonumber(opt.n_features)}
   geometry.wKernel = tonumber(opt.kernel1_size)
   geometry.hKernel = tonumber(opt.kernel1_size)
elseif tonumber(opt.num_layers) == 2 then
   geometry.layers[1] = {tonumber(opt.n_channels_in), tonumber(opt.kernel1_size),
			 tonumber(opt.kernel1_size), tonumber(opt.layer_two_size)}
   geometry.layers[2] = {tonumber(opt.layer_two_connections), tonumber(opt.kernel2_size),
			 tonumber(opt.kernel2_size), tonumber(opt.n_features)}
   geometry.wKernel = tonumber(opt.kernel1_size) + tonumber(opt.kernel2_size) - 1
   geometry.hKernel = tonumber(opt.kernel1_size) + tonumber(opt.kernel2_size) - 1
elseif tonumber(opt.num_layers) == 3 then
   geometry.layers[1] = {tonumber(opt.n_channels_in), tonumber(opt.kernel1_size),
			 tonumber(opt.kernel1_size), tonumber(opt.layer_two_size)}
   geometry.layers[2] = {tonumber(opt.layer_two_connections), tonumber(opt.kernel2_size),
			 tonumber(opt.kernel2_size), tonumber(opt.layer_three_size)}
   geometry.layers[3] = {tonumber(opt.layer_three_connections), tonumber(opt.kernel3_size),
			 tonumber(opt.kernel3_size), tonumber(opt.n_features)}
   geometry.wKernel = tonumber(opt.kernel1_size) + tonumber(opt.kernel2_size) + tonumber(opt.kernel3_size) - 2
   geometry.hKernel = tonumber(opt.kernel1_size) + tonumber(opt.kernel2_size) + tonumber(opt.kernel3_size) - 2
else
   assert(false)
end
geometry.L2Pooling = opt.l2_pooling
if opt.multiscale == 0 then
   geometry.multiscale = false
   geometry.ratios = {1}
   geometry.maxw = geometry.maxwGT
   geometry.maxh = geometry.maxhGT
else
   geometry.multiscale = true
   geometry.ratios = {}
   for i = 1,opt.multiscale do table.insert(geometry.ratios, math.pow(2, i-1)) end
   geometry.maxw = math.ceil(geometry.maxwGT / geometry.ratios[#geometry.ratios])
   geometry.maxh = math.ceil(geometry.maxhGT / geometry.ratios[#geometry.ratios])
end
geometry.wPatch2 = geometry.maxw + geometry.wKernel - 1
geometry.hPatch2 = geometry.maxh + geometry.hKernel - 1
geometry.motion_correction = opt.motion_correction
geometry.share_filters = opt.share_filters
geometry.training_mode = true
if geometry.multiscale then
   geometry.cascad_trainable_weights = opt.ms_trainable_weights
end

local learning = {}
learning.first_image = tonumber(opt.first_image)
learning.delta = tonumber(opt.delta)
learning.num_images = tonumber(opt.num_input_images)
learning.rate = opt.learning_rate
learning.rate_decay = opt.learning_rate_decay
learning.weight_decay = opt.weight_decay
learning.renew_train_set = opt.renew_train_set
learning.soft_targets = opt.soft_targets
if opt.use_liu_groundtruth then
   learning.groundtruth = 'liu'
else
   learning.groundtruth = 'cross-correlation'
end

if learning.groundtruth == 'liu' then
   geometry.hKernelGT = geometry.hKernel
   geometry.wKernelGT = geometry.wKernel
   --geometry.maxhGT = geometry.maxh
   --geometry.maxwGT = geometry.maxw
else
   assert(geometry.maxwGT >= geometry.maxw)
   assert(geometry.maxhGT >= geometry.maxh)
end


local summary = describeModel(geometry, learning)

--local model
if geometry.multiscale then
   model = getModelMultiscale(geometry, false)
else
   model = getModel(geometry, false)
end
if opt.load_weights then
   loadWeightsFrom(model, opt.load_weights)
end
local parameters, gradParameters = model:getParameters()

local criterion
if learning.soft_targets then
   criterion = nn.DistNLLCriterion()
   --criterion.targetIsProbability = true
else
   criterion = nn.ClassNLLCriterion()
end

print('Loading images...')
print(opt.root_directory)
local raw_data = loadDataOpticalFlow(geometry, learning, opt.root_directory)
print('Generating training set...')
local trainData = generateDataOpticalFlow(geometry, raw_data, opt.n_train_set)
print('Generating test set...')
local testData = generateDataOpticalFlow(geometry, raw_data, opt.n_test_set)

local score = score_epoch(geometry, learning, model, criterion, testData, raw_data, opt.n_images_test_set)
saveModel(opt.output_models_dir, 'model_of_', geometry, learning, model, 0, score)

config = {learningRate = learning.rate,
	  weightDecay = learning.weight_decay,
	  momentum = 0,
	  learningRateDecay = learning.rate_decay}

for iEpoch = 1,opt.n_epochs do
   print('Epoch ' .. iEpoch .. ' over ' .. opt.n_epochs)
   --print(model.modules[4].weight)
   print(summary)

      
   local nGood = 0
   local nBad = 0
   local meanErr = 0

   if learning.renew_train_set then
      trainData = generateDataOpticalFlow(geometry, raw_data, opt.n_train_set)
   end
   
   for t = 1,trainData:size() do
      modProgress(t, trainData:size(), 100)

      local input, itarget, target
      if geometry.multiscale then
	 local sample = trainData:getElemFovea(t)
	 input = sample[1][1]
	 model:focus(sample[1][2][2], sample[1][2][1])
	 itarget, target = prepareTarget(geometry, learning, sample[2])
      else
	 local sample = trainData[t]
	 input = prepareInput(geometry, sample[1][1], sample[1][2])
	 itarget, target = prepareTarget(geometry, learning, sample[2])
      end
      
      local feval = function(x)
		       if x ~= parameters then
			  parameters:copy(x)
		       end
		       gradParameters:zero()
		       local output = model:forward(input)
		       local err = criterion:forward(output:squeeze(), target)
		       local df_do = criterion:backward(output:squeeze(), target)
		       model:backward(input, df_do)
		       
		       meanErr = meanErr + err
		       local outputp = processOutput(geometry, output, false)
		       if outputp.index == itarget then
			  nGood = nGood + 1
		       else
			  nBad = nBad + 1
		       end
		       return err, gradParameters
		    end

      optim.sgd(feval, parameters, config)
   end
   collectgarbage()

   print(model.cascad.weight)
      
   meanErr = meanErr / (trainData:size())
   print('train: nGood = ' .. nGood .. ' nBad = ' .. nBad .. ' (' .. 100.0*nGood/(nGood+nBad) .. '%) meanErr = ' .. meanErr)

   local score = score_epoch(geometry, learning, model, criterion, testData,
			     raw_data, opt.n_images_test_set)
   saveModel(opt.output_models_dir, 'model_of_', geometry, learning, model, iEpoch, score)

end