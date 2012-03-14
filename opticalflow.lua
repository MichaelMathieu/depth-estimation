require 'torch'
require 'xlua'
require 'nnx'
require 'image'
require 'optim'
require 'load_data'
require 'groundtruth_opticalflow'
require 'opticalflow_model'
require 'sys'

op = xlua.OptionParser('%prog [options]')
op:option{'-n', '--n-train-set', action='store', dest='n_train_set', default=2000,
	  help='Number of patches in the training set'}
op:option{'-m', '--n-test-set', action='store', dest='n_test_set', default=1000,
	  help='Number of patches in the test set'}
op:option{'-fi', '--first-image', action='store', dest='first_image', default=0,
	  help='Index of first image used'}
op:option{'-d', '--delta', action='store', dest='delta', default=2,
	  help='Delta between two consecutive frames'}
op:option{'-rd', '--root-directory', action='store', dest='root_directory',
	  default='./data', help='Root dataset directory'}
op:option{'-nt', '--num-threads', action='store', dest='nThreads', default=2,
	  help='Number of threads used'}
op:option{'-ni', '--num-input-images', action='store', dest='num_input_images',
	  default=10, help='Number of annotated images used'}
op:option{'-e', '--num-epochs', action='store', dest='n_epochs', default=10,
	  help='Number of epochs'}
op:option{'-r', '--learning-rate', action='store', dest='learning_rate',
          default=5e-3, help='Learning rate'}
op:option{'-st', '--soft-targets', action='store_true', dest='soft_targets',
          default=false, help='Enable soft targets (targets are gaussians centered on groundtruth)'}

opt=op:parse()
opt.nThreads = tonumber(opt.nThreads)
opt.n_train_set = tonumber(opt.n_train_set)
opt.n_test_set = tonumber(opt.n_test_set)
opt.n_epochs = tonumber(opt.n_epochs)
opt.num_input_images = tonumber(opt.num_input_images)
opt.learning_rate = tonumber(opt.learning_rate)
opt.delta = tonumber(opt.delta)
opt.first_image = tonumber(opt.first_image)

torch.manualSeed(1)

if opt.nThreads > 1 then
   require 'openmp'
   openmp.setDefaultNumThreads(opt.nThreads)
end

local geometry = {}
geometry.wImg = 320
geometry.hImg = 180
geometry.wPatch2 = 32
geometry.hPatch2 = 32
geometry.wKernel = 16
geometry.hKernel = 16
geometry.maxw = geometry.wPatch2 - geometry.wKernel + 1
geometry.maxh = geometry.hPatch2 - geometry.hKernel + 1
geometry.wPatch1 = geometry.wPatch2 - geometry.maxw + 1
geometry.hPatch1 = geometry.hPatch2 - geometry.maxh + 1
geometry.nChannelsIn = 3
geometry.nFeatures = 10

local model = getModel(geometry, false)
local parameters, gradParameters = model:getParameters()

local criterion
if opt.soft_targets then
   criterion = nn.DistNLLCriterion()
   criterion.inputAsADistance = true
   criterion.targetIsProbability = true
else
   criterion = nn.ClassNLLCriterion()
end

print('Loading images...')
local raw_data = loadDataOpticalFlow(geometry, 'data/', opt.num_input_images,
				     opt.first_image, opt.delta)
print('Generating training set...')
local trainData = generateDataOpticalFlow(geometry, raw_data, opt.n_train_set,
					  'uniform_position');
print('Generating test set...')
local testData = generateDataOpticalFlow(geometry, raw_data, opt.n_test_set,
				      'uniform_position');

for iEpoch = 1,opt.n_epochs do
   print('Epoch ' .. iEpoch)

   nGood = 0
   nBad = 0

   for t = 1,testData:size() do
      xlua.progress(t, testData:size())
      local sample = testData[t]
      local input = prepareInput(geometry, sample[1][1], sample[1][2])
      local targetCrit, target = prepareTarget(geometry, sample[2], opt.soft_targets)
      
      local output = model:forward(input):squeeze()
      
      output = processOutput(geometry, output)
      if output.index == target then
	 nGood = nGood + 1
      else
	 nBad = nBad + 1
      end
   end
      
   print('nGood = ' .. nGood .. ' nBad = ' .. nBad)


   nGood = 0
   nBad = 0
   
   for t = 1,trainData:size() do
      xlua.progress(t, trainData:size())
      local sample = trainData[t]
      local input = prepareInput(geometry, sample[1][1], sample[1][2])
      local targetCrit, target = prepareTarget(geometry, sample[2], opt.soft_targets)
      
      local feval = function(x)
		       if x ~= parameters then
			  parameters:copy(x)
		       end
		       gradParameters:zero()
		       
		       local output = model:forward(input):squeeze()
		       local err = criterion:forward(output, targetCrit)
		       local df_do = criterion:backward(output, targetCrit)
		       model:backward(input, df_do)
		       
		       output = processOutput(geometry, output)
		       if output.index == target then
			  nGood = nGood + 1
		       else
			  nBad = nBad + 1
		       end

		       return err, gradParameters
		    end

      config = {learningRate = opt.learning_rate,
		weightDecay = 0,
		momentum = 0,
		learningRateDecay = 5e-7}
      optim.sgd(feval, parameters, config)
   end
      
   print('nGood = ' .. nGood .. ' nBad = ' .. nBad)

end

torch.save('model_opticalflow', {parameters, geometry})