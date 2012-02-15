require 'torch'
require 'nnx'
require 'image'
require 'depth_dataset_median'
require 'optim'

op = xlua.OptionParser('%prog [options]')
op:option{'-2', '--two-frames', action='store_true', dest='two_frames', default=false,
	  help='Use two consecutives frames instead of one'}
op:option{'-t', '--network-type', action='store', dest='newtork_type', default='mnist',
	  help='Network type: mnist | mul'}
op:option{'-n', '--n-train-set', action='store', dest='n_train_set', default=2000,
	  help='Number of patches in the training set'}
op:option{'-m', '--n-test-set', action='store', dest='n_test_set', default=1000,
	  help='Number of patches in the test set'}
op:option{'-l', '--load-network', action='store', dest='network', default=nil,
	  help='Load pre-trained network'}
op:option{'-ni', '--num-input-images', action='store', dest='num_input_images',
	  default=10, help='Number of annotated images used'}
op:option{'-e', '--num-epochs', action='store', dest='nEpochs', default=10,
	  help='Number of epochs'}
op:option{'-nt', '--num-threads', action='store', dest='nThreads', default=2,
	  help='Number of threads used'}
op:option{'-i', '--input-image', action='store', dest='input_image', default=nil,
	  help='Run network on image. Must be the number of the image (no .jpg)'}
op:option{'-o', '--output_mode', action='store', dest='output_model', default='model',
	  help='Name of the file to save the trained model'}
op:option{'-d', '--delta', action='store', dest='delta', default=10,
	  help='Delta between two consecutive frames'}
opt=op:parse()
opt.nThreads = tonumber(opt.nThreads)
opt.n_train_set = tonumber(opt.n_train_set)
opt.n_test_set = tonumber(opt.n_test_set)

if opt.network_type == 'mul' and not opt.two_frames then
   print("Error: '-t mul' needs '-2'")
   sys.exit(0)
end

torch.manualSeed(1)

if opt.nThreads > 1 then
   require 'openmp'
   openmp.setDefaultNumThreads(opt.nThreads)
end

classes = {'1', '2'}

if not opt.network then
   geometry = {32, 32}

   model = nn.Sequential()

   if opt.two_frames then
      print('Using 2 frames')
      if opt.network_type == 'mul' then
	 model:add(nn.SpatialSubtractiveNormalization(2, image.gaussian1D(15)))
	 model:add(nn.SpatialConvolution(2, 100, 5, 5))
	 model:add(nn.Tanh())
	 model:add(nn.SpatialSubtractiveNormalization(100, image.gaussian1D(15)))
	 model:add(nn.SpatialMaxPooling(2,2,2,2))
	 model:add(nn.Reshape(2, 50, 14, 14))
	 model:add(nn.SplitTable(1))
	 model:add(nn.CMulTable())
	 model:add(nn.Tanh())
      else
	 model:add(nn.SpatialSubtractiveNormalization(2, image.gaussian1D(15)))
	 model:add(nn.SpatialConvolution(2, 50, 5, 5))
	 model:add(nn.Tanh())
	 model:add(nn.SpatialMaxPooling(2,2,2,2))
      end
   else
      print('Using 1 frame')
      model:add(nn.SpatialSubtractiveNormalization(1, image.gaussian1D(15)))
      model:add(nn.SpatialConvolution(1, 50, 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2,2,2,2))
   end

   model:add(nn.SpatialSubtractiveNormalization(50, image.gaussian1D(15)))
   model:add(nn.SpatialConvolutionMap(nn.tables.random(50, 128, 10), 5, 5))
   model:add(nn.Tanh())
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

   model:add(nn.SpatialConvolution(128, 200, 5, 5))
   model:add(nn.Tanh())
   spatial = nn.SpatialClassifier()
   spatial:add(nn.Linear(200,#classes))
   model:add(spatial)

   --[[
   model:add(nn.Reshape(128*5*5))
   model:add(nn.Linear(128*5*5,200))
   model:add(nn.Tanh())
   model:add(nn.Linear(200,#classes))
   --]]
else
   model = torch.load(opt.network)
end

parameters, gradParameters = model:getParameters()

criterion = nn.DistNLLCriterion()
criterion.targetIsProbability = true

if not opt.network then
   loadData(opt.num_input_images, opt.delta)
   trainData = generateData(opt.n_train_set, 32, 32, true, opt.two_frames)
   testData = generateData(opt.n_test_set, 32, 32, false, opt.two_frames)

   confusion = nn.ConfusionMatrix(classes)

   for epoch = 1,opt.nEpochs do
      print("Epoch " .. epoch)
      for t = 1,trainData:size() do
	 xlua.progress(t, trainData:size())
	 local sample = trainData[t]
	 local input = sample[1]
	 local target = sample[2]
	 
	 local feval = function(x)
			  if x ~= parameters then
			     parameters:copy(x)
			  end
			  gradParameters:zero()
			  
			  local output = model:forward(input):select(3,1):select(2,1)
			  local err = criterion:forward(output, target)
			  local df_do = criterion:backward(output, target)
			  model:backward(input, df_do)
			  
			  confusion:add(output, target)
			  
			  return err, gradParameters
		       end
	 
	 config = {learningRate = 1e-2,
		   weightDecay = 0,
		   momentum = 0,
		   learningRateDecay = 5e-7}
	 optim.sgd(feval, parameters, config)
      end

      print(confusion)

      confusion = nn.ConfusionMatrix(classes)

      for t = 1,testData:size() do
	 xlua.progress(t, testData:size())
	 local sample = testData[t]
	 local input = sample[1]
	 local target = sample[2]
	 
	 local output = model:forward(input)
	 confusion:add(output, target)
	 
      end
            
      print(confusion)
   end

   torch.save(opt.output_model, model)
end

if opt.input_image then
   local im = image.loadJPG('data/images/' .. opt.input_image .. '.jpg')
   local im2 = image.loadJPG(string.format('data/images/%09d.jpg', tonumber(opt.input_image)+opt.delta))
   local h = 360
   local w = 640
   local input = torch.Tensor(2, h, w)
   image.scale(image.rgb2y(im)[1], input[1], 'bilinear')
   image.scale(image.rgb2y(im2)[1], input[2], 'bilinear')
   local output = model:forward(input)
   image.display{image=output}
end

