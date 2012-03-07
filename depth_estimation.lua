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
op:option{'-cd', '--cut-depth', action='store', dest='cut_depth', default=nil,
	  help='Specify cutDepth manually'}
op:option{'-nc', '--num-classes', action='store', dest='num_classes', default=2,
	  help='Number of depth classes'}
op:option{'-rd', '--root-directory', action='store', dest='root_directory', default='./data',
	  help='Root dataset directory'}
op:option{'-c', '--continuous', action='store_true', dest='continuous', default=false,
	  help='Continuous output (experimental)'}
opt=op:parse()
opt.nThreads = tonumber(opt.nThreads)
opt.n_train_set = tonumber(opt.n_train_set)
opt.n_test_set = tonumber(opt.n_test_set)
nClasses = tonumber(opt.num_classes)
depthDescretizer.nClasses = nClasses

if opt.network_type == 'mul' and not opt.two_frames then
   print("Error: '-t mul' needs '-2'")
   sys.exit(0)
end

torch.manualSeed(1)

if opt.nThreads > 1 then
   require 'openmp'
   openmp.setDefaultNumThreads(opt.nThreads)
end

if not opt.continuous then
   classes = {}
   for i = 1,opt.num_classes do
      table.insert(classes, i)
   end
end
geometry = {32, 32}

if not opt.network then

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
   if opt.continuous then
      spatial:add(nn.Linear(200,1))
   else
      spatial:add(nn.Linear(200,#classes))
   end
   model:add(spatial)

   --[[ old (from mnist)
   model:add(nn.Reshape(128*5*5))
   model:add(nn.Linear(128*5*5,200))
   model:add(nn.Tanh())
   model:add(nn.Linear(200,#classes))
   --]]
else
   model = torch.load(opt.network)
end

parameters, gradParameters = model:getParameters()

if opt.continuous then
   criterion = nn.MSECriterion()
else
   criterion = nn.DistNLLCriterion()
   criterion.targetIsProbability = true
end

--todo maxDepth depends on the dataset, therefore the classes depend too
if not opt.network then
   loadData(opt.num_input_images, opt.delta, geometry, opt.root_directory, opt.continuous)
   if opt.cut_depth then
      cutDepth=opt.cut_depth
   end
   trainData = generateData(opt.n_train_set, geometry[1], geometry[2], true, opt.two_frames,
			 opt.continuous)
   testData = generateData(opt.n_test_set, geometry[1], geometry[2], false, opt.two_frames,
			opt.continuous)
else
   maxDepth=model.maxDepth
   cutDepth=model.cutDepth
end


if not opt.network then
   if opt.continuous then
      sumdist = 0.0
      nsamples = 0
   else
      confusion = nn.ConfusionMatrix(classes)
   end

   for epoch = 1,opt.nEpochs do
      print("Epoch " .. epoch)
      xlua.progress(0, trainData:size())
      for t = 1,trainData:size() do
	 if (math.mod(t, 100) == 0) then
	    xlua.progress(t, trainData:size())
	 end
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
			  
			  if opt.continuous then
			     sumdist = sumdist + abs(output[1] - target[1])
			     nsamples = nsamples + 1
			  else
			     confusion:add(output, target)
			  end
			  
			  return err, gradParameters
		       end
	 
	 config = {learningRate = 1e-2,
		   weightDecay = 0,
		   momentum = 0,
		   learningRateDecay = 5e-7}
	 optim.sgd(feval, parameters, config)
      end

      if opt.continuous then
	 print('Mean distance: ' .. sumdist/nsamples)
	 sumdist = 0.0
	 nsamples = 0
      else
	 print(confusion)
	 confusion = nn.ConfusionMatrix(classes)
      end

      for t = 1,testData:size() do
	 xlua.progress(t, testData:size())
	 local sample = testData[t]
	 local input = sample[1]
	 local target = sample[2]
	 
	 local output = model:forward(input)
	 if opt.continuous then
	    sumdist = sumdist + abs(output[1][1][1] - target[1])
	    nsamples = nsamples + 1
	 else
	    confusion:add(output, target)
	 end
	 
      end

      if opt.continuous then
	 print('Mean distance: ' .. sumdist/nsamples)
      else
	 print(confusion)
      end
   end

   model.maxDepth = maxDepth
   model.cutDepth = cutDepth
   torch.save(opt.output_model, model)
end

if opt.input_image then
   if opt.continuous then
      print('input_image not implemented for continuous output')
      sys.exit(0)
   end
   local directories = {}
   local nDirs = 0
   local findIn = 'find -L ' .. opt.root_directory .. ' -name images'
   for i in io.popen(findIn):lines() do
      nDirs = nDirs + 1
      directories[nDirs] = string.gsub(i, "images", "")
   end
   
   print('Loading image: ' .. directories[1] .. 'images/' .. opt.input_image .. '.jpg')
   local im = image.loadJPG(directories[1] .. 'images/' .. opt.input_image .. '.jpg')
   local im2 = image.loadJPG(directories[1] .. string.format('images/%09d.jpg',
					   tonumber(opt.input_image)+opt.delta))
   local h_im = im:size(2)
   local w_im = im:size(3)
   local h = 360
   local w = 640
   local input = torch.Tensor(2, h, w)
   image.scale(image.rgb2y(im)[1], input[1], 'bilinear')
   image.scale(image.rgb2y(im2)[1], input[2], 'bilinear')
   local gt = torch.DiskFile(directories[1] .. 'depths/' .. opt.input_image .. '.mat')
   local nPts = gt:readInt()
   local inputdisplay = torch.Tensor(3, h, w)
   for i = 1,3 do
      inputdisplay[i]:copy(input[1])
   end
   for i = 1,nPts do
      local yo = gt:readInt()
      local xo = gt:readInt()
      local y = math.floor(yo * h / h_im)+1
      local x = math.floor(xo * w / w_im)+1
      local depth = gt:readDouble()
      inputdisplay[1][y][x] = 0
      inputdisplay[2][y][x] = 0
      inputdisplay[3][y][x] = 0
      --print (depth .. " " .. x .. " " .. y .. " " .. xo .. " " .. yo .. " " .. i .. " " .. getClass(depth))
      inputdisplay[1][y][x] = 1-math.min(depth/30., 1.)
      inputdisplay[2][y][x] = math.min(depth/30., 1.)
      if getClass(depth) == 1 then
	 inputdisplay[3][y][x] = 1
      else
	 inputdisplay[3][y][x] = 0
      end
   end
   image.display{image=inputdisplay}
   local output = model:forward(input)
   assert(output:size(1) == 2) --not implemented for nClasses > 2
   local houtput = output:size(2)
   local woutput = output:size(3)
   todisplay = torch.Tensor(houtput, woutput)
   for i = 1,houtput do
      for j = 1,woutput do
	 if output[1][i][j] < output[2][i][j] then
	    todisplay[i][j] = 0
	 else
	    todisplay[i][j] = 1
	 end
      end
   end
   image.display{image=todisplay}
end

