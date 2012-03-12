require 'torch'
require 'nnx'
require 'image'
require 'optim'
require 'load_data'
require 'groundtruth_discrete'
require 'groundtruth_continuous'
require 'sys'

op = xlua.OptionParser('%prog [options]')
--common
op:option{'-t', '--network-type', action='store', dest='network_type', default='mnist',
	  help='Network type: mnist | opticalflow'}
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
op:option{'-o', '--output_mode', action='store', dest='output_model', default='model',
	  help='Name of the file to save the trained model'}
op:option{'-d', '--delta', action='store', dest='delta', default=10,
	  help='Delta between two consecutive frames'}
op:option{'-rd', '--root-directory', action='store', dest='root_directory',
	  default='./data', help='Root dataset directory'}
--discrete
op:option{'-i', '--input-image', action='store', dest='input_image', default=nil,
	  help='Run network on image. Must be the number of the image (no .jpg)'}
op:option{'-cd', '--cut-depth', action='store', dest='cut_depth', default=nil,
	  help='Specify cutDepth manually'}
op:option{'-nc', '--num-classes', action='store', dest='num_classes', default=2,
	  help='Number of depth classes'}
--continuous
op:option{'-c', '--continuous', action='store_true', dest='continuous', default=false,
	  help='Continuous output (experimental)'}
op:option{'-mc', '--motion-correction', action='store_true', dest='motion_correction', default=false,
     help='Eliminate panning, tilting and rotation camera movements'}
opt=op:parse()
opt.nThreads = tonumber(opt.nThreads)
opt.n_train_set = tonumber(opt.n_train_set)
opt.n_test_set = tonumber(opt.n_test_set)
depthDiscretizer.nClasses = tonumber(opt.num_classes)

torch.manualSeed(1)

if opt.nThreads > 1 then
   require 'openmp'
   openmp.setDefaultNumThreads(opt.nThreads)
end

geometry = {}
geometry.wImg = 640
geometry.hImg = 360
geometry.wPatch = 32
geometry.hPatch = 32
geometry.nImgsPerSample = 2 --todo
geometry.maxw = 17
geometry.maxh = 17

if not opt.continuous then
   classes = {}
   for i = 1,depthDiscretizer.nClasses do
      table.insert(classes, i)
   end
end
if opt.network_type == 'opticalflow' then
   classes = {}
   for i = 1,geometry.maxw*geometry.maxh do
      table.insert(classes, i)
   end
end

if not opt.network then

   input_dim = 2
   if opt.continuous then
      output_dim = 1
   else
      output_dim = #classes
   end
   
   model = nn.Sequential()

   if opt.network_type == 'opticalflow' then
      local hInput = geometry.hPatch
      local wInput = geometry.wPatch
      local kernelSize = 16
      local nChannelsIn = 1
      local nFeatures = 10
      --model:add(nn.SpatialSubtractiveNormalization(input_dim, image.gaussian1D(15)))
      model:add(nn.SplitTable(1))
      local parallel = nn.ParallelTable()
      local parallelElem1 = nn.Sequential()
      local parallelElem2 = nn.Sequential()
      local conv = nn.SpatialConvolution(nChannelsIn, nFeatures, kernelSize, kernelSize)
      parallelElem1:add(nn.Reshape(nChannelsIn, hInput, wInput))
      parallelElem1:add(nn.Narrow(2, math.ceil(geometry.maxh/2), kernelSize))
      parallelElem1:add(nn.Narrow(3, math.ceil(geometry.maxw/2), kernelSize))
      parallelElem1:add(conv)
      parallelElem1:add(nn.Tanh())
      
      parallelElem2:add(nn.Reshape(nChannelsIn, hInput, wInput))
      parallelElem2:add(conv)
      parallelElem2:add(nn.Tanh())

      parallel:add(parallelElem1)
      parallel:add(parallelElem2)
      model:add(parallel)

      model:add(nn.SpatialMatching(geometry.maxh, geometry.maxw, false))
      --model:add(nn.Tanh())
      model:add(nn.Reshape(hInput-kernelSize-geometry.maxh+2,
			   wInput-kernelSize-geometry.maxw+2,
			   geometry.maxw*geometry.maxh))
     
   else
      --model:add(nn.SpatialNormalization{nInputPlane=input_dim,
	--				kernel=image.gaussian(15)})
      model:add(nn.SpatialSubtractiveNormalization(input_dim, image.gaussian1D(15)))
      model:add(nn.SpatialConvolution(input_dim, 50, 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2,2,2,2))
      
      model:add(nn.SpatialSubtractiveNormalization(50, image.gaussian1D(15)))
      model:add(nn.SpatialConvolutionMap(nn.tables.random(50, 128, 10), 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      
      model:add(nn.SpatialConvolution(128, 200, 5, 5))
      model:add(nn.Tanh())
      local spatial = nn.SpatialClassifier()
      spatial:add(nn.Linear(200,output_dim))
      model:add(spatial)
   end
else
   model = torch.load(opt.network)
end

parameters, gradParameters = model:getParameters()

if opt.continuous then
   if opt.network_type == 'opticalflow' then
      criterion = nn.ClassNLLCriterion()
   else
      criterion = nn.MSECriterion()
   end
else
   criterion = nn.DistNLLCriterion()
   criterion.targetIsProbability = true
end

if not opt.network then
   loadData(opt.num_input_images, opt.delta, opt.root_directory)
   if opt.continuous then
      local data = preSortDataContinuous(geometry, raw_data, 26, 20, true);
      if data == nil then
	 sys:exit(0)
      end
      if opt.network_type == 'opticalflow' then
	 trainData = generateContinuousDatasetOpticalFlow(geometry, data, opt.n_train_set);
	 testData = generateContinuousDatasetOpticalFlow(geometry, data, opt.n_test_set);
      else
	 trainData = generateContinuousDataset(geometry, data, opt.n_train_set);
	 testData = generateContinuousDataset(geometry, data, opt.n_test_set);
      end
   else
      --todo maxDepth depends on the dataset, therefore the classes depend too
      preSortDataDiscrete(geometry.hPatch, geometry.wPatch, false, opt.motion_correction)
      if opt.cut_depth then
	 cutDepth=opt.cut_depth
      end
      trainData = generateDataDiscrete(opt.n_train_set, geometry.hPatch, geometry.wPatch, true, true, opt.motion_correction)
      testData = generateDataDiscrete(opt.n_test_set, geometry.hPatch, geometry.wPatch, false, true, opt.motion_correction)
   end

else
   maxDepth=model.maxDepth
   cutDepth=model.cutDepth
end


if not opt.network then
   for epoch = 1,opt.nEpochs do
      print("Epoch " .. epoch)
      xlua.progress(0, trainData:size())

      if opt.continuous then
	 if opt.network_type == 'opticalflow' then
	    nGood = 0
	    nBad = 0
	 else
	    sumdist = torch.Tensor(output_dim):zero()
	    nsamples = 0
	 end
      else
	 confusion = nn.ConfusionMatrix(classes)
      end

      for t = 1,trainData:size() do
	 modProgress(t, trainData:size(), 100);
	 local sample = trainData[t]
	 local input = sample[1]
	 local target
	 if opt.network_type == 'opticalflow' then
	    target = sample[2][2] * geometry.maxw + sample[2][1] + 1
	 else
	    target = sample[2]
	 end
	 
	 local feval = function(x)
			  if x ~= parameters then
			     parameters:copy(x)
			  end
			  gradParameters:zero()
			  
			  local output
			  if opt.network_type == 'opticalflow' then
			     output = model:forward(input)
			     output = output:squeeze()
			     output = torch.Tensor(output:size()):fill(1) - output:abs()
			  else
			     output = model:forward(input):select(3,1):select(2,1)
			  end
			  local err = criterion:forward(output, target)
			  local df_do = criterion:backward(output, target)
			  model:backward(input, df_do)
			  
			  if opt.continuous then
			     if opt.network_type == 'opticalflow' then
				_, ioutput = output:max(1)
				ioutput = ioutput[1]
				--print(output .. ' ' .. target)
				if ioutput == target then
				   nGood = nGood + 1
				else
				   nBad = nBad + 1
				end
			     else
				sumdist = sumdist + torch.abs(output - target)
				nsamples = nsamples + 1
			     end
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
	 if opt.network_type == 'opticalflow' then
	    print('nGood = ' .. nGood .. ' nBad = ' .. nBad)
	    nGood = 0
	    nBad = 0
	 else
	    print('Mean error: ')
	    print(sumdist/nsamples)
	 end
	 sumdist = torch.Tensor(output_dim):zero()
	 nsamples = 0
      else
	 print(confusion)
	 confusion = nn.ConfusionMatrix(classes)
      end

      for t = 1,testData:size() do
	 --modProgress(t, testData:size(), 100)
	 local sample = testData[t]
	 local input = sample[1]
	 --input[2] = input[1]:clone()
	 local target
	 if opt.network_type == 'opticalflow' then
	    target = sample[2][2] * geometry.maxw + sample[2][1] + 1
	 else
	    target = sample[2]
	 end

	 local output
	 if opt.network_type == 'opticalflow' then
	    output = model:forward(input):squeeze()
	    --output = torch.Tensor(output:size()):fill(1) - output:abs()
	    output = output:abs()
	 else
	    output = model:forward(input):select(3,1):select(2,1)
	 end
	 if opt.continuous then
	       if opt.network_type == 'opticalflow' then
		  _, ioutput = output:min(1)
		  ioutput = ioutput[1]
		  --[[
		  youtput = (ioutput-1)/geometry.maxw
		  xoutput = math.mod(ioutput-1, geometry.maxw)
		  if (sample[2][2]-1 <= youtput) and (youtput <= sample[2][2]+1) and
	             (sample[2][1]-1 <= xoutput) and (xoutput <= sample[2][1]+1) then
		  --]]
		  if ioutput == target then
		     nGood = nGood + 1
		  else
		     nBad = nBad + 1
		  end
	       else
		  sumdist = sumdist + torch.abs(output - target)
		  nsamples = nsamples + 1
	       end
	 else
	    confusion:add(output, target)
	 end
	 
      end

      if opt.continuous then
	 if opt.network_type == 'opticalflow' then
	    print('nGood = ' .. nGood .. ' nBad = ' .. nBad)
	 else
	    print('Mean error:')
	    print(sumdist/nsamples)
	 end
      else
	 print(confusion)
      end
   end

   model.maxDepth = maxDepth
   model.cutDepth = cutDepth
   torch.save(opt.output_model, model)
end

if opt.input_image then --todo this is old code, not sure this works anymore
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

