require 'opticalflow_model'

function getKernels(geometry, model)
   local kernel = {}
   if geometry.multiscale then
      for i = 1,#geometry.ratios do
	 local matcher = model.modules[2].unfocused_pipeline.modules[i].modules[3]
	 local weight = matcher.modules[1].modules[1].modules[3].modules[1].weight
	 table.insert(kernel, weight)
	 if #geometry.layers > 1 then
	    local weight2 = matcher.modules[1].modules[1].modules[3].modules[3].weight
	    if weight2:nDimension() > 3 then --what that happens *only* sometimes??
	       weight2 = weight2:reshape(weight2:size(1)*weight2:size(2), weight2:size(3),
					 weight2:size(4))
	    end
	    table.insert(kernel, weight2)
	 end
	 if #geometry.layers > 2 then
	    local weight3 = matcher.modules[1].modules[1].modules[3].modules[3].weight
	    if weight3:nDimension() > 3 then --what that happens *only* sometimes??
	       weight3 = weight2:reshape(weight3:size(1)*weight3:size(2), weight3:size(3),
					 weight3:size(4))
	    end
	    table.insert(kernel, weight3)
	 end
      end
   else
      local weight = model.modules[1].modules[1].modules[1].weight
      table.insert(kernel, weight)
      if #geometry.layers > 1 then
	 local weight2 = model.modules[1].modules[1].modules[3].weight
	 if weight2:nDimension() > 3 then --what that happens *only* sometimes??
	    weight2 = weight2:reshape(weight2:size(1)*weight2:size(2), weight2:size(3),
				      weight2:size(4))
	 end
	 table.insert(kernel, weight2)
      end
      if #geometry.layers > 2 then
	 local weight3 = matcher.modules[1].modules[1].modules[3].modules[3].weight
	 if weight3:nDimension() > 3 then --what that happens *only* sometimes??
	    weight3 = weight2:reshape(weight3:size(1)*weight3:size(2), weight3:size(3),
				      weight3:size(4))
	 end
	 table.insert(kernel, weight3)
      end
   end
   return kernel
end

--do not change that function anymore (eventually, remove it)
function getKernelsLegacy(geometry, model)
   local kernels = {}
   if geometry.multiscale then
      for i = 1,#geometry.ratios do
	 local matcher = model.modules[2].unfocused_pipeline.modules[i].modules[3]
	 local weight = matcher.modules[1].modules[1].modules[3].modules[1].weight
	 table.insert(kernels, weight)
	 if #geometry.layers > 1 then
	    local weight2 = matcher.modules[1].modules[1].modules[3].modules[3].weight
	    if weight2:nDimension() > 3 then --what that happens *only* sometimes??
	       weight2 = weight2:reshape(weight2:size(1)*weight2:size(2), weight2:size(3),
					 weight2:size(4))
	    end
	    table.insert(kernels, weight2)
	 end
      end
   else
      local weight = model.modules[1].modules[1].modules[1].weight
      table.insert(kernels, weight)
      if #geometry.layers > 1 then
	 local weight2 = model.modules[1].modules[1].modules[3].weight
	 if weight2:nDimension() > 3 then --what that happens *only* sometimes??
	    weight2 = weight2:reshape(weight2:size(1)*weight2:size(2), weight2:size(3),
				      weight2:size(4))
	 end
	 table.insert(kernels, weight2)
      end
   end
   return kernels
end

function describeModel(geometry, learning)
   local imgSize = 'imgSize=(' .. geometry.hImg .. 'x' .. geometry.wImg .. ')'
   local kernel = 'kernel=('
   for i = 1,#geometry.layers do
      kernel = kernel .. geometry.layers[i][1] .. 'x' .. geometry.layers[i][2] .. 'x'
      kernel = kernel .. geometry.layers[i][3] .. 'x' .. geometry.layers[i][4]
      if i ~= #geometry.layers then
	 kernel = kernel .. ', '
      end
   end
   if geometry.L2Pooling then kernel = kernel .. ' l2' end
   kernel = kernel .. ')'
   if geometry.multiscale then
      kernel = kernel .. 'x{' .. geometry.ratios[1]
      for i = 2,#geometry.ratios do
	 kernel = kernel .. ',' .. geometry.ratios[i]
      end
      kernel = kernel .. '}'
   end
   local win = 'win=(' .. geometry.maxh .. 'x' .. geometry.maxw .. ')'
   local images = 'imgs=(' .. learning.first_image .. ':' .. learning.delta .. ':' 
   images = images .. learning.first_image+learning.delta*(learning.num_images-1) .. ')'
   local targets = ''
   local motion = ''
   local share = ''
   local train_cascad = ''
   if geometry.multiscale then
      if geometry.cascad_trainable_weights then
	 train_cascad = ' TrainCascad'
      else
	 train_cascad = ' NoTrainCascad'
      end
   end
   local learning_ = 'learning rate=(' .. learning.rate .. ', ' .. learning.rate_decay
   learning_ = learning_ .. ') weightDecay=' .. learning.weight_decay .. targets
   if learning.renew_train_set then learning_ = learning_ .. ' renewTrainSet' end
   if geometry.motion_correction then motion = ' MotionCorrection' end
   if geometry.share_filters then share = ' ShareFilters' end
   local summary = imgSize .. ' ' .. kernel .. ' ' .. win .. ' ' .. images .. ' ' .. learning_
   summary = summary .. motion .. share .. train_cascad
   return summary
end

function saveModel(dir, basefilename, geometry, learning, model, nEpochs, score)
   if dir:sub(-1) ~= '/' then dir = dir..'/' end
   local modelsdirbase = dir
   local kernel = ''
   for i = 1,#geometry.layers do
      kernel = kernel .. geometry.layers[i][1] .. 'x' .. geometry.layers[i][2] .. 'x'
      kernel = kernel .. geometry.layers[i][3] .. 'x' .. geometry.layers[i][4]
      if i ~= #geometry.layers then
	 kernel = kernel .. '_'
      end
   end
   if geometry.L2Pooling then kernel = kernel .. '_l2' end
   if geometry.share_filters then kernel = kernel .. '_sf' end
   if geometry.multiscale then
      for i = 1,#geometry.ratios do
	 kernel = kernel .. '-' .. geometry.ratios[i]
      end
   end
   local modeldir = modelsdirbase .. kernel
   local targets = ''
   local renew = ''
   local motion = ''
   local share = ''
   local train_cascad = ''
   if geometry.multiscale then
      if geometry.cascad_trainable_weights then
	 train_cascad = '_tcw'
      else
	 train_cascad = '_ntcw'
      end
   end
   if learning.renew_train_set then renew = '_renew' end
   if geometry.motion_correction then motion = '_mc' end
   local train_params = 'r' .. learning.rate .. '_rd' .. learning.rate_decay
   train_params = train_params .. '_wd' ..learning.weight_decay .. targets .. renew
   train_params = train_params .. train_cascad
   modeldir = modeldir .. '/' .. train_params
   local images = learning.first_image .. '_' .. learning.delta .. '_' 
   images = images .. (learning.first_image+learning.delta*(learning.num_images-1)) .. motion
   modeldir = modeldir .. '/' .. images
   os.execute('mkdir -p ' .. modeldir)

   local tosave = {}
   tosave.version = 8
   if geometry.multiscale then
      tosave.getModel = getModelMultiscale
   else
      tosave.getModel = getModel
   end
   tosave.model_descr = model:__tostring__()
   tosave.weights = model:getWeights()
   tosave.geometry = geometry
   tosave.learning = learning
   tosave.getKernels = getKernels
   tosave.getFilter = getFilter
   tosave.score = score
   torch.save(modeldir .. '/' .. basefilename .. '_e'..nEpochs, tosave)
end

function loadModel(filename, full_output, prefilter, wImg, hImg)
   local loaded = torch.load(filename)
   local ret = {}

   if not loaded.version then -- old version
      ret.geometry = loaded[2]
      if not ret.geometry.layers then
	 local nLayers
	 if ret.geometry.features then
	    if ret.geometry.features == 'two_layers' then
	       nLayers = 2
	    elseif ret.geometry.featues == 'one_layer' then
	       nLayers = 1
	    else
	       print(ret.geometry)
	       error('Unknown model version')
	    end
	 elseif ret.geometry.nLayers then
	    nLayers = ret.geometry.nLayers
	 else
	    print(ret.geometry)
	    error('Unknown model version')
	 end
	 ret.geometry.layers = {}
	 if nLayers == 2 then
	    ret.geometry.layers[1] = {ret.geometry.nChannelsIn, ret.geometry.hKernel1,
				      ret.geometry.wKernel1, ret.geometry.layerTwoSize}
	    ret.geometry.layers[2] = {ret.geometry.layerTwoConnections, ret.geometry.hKernel2,
				      ret.geometry.wKernel2, ret.geometry.nFeatures}
	 else
	    ret.geometry.layers[1] = {ret.geometry.nChannelsIn, ret.geometry.hKernel,
				      ret.geometry.wKernel, ret.geometry.nFeatures}
	 end
      end
      if not ret.geometry.maxhGT then
	 ret.geometry.maxhGT = ret.geometry.maxh
	 ret.geometry.maxwGT = ret.geometry.maxw
      end
      if wImg then ret.geometry.wImg = wImg end
      if hImg then ret.geometry.hImg = hImg end
      ret.geometry.training_mode = true
      if ret.geometry.multiscale then
	 ret.model = getModelMultiscale(ret.geometry, full_output)
      else
	 ret.model = getModel(ret.geometry, full_output)
      end
      local parameters = ret.model:getParameters()
      parameters:copy(loaded[1])
      ret.getKernels = getKernelLegacy

   elseif loaded.version >= 1 then 
      ret.geometry = loaded.geometry
      if wImg then ret.geometry.wImg = wImg end
      if hImg then ret.geometry.hImg = hImg end
      if full_output and ret.geometry.training_mode then
	 ret.geometry.training_mode = false
      else
	 ret.geometry.training_mode = true
      end
      ret.model = loaded.getModel(ret.geometry, full_output, prefilter)
      if loaded.getKernels and loaded.version ~= 5 then
	 ret.getKernels = loaded.getKernels
      else
	 ret.getKernels = getKernelsLegacy
      end
      if loaded.version >= 5 then
	 ret.score = loaded.score
      end
      if prefilter == true then
	 if loaded.version < 2 then
	    error("loadModel: prefilter didn't exist before version 2")
	 end
	 if ret.geometry.multiscale then
	    local filter = loaded.getFilter(ret.geometry)
	    ret.filter = getMultiscalePrefilter(ret.geometry, filter)
	 else
	    ret.filter = loaded.getFilter(ret.geometry)
	 end
	 if loaded.version == 2 then
	    local parameters = ret.filter:getParameters()
	    parameters:copy(loaded.parameters)
	 elseif loaded.version == 3 then
	    ret.filter:getParameters():copy(loaded.filter_parameters)
	    if loaded.cascad_parameters then
	       ret.model:getParameters():copy(loaded.cascad_parameters)
	    end
	 elseif loaded.version >= 4  and loaded.version < 6 then
	    local weights = ret.filter:getWeights()
	    if ret.geometry.share_filters then
	       for k,v in pairs(weights) do
		  local kloaded = 'scale1_'..k
		  weights[k]:copy(loaded.weights[kloaded])
	       end
	    else
	       for k,v in pairs(weights) do
		  weights[k]:copy(loaded.weights[k])
	       end
	    end
	    weights = ret.model:getWeights()
	    for k,v in pairs(weights) do
	       weights[k]:copy(loaded.weights[k])
	    end
	 elseif loaded.version >= 6 then
	    local weights = ret.filter:getWeights()
	    for k,v in pairs(weights) do
	       weights[k]:copy(loaded.weights[k])
	    end
	    weights = ret.model:getWeights()
	    for k,v in pairs(weights) do
	       weights[k]:copy(loaded.weights[k])
	    end
	 end
	 if loaded.version < 8 and ret.geometry.multiscale then
	    local nFeatures = ret.geometry.layers[#ret.geometry.layers][4]
	    for i = 1,#ret.model.modules[2].processors do
	       ret.model.modules[2].processors[i].modules[1].modules[1].modules[1].length = nFeatures
	       ret.model.modules[2].processors[i].modules[1].modules[2].modules[1].index = nFeatures+1
	       ret.model.modules[2].processors[i].modules[1].modules[2].modules[1].length = nFeatures
	    end
	 end
      else
	 if loaded.version < 4 then
	    local parameters = ret.model:getParameters()
	    parameters:copy(loaded.parameters)
	 else
	    local weights = ret.model:getWeights()
	    for k,v in pairs(weights) do
	       weights[k]:copy(loaded.weights[k])
	    end
	 end
      end
   else
      error('loadModel: wrong version')
   end
   return ret
end

function loadWeightsFrom(model, filename)
   local loaded = torch.load(filename)
   if loaded.version < 4 then
      error("Can't load weights from file before version 4")
   end
   local weights = model:getWeights()
   for k,v in pairs(loaded.weights) do
      if weights[k] then
	 weights[k]:copy(v)
      end
   end
end
