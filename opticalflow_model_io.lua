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
	       weight3 = weight3:reshape(weight3:size(1)*weight3:size(2), weight3:size(3),
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
	 local weight3 = model.modules[1].modules[1].modules[5].weight
	 if weight3:nDimension() > 3 then --what that happens *only* sometimes??
	    weight3 = weight3:reshape(weight3:size(1)*weight3:size(2), weight3:size(3),
				      weight3:size(4))
	 end
	 table.insert(kernel, weight3)
      end
   end
   return kernel
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
   if (geometry.maxhHR ~= geometry.maxhGT) or (geometry.maxwHR ~= geometry.maxwGT) then
      win = win .. ' gtwin=(' .. geometry.maxhGT .. 'x' .. geometry.maxwGT .. ')'
   end
   local images = 'imgs=(' .. learning.first_image .. ':' .. learning.delta .. ':' 
   images = images .. learning.first_image+learning.delta*(learning.num_images-1) .. ')'
   local learning_ = 'learning rate=(' .. learning.rate .. ', ' .. learning.rate_decay
   learning_ = learning_ .. ') weightDecay=' .. learning.weight_decay
   local summary = imgSize .. ' ' .. kernel .. ' ' .. win .. ' ' .. images .. ' ' .. learning_ .. ' '

   local extra = {}
   if geometry.multiscale then
      if geometry.cascad_trainable_weights then
	 table.insert(extra, 'TrainCascad')
      else
	 table.insert(extra, 'NoTrainCascad')
      end
   end
   if learning.renew_train_set then table.insert(extra, 'renewTrainSet') end
   if geometry.motion_correction then table.insert(extra, 'MotionCorrection') end
   if geometry.share_filters then table.insert(extra, 'ShareFilters') end
   if learning.soft_targets then table.insert(extra, 'SoftTargets('..learning.st_sigma2..')') end
   if geometry.single_beta then table.insert(extra, 'SingleBeta') end
   if learning.groundtruth=='liu' then table.insert(extra, 'Liu') end
   summary = summary .. table.concat(extra, ' ')
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
   kernel = kernel .. '-' .. geometry.maxhHR .. 'x' .. geometry.maxwHR .. '-'
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
   local gt = ''
   if geometry.multiscale then
      if geometry.cascad_trainable_weights then
	 train_cascad = '_tcw'
      else
	 train_cascad = '_ntcw'
      end
      if geometry.single_beta then
	 train_cascad = train_cascad..'_sb'
      end
   end
   if learning.renew_train_set then renew = '_renew' end
   if geometry.motion_correction then motion = '_mc' end
   if learning.soft_targets then targets = '_st'..learning.st_sigma2 end
   if learning.groundtruth == 'liu' then gt = '_liu' end
   local train_params = geometry.maxhGT .. 'x' .. geometry.maxwGT .. '-'
   train_params = train_params .. 'r' .. learning.rate .. '_rd' .. learning.rate_decay
   train_params = train_params .. '_wd' ..learning.weight_decay .. targets .. renew .. gt
   train_params = train_params .. train_cascad
   modeldir = modeldir .. '/' .. train_params
   local images = learning.first_image .. '_' .. learning.delta .. '_' 
   images = images .. (learning.first_image+learning.delta*(learning.num_images-1)) .. motion
   modeldir = modeldir .. '/' .. images
   os.execute('mkdir -p ' .. modeldir)

   local tosave = {}
   tosave.version = 9
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
   torch.save(string.format("%s/%s_e%06d",modeldir, basefilename, nEpochs), tosave)
end

function loadModel(filename, full_output, prefilter, wImg, hImg)
   local loaded = torch.load(filename)
   local ret = {}

   if loaded.version < 9 then
      error("loadModel: can't load before version 9 (structure has changed too much)")
   else
      ret.geometry = loaded.geometry
      if wImg then ret.geometry.wImg = wImg end
      if hImg then ret.geometry.hImg = hImg end
      if full_output then
	 ret.geometry.training_mode = false
      else
	 ret.geometry.training_mode = true
      end
      ret.model = loaded.getModel(ret.geometry, full_output, prefilter)
      ret.getKernels = loaded.getKernels
      ret.score = loaded.score
      if prefilter == true then
	 if ret.geometry.multiscale then
	    local filter = loaded.getFilter(ret.geometry)
	    ret.filter = getMultiscalePrefilter(ret.geometry, filter)
	 else
	    ret.filter = loaded.getFilter(ret.geometry)
	 end
	 local weights = ret.filter:getWeights()
	 for k,v in pairs(weights) do
	    weights[k]:copy(loaded.weights[k])
	 end
	 weights = ret.model:getWeights()
	 for k,v in pairs(weights) do
	    weights[k]:copy(loaded.weights[k])
	 end
      else
	 local weights = ret.model:getWeights()
	 for k,v in pairs(weights) do
	    weights[k]:copy(loaded.weights[k])
	 end
      end
   end
   return ret
end

function loadWeightsFrom(model, filename)
   local loaded = torch.load(filename)
   if loaded.version < 9 then
      error("Can't load weights from file before version 9")
   end
   local weights = model:getWeights()
   for k,v in pairs(loaded.weights) do
      if weights[k] then
	 weights[k]:copy(v)
      end
   end
end
