require 'torch'
require 'xlua'
require 'nnx'
require 'SmartReshape'
require 'common'
require 'CascadingAddTable'
require 'OutputExtractor'

function yx2x(geometry, y, x)
   return (y-1) * geometry.maxwGT + x
end

function x2yx(geometry, x)
   if type(x) == 'number' then
      return (math.floor((x-1)/geometry.maxwGT)+1), (math.mod(x-1, geometry.maxwGT)+1)
   else
      local xdbl = torch.DoubleTensor(x:size()):copy(x)-1
      local yout = (xdbl/geometry.maxwGT):floor()
      local xout = xdbl - yout*geometry.maxwGT
      return (yout+1.5):floor(), (xout+1.5):floor() --(a+0.5):floor() is a:round()
   end
end

function yx2xMulti(geometry, y, x)
   function isIn(size, x)
      return (x >= -math.ceil(size/2)+1) and (x <= math.floor(size/2))
   end
   local targetx, targety
   local i = 1
   while i <= #geometry.ratios do
      if (isIn(geometry.maxw*geometry.ratios[i], x) and
       isIn(geometry.maxh*geometry.ratios[i], y)) then
	 --todo this doesn't work it geometry.maxw/geometry.ratios[i] is odd
	 targetx = math.ceil(x/geometry.ratios[i]) + math.ceil(geometry.maxw/2)
	 targety = math.ceil(y/geometry.ratios[i]) + math.ceil(geometry.maxh/2)
	 break
      end
      i = i + 1
   end
   assert(i <= #geometry.ratios)
   if i == 1 then
      itarget = (targety-1) * geometry.maxw + targetx
   else
      -- skip the middle area
      local d = math.floor(geometry.maxw*(geometry.ratios[i]-geometry.ratios[i-1])/(2*geometry.ratios[i]) + 0.5)
      if targety <= d then
	 itarget = (targety-1)*geometry.maxw+targetx
      elseif targety > geometry.maxh-d then
	 itarget = d*geometry.maxw + 2*(geometry.maxh-2*d)*d
	    + (targety-(geometry.maxh-d)-1)*geometry.maxw+targetx
      elseif targetx <= d then
	 itarget = d*geometry.maxw + (targety-d-1)*d+targetx
      elseif targetx > geometry.maxw-d then
	 itarget = d*geometry.maxw + (geometry.maxh-2*d)*d
            + (targety-d-1)*d + targetx-(geometry.maxw-d)
      else
	 assert(false)
      end
      itarget = geometry.maxw*geometry.maxh
	 + (i-2)*(2*d*geometry.maxw + 2*(geometry.maxh-2*d)*d) + itarget
   end
   return itarget
end

function x2yxMulti(geometry, x)
   --todo : do a C module to speed up process
   if type(x) == 'number' then
      return x2yxMultiNumber(geometry, x)
   else
      local retx = torch.Tensor(x:size())
      local rety = torch.Tensor(x:size())
      for i = 1,x:size(1) do
	 for j = 1,x:size(2) do
	    rety[i][j], retx[i][j] = x2yxMultiNumber(geometry, x[i][j])
	 end
      end
      return rety, retx
   end
end

function x2yxMultiNumber(geometry, x)
   assert(type(x) == 'number')
   if x <= geometry.maxh*geometry.maxw then
      -- higher resolution : full patch used
      local targety = math.floor((x-1)/geometry.maxw)+1
      local targetx = math.mod(x-1, geometry.maxw)+1
      return targety - math.ceil(geometry.maxh/2), targetx - math.ceil(geometry.maxw/2)
   else
      -- smaller resolution : middle area isn't used
      x = x - geometry.maxh*geometry.maxw
      for i = 2,#geometry.ratios do
	 local d = round(geometry.maxw * (geometry.ratios[i]-geometry.ratios[i-1])
		       /(2*geometry.ratios[i]))
	 local len = 2*d*geometry.maxw + 2*(geometry.maxh-2*d)*d
	 if x <= len then
	    if x <= d*geometry.maxw then
	       local targety = math.floor((x-1) / geometry.maxw) + 1
	       local targetx = math.mod(x-1, geometry.maxw) + 1
	       return (targety - math.ceil(geometry.maxh/2))*geometry.ratios[i],
	              (targetx - math.ceil(geometry.maxw/2))*geometry.ratios[i]
	    end
	    x = x - d*geometry.maxw
	    if x <= (geometry.maxh-2*d)*d then
	       local targety = math.floor((x-1) / d) + 1 + d
	       local targetx = math.mod(x-1, d) + 1
	       return (targety - math.ceil(geometry.maxh/2))*geometry.ratios[i],
	              (targetx - math.ceil(geometry.maxw/2))*geometry.ratios[i]
	    end
	    x = x - (geometry.maxh-2*d)*d
	    if x <= (geometry.maxh-2*d)*d then
	       local targety = math.floor((x-1) / d) + 1 + d
	       local targetx = math.mod(x-1, d) + 1 + geometry.maxw-d
	       return (targety - math.ceil(geometry.maxh/2))*geometry.ratios[i],
	              (targetx - math.ceil(geometry.maxw/2))*geometry.ratios[i]
	    end
	    x = x - (geometry.maxh-2*d)*d
	    if x <= d*geometry.maxw then
	       local targety = math.floor((x-1) / geometry.maxw) + 1 + geometry.maxh-d
	       local targetx = math.mod(x-1, geometry.maxw) + 1
	       return (targety - math.ceil(geometry.maxh/2))*geometry.ratios[i],
	              (targetx - math.ceil(geometry.maxw/2))*geometry.ratios[i]
	    end
	    assert(false) --this should not happen if the code is correct
	 else
	    x = x - len
	 end
      end
   end
   assert(false) -- this should not happen if geometry is coherent with x
end

-- these 2 functions are kinda deprecated and confusing (because maxhGT != maxh)
-- todo: try not to use them, eventually remove them
function centered2onebased(geometry, y, x)
   return (y+math.ceil(geometry.maxhGT/2)), (x+math.ceil(geometry.maxwGT/2))
end
function onebased2centered(geometry, y, x)
   return (y-math.ceil(geometry.maxhGT/2)), (x-math.ceil(geometry.maxwGT/2))
end

function getMiddleIndex(geometry)
   if geometry.multiscale then
      return yx2xMulti(geometry, 0, 0)
   else
      local y, x = centered2onebased(geometry, 0, 0)
      return yx2x(geometry, y, x)
   end
end


function getFilter(geometry)
   assert(not geometry.L2Pooling)
   local filter = nn.Sequential()
   for i = 1,#geometry.layers do
      if i == 1 or geometry.layers[i-1][4] == geometry.layers[i][1] then
	 filter:add(nn.SpatialConvolution(geometry.layers[i][1], geometry.layers[i][4],
					  geometry.layers[i][2], geometry.layers[i][3]))
      else
	 filter:add(nn.SpatialConvolutionMap(nn.tables.random(geometry.layers[i-1][4],
							      geometry.layers[i][4],
							      geometry.layers[i][1]),
					     geometry.layers[i][2], geometry.layers[i][3]))
      end
      if i ~= #geometry.layers then
	 filter:add(nn.Tanh())
      end
   end
   return filter
end

function getMultiscalePrefilter(geometry, filter)
   assert(geometry.multiscale)
   local wPad = geometry.wPatch2-1
   local hPad = geometry.hPatch2-1
   local padLeft   = math.floor(wPad/2)
   local padRight  = math.ceil (wPad/2)
   local padTop    = math.floor(hPad/2)
   local padBottom = math.ceil (hPad/2)

   local prefilter = nn.ConcatTable()
   for i = 1,#geometry.ratios do
      local seq = nn.Sequential()
      seq:add(nn.SpatialDownSampling(geometry.ratios[i], geometry.ratios[i]))
      seq:add(nn.SpatialZeroPadding(padLeft, padRight, padTop, padBottom))
      if geometry.share_filters then
	 seq:add(filter:clone('weight', 'bias', 'gradWeight', 'gradBias'))
      else
	 seq:add(filter:clone())
      end
      prefilter:add(seq)
   end
   return prefilter
end

function getModel(geometry, full_image, prefiltered)
   if prefiltered == nil then prefiltered = false end
   local model = nn.Sequential()

   if not prefiltered then
      local filter = getFilter(geometry)
      local parallel = nn.ParallelTable()
      parallel:add(filter)
      parallel:add(filter:clone('weight', 'bias', 'gradWeight', 'gradBias'))
      model.filter = filter
      model:add(parallel)
   end

   model:add(nn.SpatialMatching(geometry.maxh, geometry.maxw, false))
   if full_image then
      model:add(nn.Reshape(geometry.maxw*geometry.maxh,
			   geometry.hImg - geometry.hPatch2 + 1,
			   geometry.wImg - geometry.wPatch2 + 1))
   else
      model:add(nn.Reshape(geometry.maxw*geometry.maxh, 1, 1))
   end

   model:add(nn.OutputExtractor(geometry.training_mode, getMiddleIndex(geometry)))

   return model
end

function getModelMultiscale(geometry, full_image, prefiltered)
   local model = nn.Sequential()
   if prefiltered == nil then prefiltered = false end
   assert(geometry.ratios[1] == 1)
   local rmax = geometry.ratios[#geometry.ratios]
   for i = 1,#geometry.ratios do
      local k = rmax - geometry.ratios[i]
      assert(math.mod(geometry.maxh * k, 2) == 0)
      assert(math.mod(geometry.maxw * k, 2) == 0)
   end
   local nChannelsIn = geometry.layers[1][1]

   local filter1 = nn.Sequential()
   local filter2 = nn.Sequential()
   filter1:add(nn.Narrow(1, 1, nChannelsIn))
   filter1:add(nn.SpatialZeroPadding(-math.floor((geometry.maxw-1)/2),
				     -math.ceil ((geometry.maxw-1)/2),
				     -math.floor((geometry.maxh-1)/2),
				     -math.ceil ((geometry.maxh-1)/2)))
   filter2:add(nn.Narrow(1, nChannelsIn+1, nChannelsIn))
   if not prefiltered then
      local filter = getFilter(geometry)
      filter1:add(filter)
      filter2:add(filter:clone('weight', 'bias', 'gradWeight', 'gradBias'))
   end

   local matcher_filters = nn.ConcatTable()
   matcher_filters:add(filter1)
   matcher_filters:add(filter2)

   local matcher = nn.Sequential()
   matcher:add(matcher_filters)
   matcher:add(nn.SpatialMatching(geometry.maxh, geometry.maxw, false))
   if full_image then
      matcher:add(nn.SmartReshape(geometry.maxw*geometry.maxh, -3, -4))
   else
      matcher:add(nn.Reshape(geometry.maxw*geometry.maxh, 1, 1))
   end
   
   local matchers = {}
   for i = 1,#geometry.ratios do
      if geometry.share_filters then
	 matchers[i] = matcher:clone('weight', 'bias', 'gradWeight', 'gradBias')
      else
	 matchers[i] = matcher:clone()
      end
   end

   local pyramid = nn.SpatialPyramid(geometry.ratios, matchers,
				     geometry.wPatch2, geometry.hPatch2, 1, 1, prefiltered)
   model.filter = pyramid

   if not prefiltered then
      model:add(nn.JoinTable(1))
   else
      --todo: this is not smart
      local parallel = nn.ParallelTable()
      for i = 1,#geometry.ratios do
	 parallel:add(nn.JoinTable(1))
      end
      model:add(parallel)
   end
   model:add(pyramid)
   local precascad = nn.ParallelTable()
   for i = 1,#geometry.ratios do
      precascad:add(nn.SmartReshape(geometry.maxh, geometry.maxw, -2, -3))
   end
   model:add(precascad)
   local cascad = nn.CascadingAddTable(geometry.ratios)
   model.cascad = cascad
   model:add(cascad)
   

   local postprocessors = nn.ParallelTable()
   postprocessors:add(nn.SmartReshape({-1,-2},-3,-4))
   for i = 2,#geometry.ratios do
      local d = round(geometry.maxw*(geometry.ratios[i]-geometry.ratios[i-1])/(2*geometry.ratios[i]))
      local remover1 = nn.Sequential()
      local remover2 = nn.Sequential()
      local remover3 = nn.Sequential()
      local remover4 = nn.Sequential()
      remover1:add(nn.Narrow(1, 1, d))
      remover2:add(nn.Narrow(1, d+1, geometry.maxh-2*d))
      remover2:add(nn.Narrow(2, 1, d))
      remover3:add(nn.Narrow(1, d+1, geometry.maxh-2*d))
      remover3:add(nn.Narrow(2, geometry.maxw-d+1, d))
      remover4:add(nn.Narrow(1, geometry.maxh-d+1, d))
      remover1:add(nn.SmartReshape({-1,-2},-3,-4))
      remover2:add(nn.SmartReshape({-1,-2},-3,-4))
      remover3:add(nn.SmartReshape({-1,-2},-3,-4))
      remover4:add(nn.SmartReshape({-1,-2},-3,-4))
      local removers = nn.ConcatTable()
      removers:add(remover1)
      removers:add(remover2)
      removers:add(remover3)
      removers:add(remover4)
      
      local middleRemover = nn.Sequential()
      --middleRemover:add(nn.SmartReshape(geometry.maxh, geometry.maxw, -2, -3))
      middleRemover:add(removers)
      middleRemover:add(nn.JoinTable(1))
      postprocessors:add(middleRemover:clone())
   end
   
   model:add(postprocessors)
   model:add(nn.JoinTable(1))
   
   model:add(nn.OutputExtractor(geometry.training_mode, getMiddleIndex(geometry)))

   if not full_image then
      function model:focus(x, y)
	 pyramid:focus(x, y, 1, 1)
      end
   end
   
   return model
end

function prepareInput(geometry, patch1, patch2)
   assert(patch1:size(2) == patch2:size(2) and patch1:size(3) == patch2:size(3))
   if geometry.multiscale then
      return {patch1, patch2}
   else
      ret = {}
      --TODO this should be floor, according to the way the gt is computed. why?
      ret[1] = patch1:narrow(2, math.ceil(geometry.maxh/2), patch1:size(2)-geometry.maxh+1)
                     :narrow(3, math.ceil(geometry.maxw/2), patch1:size(3)-geometry.maxw+1)
      ret[2] = patch2
      return ret
   end
end

function processOutput(geometry, output, process_full)
   local ret = {}
   if geometry.training_mode then
      local m
      m, ret.index = output:max(1)
      local middleIndex = getMiddleIndex(geometry)
      local flatPixels = torch.LongTensor(m:size(2), m:size(3)):copy(m:eq(output[middleIndex]))
      ret.index = flatPixels * middleIndex + (-flatPixels+1):cmul(ret.index:reshape(ret.index:size(2), ret.index:size(3)))
   else
      ret.index = output
   end
   ret.index = ret.index:squeeze()
   if geometry.multiscale then
      ret.y, ret.x = x2yxMulti(geometry, ret.index)
   else
      ret.y, ret.x = x2yx(geometry, ret.index)
      local yoffset, xoffset = centered2onebased(geometry, 0, 0)
      ret.y = ret.y - yoffset
      ret.x = ret.x - xoffset
   end
   if process_full == nil then
      process_full = type(ret.y) ~= 'number'
   end
   if process_full then
      local hoffset, woffset
      hoffset = math.floor((geometry.hImg-ret.y:size(1))/2)
      woffset = math.floor((geometry.wImg-ret.y:size(2))/2)
      if type(ret.y) == 'number' then
	 ret.full = torch.Tensor(2, geometry.hPatch2, geometry.wPatch2):zero()
	 ret.full[1]:fill(math.ceil(geometry.maxhGT/2))
	 ret.full[2]:fill(math.ceil(geometry.maxwGT/2))
	 ret.full[1][1+hoffset][1+hoffset] = ret.y
	 ret.full[2][1+hoffset][1+woffset] = ret.x
      else
	 ret.full = torch.Tensor(2, geometry.hImg, geometry.wImg):zero()
	 ret.full[1]:fill(math.ceil(geometry.maxhGT/2))
	 ret.full[2]:fill(math.ceil(geometry.maxwGT/2))
	 ret.full:sub(1, 1,
		      1 + hoffset, ret.y:size(1) + hoffset,
		      1 + woffset, ret.y:size(2) + woffset):copy(ret.y)
	 ret.full:sub(2, 2,
		      1 + hoffset, ret.x:size(1) + hoffset,
		      1 + woffset, ret.x:size(2) + woffset):copy(ret.x)
      end
   end
   return ret
end

function processOutput2(geometry, output)
   local ret = {}
   if not CST_Tx then --todo : cleaner
      CST_Tx = torch.Tensor(geometry.maxh, geometry.maxw)
      CST_Ty = torch.Tensor(geometry.maxh, geometry.maxw)
      for i = 1,geometry.maxh do
	 for j = 1,geometry.maxw do
	    CST_Ty[i][j] = i-math.ceil(geometry.maxh/2)
	    CST_Tx[i][j] = j-math.ceil(geometry.maxw/2)
	 end
      end
   end
   local normer = 1.0 / (geometry.maxh*geometry.maxw)
   --local outputr = output:resize(geometry.maxh, geometry.maxw, output:size(2), output:size(3))
   local outputr = output:resize(geometry.maxh, geometry.maxw):exp()
   --print(outputr)
   image.display{image=outputr,zoom=4}
   ret.y = math.floor(outputr:dot(CST_Ty)*normer+0.5)
   ret.x = math.floor(outputr:dot(CST_Tx)*normer+0.5)
   ret.index = yx2x(geometry, ret.y, ret.x)
   return ret
end

function prepareTarget(geometry, target)
   local itarget
   if geometry.multiscale then
      itarget = yx2xMulti(geometry, target[1], target[2])
   else
      local targetx = target[2] + math.ceil(geometry.maxwGT/2)
      local targety = target[1] + math.ceil(geometry.maxhGT/2)
      itarget = (targety-1) * geometry.maxwGT + targetx
   end
   return itarget
end

function getKernels(geometry, model)
   local kernels = {}
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
   return kernels
end

function describeModel(geometry, learning, nImgs, first_image, delta)
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
   local images = 'imgs=('..first_image..':'..delta..':'.. first_image+delta*(nImgs-1)..')'
   local targets = ''
   local motion = ''
   local share = ''
   local learning_ = 'learning rate=(' .. learning.rate .. ', ' .. learning.rate_decay
   learning_ = learning_ .. ') weightDecay=' .. learning.weight_decay .. targets
   if learning.renew_train_set then learning_ = learning_ .. ' renewTrainSet' end
   if geometry.motion_correction then motion = ' MotionCorrection' end
   if geometry.share_filters then share = ' ShareFilters' end
   local summary = imgSize .. ' ' .. kernel .. ' ' .. win .. ' ' .. images .. ' ' .. learning_
   summary = summary .. motion .. share
   return summary
end

function saveModel(basefilename, geometry, learning, parameters, model, nImgs,
		   first_image, delta, nEpochs)
   local modelsdirbase = 'models'
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
   local modeldir = modelsdirbase .. '/' .. kernel
   local targets = ''
   local renew = ''
   local motion = ''
   local share = ''
   if learning.renew_train_set then renew = '_renew' end
   if geometry.motion_correction then motion = '_mc' end
   local train_params = 'r' .. learning.rate .. '_rd' .. learning.rate_decay
   train_params = train_params .. '_wd' ..learning.weight_decay .. targets .. renew
   modeldir = modeldir .. '/' .. train_params
   local images = first_image..'_'..delta..'_'..(first_image+delta*(nImgs-1)) .. motion
   modeldir = modeldir .. '/' .. images
   os.execute('mkdir -p ' .. modeldir)

   local tosave = {}
   tosave.version = 3
   if geometry.multiscale then
      tosave.getModel = getModelMultiscale
   else
      tosave.getModel = getModel
   end
   tosave.model_descr = model:__tostring__()
   if model.cascad then
      tosave.cascad_parameters = model.cascad.weight
      tosave.filter_parameters = parameters[{{1,-tosave.cascad_parameters:size(1)-1}}]
   else
      tosave.filter_parameters = parameters
   end
   tosave.parameters = parameters
   tosave.geometry = geometry
   tosave.learning = learning
   tosave.getKernels = getKernels
   tosave.getFilter = getFilter
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
      ret.getKernel = loaded.getKernels
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
	 if loaded.version < 3 then
	    local parameters = ret.filter:getParameters()
	    parameters:copy(loaded.parameters)
	 else
	    ret.filter:getParameters():copy(loaded.filter_parameters)
	    if loaded.cascad_parameters then
	       ret.model:getParameters():copy(loaded.cascad_parameters)
	    end
	 end
      else
	 local parameters = ret.model:getParameters()
	 parameters:copy(loaded.parameters)
      end
   else
      error('loadModel: wrong version')
   end
   return ret
end

function postProcessImage(input, winsize)
   local output = torch.Tensor(2, input[1]:size(1), input[1]:size(2)):zero()
   local winsizeh1 = math.ceil(winsize/2)-1
   local winsizeh2 = math.floor(winsize/2)
   local win = torch.Tensor(2,winsize,winsize)
   for i = 1+winsizeh1,output:size(2)-winsizeh2 do
      for j = 1+winsizeh1,output:size(3)-winsizeh2 do
	 win[1] = (input[1]:sub(i-winsizeh1,i+winsizeh2, j-winsizeh1, j+winsizeh2)+0.5):floor()
	 win[2] = (input[2]:sub(i-winsizeh1,i+winsizeh2, j-winsizeh1, j+winsizeh2)+0.5):floor()
	 local win2 = win:reshape(2, winsize*winsize)
	 win2 = win2:sort(2)
	 local t = 1
	 local tbest = 1
	 local nbest = 1
	 for k = 2,9 do
	    if (win2:select(2,k) - win2:select(2,t)):abs():sum(1)[1] < 0.5 then
	       if k-t > nbest then
		  nbest = k-t
		  tbest = t
	       end
	    else
	       t = k
	    end
	 end
	 output[1][i][j] = win2[1][tbest]
	 output[2][i][j] = win2[2][tbest]
      end
   end
   return output
end
