require 'torch'
require 'xlua'
require 'nnx'

function yx2x(geometry, y, x)
   return (y-1) * geometry.maxw + x
end

function x2yx(geometry, x)
   if type(x) == 'number' then
      return (math.floor((x-1)/geometry.maxw)+1), (math.mod(x-1, geometry.maxw)+1)
   else
      local xdbl = torch.DoubleTensor(x:size()):copy(x)-1
      local yout = (xdbl/geometry.maxw):floor()
      local xout = xdbl - yout*geometry.maxw
      return (yout+1.5):floor(), (xout+1.5):floor() --(a+0.5):floor() is a:round()
   end
end

function centered2onebased(geometry, y, x)
   return (y+math.ceil(geometry.maxh/2)), (x+math.ceil(geometry.maxw/2))
end

function onebased2centered(geometry, y, x)
   return (y-math.ceil(geometry.maxh/2)), (x-math.ceil(geometry.maxw/2))
end

function getModel(geometry, full_image)
   local model = nn.Sequential()
   local parallel = nn.ParallelTable()
   local features1 = nn.Sequential()
   local features2
   if geometry.nLayers == 1 then
      features1:add(nn.SpatialConvolution(geometry.nChannelsIn, geometry.nFeatures,
					  geometry.wKernel, geometry.hKernel))
      features2 = features1:clone('weight', 'bias', 'gradWeight', 'gradBias')
   elseif geometry.nLayers == 2 then
      features1:add(nn.SpatialConvolution(geometry.nChannelsIn, geometry.layerTwoSize,
					  geometry.wKernel1, geometry.hKernel1))
      features1:add(nn.Tanh())
      if not geometry.L2Pooling then
	 features1:add(nn.SpatialConvolutionMap(nn.tables.random(geometry.layerTwoSize,
								 geometry.nFeatures,
								 geometry.layerTwoConnections),
						geometry.wKernel2, geometry.hKernel2))
	 features2 = features1:clone('weight', 'bias', 'gradWeight', 'gradBias')
      else
	 features1:add(nn.SpatialConvolutionMap(nn.tables.random(geometry.layerTwoSize,
								 geometry.nFeatures*2,
								 geometry.layerTwoConnections),
						geometry.wKernel2, geometry.hKernel2))
	 features1:add(nn.Square())
	 features2 = features1:clone('weight', 'bias', 'gradWeight', 'gradBias')
	 if full_image then
	    features1:add(nn.Reshape(2, geometry.nFeatures,
				    geometry.hImg - geometry.hPatch2 + 1,
				    geometry.wImg - geometry.wPatch2 + 1))
	    features2:add(nn.Reshape(2, geometry.nFeatures,
				     geometry.hImg - geometry.hKernel + 1,
				     geometry.wImg - geometry.wKernel + 1))
	 else
	    features1:add(nn.Reshape(2, geometry.nFeatures, 1, 1))
	    features2:add(nn.Reshape(2, geometry.nFeatures,
				     geometry.maxh, geometry.maxw))
	 end
	 features1:add(nn.SplitTable(1))
	 features2:add(nn.SplitTable(1))
	 features1:add(nn.CAddTable())
	 features2:add(nn.CAddTable())
	 features1:add(nn.Sqrt())
	 features2:add(nn.Sqrt())
      end
   else
      assert(false)
   end

   parallel:add(features1)
   parallel:add(features2)
   model:add(parallel)

   model:add(nn.SpatialMatching(geometry.maxh, geometry.maxw, false))
   if full_image then
      model:add(nn.Reshape(geometry.maxw*geometry.maxh,
			   geometry.hImg - geometry.hPatch2 + 1,
			   geometry.wImg - geometry.wPatch2 + 1))
   else
      model:add(nn.Reshape(geometry.maxw*geometry.maxh, 1, 1))
   end
   if not geometry.soft_targets then
      model:add(nn.Minus())
      local spatial = nn.SpatialClassifier()
      spatial:add(nn.LogSoftMax())
      model:add(spatial)
   end
   
   return model
end

function getModelFovea(geometry, full_image)
   local model = nn.Sequential()
   local parallel = nn.ParallelTable()
   local preproc1 = nn.Sequential()
   local filter = nn.Sequential()

   --TODO this should be floor, according to the way the gt is computed. why?
   -- > anyway, this seems to works perfectly (see test_patches.lua)
   preproc1:add(nn.Narrow(2, math.ceil(geometry.maxh/2), geometry.hImg-geometry.maxh+1))
   preproc1:add(nn.Narrow(3, math.ceil(geometry.maxw/2), geometry.wImg-geometry.maxw+1))

   if geometry.nLayers == 1 then

      filter:add(nn.SpatialConvolution(geometry.nChannelsIn, geometry.nFeatures,
      				       geometry.wKernel, geometry.hKernel))
      --[[
      filter:add(nn.SpatialPadding(-(math.ceil(geometry.wKernel/2)-1),
				   -(math.ceil(geometry.hKernel/2)-1),
				   -math.floor(geometry.wKernel/2),
				   -math.floor(geometry.hKernel/2)))
      --]]
   else
      print('ERROR: two_layers (or other) + fovea not implemented')
      assert(false)
   end

   local filters1 = {}
   local filters2 = {}
   for i = 1,#geometry.ratios do
      local filter_t = filter:clone()
      table.insert(filters1, filter_t)
      table.insert(filters2, filter_t:clone('weight', 'bias', 'gradWeight', 'gradBias'))
   end
   local fovea1 = nn.SpatialPyramid(geometry.ratios, filters1,
				    geometry.wKernel, geometry.hKernel, 1, 1)
   local fovea2 = nn.SpatialPyramid(geometry.ratios, filters2,
				    geometry.wKernel, geometry.hKernel, 1, 1)
   function model:focus(x, y)
      fovea1:focus(x + math.ceil(geometry.wKernel/2)-1,
		   y + math.ceil(geometry.hKernel/2)-1,
		   1, 1)
      fovea2:focus(x + math.ceil(geometry.wPatch2/2)-1,
		   y + math.ceil(geometry.hPatch2/2)-1,
		   geometry.maxw, geometry.maxh)
   end
   
   local seq1 = nn.Sequential()
   seq1:add(preproc1)
   seq1:add(fovea1)
   parallel:add(seq1)
   parallel:add(fovea2)
   
   model:add(parallel)

   model:add(nn.SpatialMatching(geometry.maxh, geometry.maxw, false))
   if full_image then
      model:add(nn.Reshape(geometry.maxw*geometry.maxh,
			   geometry.hImg - geometry.hPatch2 + 1,
			   geometry.wImg - geometry.wPatch2 + 1))
   else
      model:add(nn.Reshape(geometry.maxw*geometry.maxh, 1, 1))
   end
   if not geometry.soft_targets then
      model:add(nn.Minus())
      local spatial = nn.SpatialClassifier()
      spatial:add(nn.LogSoftMax())
      model:add(spatial)
   end

   return model
end

function prepareInput(geometry, patch1, patch2)
   assert(patch1:size(2)==patch2:size(2) and patch1:size(3) == patch2:size(3))
   ret = {}
   --TODO this should be floor, according to the way the gt is computed. why?
   ret[1] = patch1:narrow(2, math.ceil(geometry.maxh/2), patch1:size(2)-geometry.maxh+1)
                  :narrow(3, math.ceil(geometry.maxw/2), patch1:size(3)-geometry.maxw+1)
   ret[2] = patch2
   return ret
end

function processOutput(geometry, output)
   local ret = {}
   if geometry.soft_targets then
      _, ret.index = output:min(1)
   else
      _, ret.index = output:max(1)
   end
   ret.index = ret.index:squeeze()
   ret.y, ret.x = x2yx(geometry, ret.index)
   local hoffset = math.ceil(geometry.maxh/2) + math.ceil(geometry.hKernel/2) - 2
   local woffset = math.ceil(geometry.maxw/2) + math.ceil(geometry.wKernel/2) - 2
   if type(ret.y) == 'number' then
      ret.full = torch.Tensor(2, geometry.hPatch2, geometry.wPatch2):zero()
      ret.full[1][1+hoffset][1+hoffset] = ret.y
      ret.full[2][1+hoffset][1+woffset] = ret.x
   else
      ret.full = torch.Tensor(2, geometry.hImg, geometry.wImg):zero()
      ret.full:sub(1, 1,
		   1 + hoffset, ret.y:size(1) + hoffset,
		   1 + woffset, ret.y:size(2) + woffset):copy(ret.y)
      ret.full:sub(2, 2,
		   1 + hoffset, ret.x:size(1) + hoffset,
		   1 + woffset, ret.x:size(2) + woffset):copy(ret.x)
   end
   return ret
end

function prepareTarget(geometry, target)
   local itarget = yx2x(geometry, target[1], target[2])
   if geometry.soft_targets then
      local ret = torch.Tensor(geometry.maxh*geometry.maxw):zero()
      local sigma2 = 1
      local normer = 1.0 / math.sqrt(sigma2 * 2.0 * math.pi)
      for i = 1,geometry.maxh do
	 for j = 1,geometry.maxw do
	    local dist = math.sqrt((target[1]-i)*(target[1]-i)+(target[2]-j)*(target[2]-j))
	    ret[yx2x(geometry, i, j)] = normer * math.exp(-dist*dist/sigma2)
	 end
      end
      return ret, itarget
   else
      return itarget, itarget
   end
end

function describeModel(geometry, learning, nImgs, first_image, delta)
   local imgSize = 'imgSize=(' .. geometry.hImg .. 'x' .. geometry.wImg .. ')'
   local kernel
   if geometry.nLayers == 1 then
      kernel = 'kernel=(' .. geometry.nChannelsIn .. 'x' .. geometry.hKernel
      kernel = kernel .. 'x' .. geometry.wKernel .. geometry.nFeatures .. ')'
   else
      kernel = 'kernels=(' .. geometry.nChannelsIn .. 'x' .. geometry.hKernel1
      kernel = kernel .. 'x' .. geometry.wKernel1 .. 'x' .. geometry.layerTwoSize .. ', '
      kernel = kernel .. geometry.layerTwoConnections .. 'x' .. geometry.hKernel2 .. 'x'
      kernel = kernel .. geometry.wKernel2 .. 'x' .. geometry.nFeatures
      if geometry.L2Pooling then kernel = kernel .. ' l2' end
      if geometry.multiscale then kernel = kernel .. ' multi' end
      kernel = kernel .. ')'
   end
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
   local sampling = ''
   if geometry.soft_targets then targets = '_softTargets' end
   if learning.sampling_method ~= 'uniform_position' then
      sampling = '_' .. learning.sampling_method
   end
   local learning_ = 'learning rate=(' .. learning.rate .. ', ' .. learning.rate_decay
   learning_ = learning_ .. ') weight decay=' .. learning.weight_decay .. targets .. sampling
   local summary = imgSize .. ' ' .. kernel .. ' ' .. win .. ' ' .. images .. ' ' .. learning_
   return summary
end

function saveModel(basefilename, geometry, learning, parameters, nImgs, first_image, delta,
		   nEpochs)
   local modelsdirbase = 'models'
   local modeldir = modelsdirbase .. '/'
   if geometry.nLayers == 1 then
      modeldir = modeldir .. geometry.nChannelsIn .. 'x' .. geometry.hKernel
      modeldir = modeldir .. 'x' .. geometry.wKernel .. geometry.nFeatures
   else
      modeldir = modeldir .. geometry.nChannelsIn .. 'x' .. geometry.hKernel1
      modeldir = modeldir .. 'x' .. geometry.wKernel1 .. 'x' .. geometry.layerTwoSize .. '_'
      modeldir = modeldir .. geometry.layerTwoConnections .. 'x' .. geometry.hKernel2 .. 'x'
      modeldir = modeldir .. geometry.wKernel2 .. 'x' .. geometry.nFeatures
      if geometry.L2Pooling then modeldir = modeldir .. '_l2' end
   end
   if geometry.multiscale then
      modeldir = modeldir .. '_multi'
      for i = 1,#geometry.ratios do
	 modeldir = modeldir .. '-' .. geometry.ratios[i]
      end
   end
   
   local targets = ''
   local sampling = ''
   if geometry.soft_targets then targets = ' softTargets' end
   if learning.sampling_method ~= 'uniform_position' then
      sampling = ' ' ..learning.sampling_method
   end
   local train_params = 'r' .. learning.rate .. '_rd' .. learning.rate_decay .. '_wd'
   train_params = train_params .. learning.weight_decay .. sampling .. targets
   modeldir = modeldir .. '/' .. train_params
   local images = first_image..'_'..delta..'_'..(first_image+delta*(nImgs-1))
   modeldir = modeldir .. '/' .. images
   os.execute('mkdir -p ' .. modeldir)
   torch.save(modeldir .. '/' .. basefilename .. '_e'..nEpochs,
	      {parameters, geometry})
end

function loadModel(filename, full_output)
   local loaded = torch.load(filename)
   local geometry = loaded[2]
   local model = getModel(geometry, full_output)
   local parameters = model:getParameters()
   parameters:copy(loaded[1])
   return geometry, model
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
