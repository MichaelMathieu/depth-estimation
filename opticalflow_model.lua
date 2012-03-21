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
   local features = nn.Sequential()
   if geometry.features == 'one_layer' then
      features:add(nn.SpatialConvolution(geometry.nChannelsIn, geometry.nFeatures,
					 geometry.wKernel, geometry.hKernel))
      features:add(nn.Tanh())
   elseif geometry.features == 'two_layers' then
      local fst_wsize = math.floor(geometry.wKernel/2)
      features:add(nn.SpatialConvolution(geometry.nChannelsIn, geometry.layerTwoSize,
					 geometry.wKernel1, geometry.hKernel1))
      features:add(nn.Tanh())
      features:add(nn.SpatialConvolutionMap(nn.tables.random(geometry.layerTwoSize,
							     geometry.nFeatures,
							     geometry.layerTwoSize/2),
					    geometry.wKernel2, geometry.hKernel2))
      features:add(nn.Tanh())
   else
      assert(false)
   end
   
   parallel:add(features)
   parallel:add(features:clone('weight', 'bias', 'gradWeight', 'gradBias'))
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
      ret.full:sub(1,1,1+hoffset,ret.y:size(1)+hoffset,1+woffset,ret.y:size(2)+woffset):copy(ret.y)
      ret.full:sub(2,2,1+hoffset,ret.x:size(1)+hoffset,1+woffset,ret.x:size(2)+woffset):copy(ret.x)
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

function describeModel(geometry, nImgs, first_image, delta)
   local imgSize = 'imgSize=(' .. geometry.hImg .. 'x' .. geometry.wImg .. ')'
   local kernel = 'kernel=(' .. geometry.hKernel .. 'x' .. geometry.wKernel .. ')'
   local win = 'win=(' .. geometry.maxh .. 'x' .. geometry.maxw .. ')'
   local images = 'imgs=('..first_image..':'..delta..':'.. first_image+delta*(nImgs-1)..')'
   local features = 'nFeatures=' .. geometry.nFeatures
   local summary = imgSize .. ' ' .. kernel .. ' ' .. win .. ' ' .. images .. ' ' .. features
   return summary
end

function saveModel(basefilename, geometry, parameters, nFeatures, nImgs, first_image, delta,
		   nEpochs, learningRate, sampling_method)
   local modelsdirbase = 'models'
   local modeldir = modelsdirbase .. '/' .. geometry.hImg .. 'x' .. geometry.wImg .. '/'
   modeldir = modeldir .. geometry.maxh .. 'x' .. geometry.maxw .. 'x' .. geometry.hKernel
   modeldir = modeldir .. 'x' .. geometry.wKernel .. '/' .. nFeatures
   os.execute('mkdir -p ' .. modeldir)
   
   local st, sampling
   if geometry.soft_targets then st = 'st' else st = 'ht' end
   if sampling_method == 'uniform_position' then sampling = 'unipos' else sampling = 'uniflow' end
   local images = 'imgs_'..first_image..'_'..delta..'_'..(first_image+delta*(nImgs-1))
   local train_params = '_r_' .. learningRate .. '_' .. sampling .. '_' .. st
   torch.save(modeldir .. '/' .. basefilename .. train_params .. '_' .. images..'_e'..nEpochs,
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
