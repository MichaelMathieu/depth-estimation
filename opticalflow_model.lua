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
      local xout = (xdbl/geometry.maxw):floor()
      local yout = xdbl - xout*geometry.maxw
      return (yout+1.5):floor(), (xout+1.5):floor() --(a+0.5):floor() is a:round()
   end
end

function centered2onebased(geometry, y, x)
   return (y+math.ceil(geometry.maxh/2)), (x+math.ceil(geometry.maxw/2))
end

function onebased2centered(geometry, y, x)
   return (y-math.ceil(geometry.maxh/2)), (x-math.ceil(geometry.maxw/2))
end

function getModel(geometry, full_image, soft_targets)
   local model = nn.Sequential()
   local parallel = nn.ParallelTable()
   local parallelElem1 = nn.Sequential()
   local parallelElem2 = nn.Sequential()
   local conv = nn.SpatialConvolution(geometry.nChannelsIn, geometry.nFeatures,
				      geometry.wKernel, geometry.hKernel)
   parallelElem1:add(conv)
   parallelElem1:add(nn.Tanh())
   
   parallelElem2:add(conv)
   parallelElem2:add(nn.Tanh())
   
   parallel:add(parallelElem1)
   parallel:add(parallelElem2)
   model:add(parallel)

   model:add(nn.SpatialMatching(geometry.maxh, geometry.maxw, full_image))
   if full_image then
      model:add(nn.Reshape(geometry.maxw*geometry.maxh,
			   geometry.hImg - geometry.hKernel + 1,
			   geometry.wImg - geometry.wKernel + 1))
   else
      model:add(nn.Reshape(geometry.maxw*geometry.maxh,
			   geometry.hPatch2 - geometry.hKernel - geometry.maxh + 2,
			   geometry.wPatch2 - geometry.wKernel - geometry.maxw + 2))
   end

   if not soft_targets then
      model:add(nn.Minus())
      local spatial = nn.SpatialClassifier()
      spatial:add(nn.LogSoftMax())
      model:add(spatial)
   end

   return model
end

function prepareInput(geometry, patch1, patch2)
   ret = {}
   --TODO this should be floor, according to the way the gt is computed. why?
   ret[1] = patch1:narrow(2, math.ceil(geometry.maxh/2), geometry.hKernel)
                  :narrow(3, math.ceil(geometry.maxw/2), geometry.wKernel)
   ret[2] = patch2
   return ret
end

function processOutput(geometry, logprobs)
   ret = {}
   _, ret.index = logprobs:max(1)
   ret.index = ret.index:squeeze()
   ret.y, ret.x = x2yx(geometry, ret.index)
   return ret
end

function prepareTarget(geometry, target, soft_targets)
   local itarget = yx2x(geometry, target[2], target[1])
   if soft_targets then
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

function describeModel(geometry)
   local summary = ''
   for key, value in pairs(geometry) do
      summary = summary .. key .. '=' .. value .. ' '
   end
   return summary
   --[[
   local st
   if opt.soft_targets then
      st = 'st'
   else
      st = 'ht'
   end
   local summary = 'nf=' .. opt.n_features .. ' e=' .. opt.n_epochs .. ' r=' .. opt.learning_rate .. ' ni=' .. opt.num_input_images .. ' d=' .. opt.delta .. ' n=' .. opt.n_train_set .. ' ' .. st
   --]]
end

function saveModel(basefilename, geometry, parameters)
   local modelsdirbase = 'models'
   local modeldir = modelsdirbase .. '/' .. geometry.hImg .. 'x' .. geometry.wImg .. '/'
   modeldir = modeldir .. geometry.maxh .. 'x' .. geometry.maxw .. 'x' .. geometry.hKernel
   modeldir = modeldir .. 'x' .. geometry.wKernel
   io.popen('mkdir -p ' .. modeldir)
   local st, sampling
   if opt.soft_targets then st = 'st' else st = 'ht' end
   if opt.sampling_method == 'uniform_position' then sampling = 'unipos' else sampling = 'uniflow' end
   torch.save(modeldir .. '/' .. basefilename .. 'nf_' .. opt.n_features .. '_e_' .. opt.n_epochs .. '_r_' .. opt.learning_rate .. '_ni_' .. opt.num_input_images .. '_d_' .. opt.delta .. '_n_' .. opt.n_train_set .. '_s_' .. sampling .. '_' .. st, {parameters, geometry})
end

function loadModel(filename, full_output)
   local loaded = torch.load(filename)
   local geometry = loaded[2]
   local model = getModel(geometry, full_output)
   local parameters = model:getParameters()
   parameters:copy(loaded[1])
   return geometry, model
end