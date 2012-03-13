require 'torch'
require 'xlua'
require 'nnx'

function yx2x(geometry, y, x)
   return (x-1) * geometry.maxw + y
end

function x2yx(geometry, x)
   if type(x) == 'number' then
      return (math.floor((x-1)/geometry.maxw)+1), (math.mod(x-1, geometry.maxw)+1)
   else
      local xdbl = torch.DoubleTensor(x:size()):copy(x-1)-1
      local xout = (xdbl/geometry.maxw):floor()
      local yout = xdbl - xout*geometry.maxw
      return (yout+1.5):floor(), (xout+1.5):floor() --(a+0.5):floor() is a:round()
   end
end

function getModel(geometry, full_image)
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
   
   model:add(nn.SpatialMatching(geometry.maxh, geometry.maxw, false))
   model:add(nn.Minus())
   
   if full_image then
      model:add(nn.Reshape(geometry.maxw*geometry.maxh,
			   geometry.hImg - geometry.hKernel - geometry.maxh + 2,
			   geometry.wImg - geometry.wKernel - geometry.maxw + 2))
   else
      model:add(nn.Reshape(geometry.maxw*geometry.maxh,
			   geometry.hPatch2 - geometry.hKernel - geometry.maxh + 2,
			   geometry.wPatch2 - geometry.wKernel - geometry.maxw + 2))
   end
   local spatial = nn.SpatialClassifier()
   spatial:add(nn.LogSoftMax())
   model:add(spatial)
   
   return model
end

function prepareInput(geometry, patch1, patch2)
   ret = {}
   ret[1] = patch1:narrow(2, math.ceil(geometry.maxh/2), geometry.hPatch1)
                  :narrow(3, math.ceil(geometry.maxw/2), geometry.wPatch1)
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
