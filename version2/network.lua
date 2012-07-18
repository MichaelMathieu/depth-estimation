require 'torch'
require 'nnx'
require 'image'

function getNetwork(datap)
   local network = nn.Sequential()
   local filters = nn.ParallelTable()
   network:add(filters)
   
   local filter1 = nn.Sequential()
   filters:add(filter1)
   filter1:add(nn.SpatialContrastiveNormalization(3, image.gaussian1D(datap.normalization_k)))
   filter1:add(nn.SpatialPadding(-datap.lWin, -datap.tWin, -datap.rWin, -datap.bWin))
   filter1:add(nn.SpatialConvolution(3, 10, datap.hKernel, datap.wKernel))
   
   local filter2 = nn.Sequential()
   filters:add(filter2)
   filter2:add(filter1.modules[1]:clone())
   filter2:add(filter1.modules[3]:clone('weight', 'bias', 'gradWeight', 'gradBias'))

   network:add(nn.SpatialMatching(datap.hWin, datap.wWin, false))

   function network:getWeights()
      local weights = {}
      weights.layer1 = self.modules[1].modules[1].modules[3].weight
      return weights
   end
   
   return network
end

function getTrainerNetwork(datap)
   local network = getNetwork(datap)
   network:add(nn.Reshape(datap.wWin*datap.hWin))
   network:add(nn.Minus())
   network:add(nn.LogSoftMax())
   return network
end