require 'torch'
require 'nn'
require 'nnx'
require 'SmartReshape'

function getFilter(networkp)
   local filter = nn.Sequential()
   local last_n_output = nil
   for i = 1,#networkp.layers do
      local layer = networkp.layers[i]
      if layer == 'tanh' then
	 filter:add(nn.Tanh())
      elseif type(layer) == 'table' then
	 if last_n_output == nil then
	    filter:add(nn.SpatialConvolution(layer[1], layer[4], layer[3], layer[2]))
	 elseif layer[1] == last_n_output then
	    filter:add(nn.SpatialConvolution(layer[1], layer[4], layer[3], layer[2]))
	 else
	    filter:add(nn.SpatialConvolutionMap(nn.tables.random(last_n_output,
								 layer[4], layer[1]),
						layer[3], layer[2]))
	 end
	 last_n_output = layer[4]
      else
	 print(layer)
	 error('Unknown layer')
      end
   end
   return filter
end

function getMatcher(networkp)
   return nn.SpatialMatching(networkp.hWin, 1, false)
end

function getTrainerNetwork(networkp)
   local network = nn.Sequential()
   local filters = nn.ParallelTable()
   local padder = nn.SpatialPadding(0, 0, 0, -networkp.hWin+1)
   local seq_prev = nn.Sequential()
   seq_prev:add(padder)
   local filter = getFilter(networkp)
   seq_prev:add(filter)
   filters:add(seq_prev)
   filters:add(filter:clone('weight', 'bias', 'gradWeight', 'gradBias'))
   network:add(filters)
   local matcher = getMatcher(networkp)
   network:add(matcher)
   network:add(nn.Reshape(1, networkp.hWin))
   network:add(nn.Minus())
   network:add(nn.LogSoftMax())
   network:add(nn.Reshape(networkp.hWin))
   return network
end

function getTesterNetwork(networkp)
   local network = nn.Sequential()
   local filters = nn.ParallelTable()
   local padder = nn.SpatialPadding(0, 0, 0, -networkp.hWin+1)
   local seq_prev = nn.Sequential()
   seq_prev:add(padder)
   local filter = getFilter(networkp)
   seq_prev:add(filter)
   filters:add(seq_prev)
   filters:add(filter:clone('weight', 'bias', 'gradWeight', 'gradBias'))
   network:add(filters)
   local matcher = getMatcher(networkp)
   network:add(matcher)
   network:add(nn.SmartReshape(-1, -2, networkp.hWin))
   return network
end