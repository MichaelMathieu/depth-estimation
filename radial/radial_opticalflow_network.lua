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

function getWeights(network)
   local weights = {}
   local bias = {}
   local layers = network.modules[1].modules[2].modules
   for i = 1,#layers do
      if layers[i].weight then
	 table.insert(weights, layers[i].weight)
      end
      if layers[i].bias then
	 table.insert(bias, layers[i].bias)
      end
   end
   return weights, bias
end

function copyWeights(srcnetwork, dstnetwork)
   local srcweights, srcbias
   if type(srcnetwork) == 'table' and #srcnetwork == 2 then
      srcweights = srcnetwork[1]
      srcbias = srcnetwork[2]
   else
      srcweights, srcbias = getWeights(srcnetwork)
   end
   local dstweights, dstbias = getWeights(dstnetwork)
   assert(#srcweights == #dstweights)
   assert(#dstbias == #dstbias)
   for i = 1,#srcweights do
      dstweights[i]:copy(srcweights[i])
   end
   for i = 1,#srcbias do
      dstbias[i]:copy(srcbias[i])
   end
end

function displayWeights(network, wins)
   wins = wins or {}
   local weights, _ = getWeights(network)
   for i = 1,#weights do
      local w = weights[i]
      if w:size(2) ~= 3 then
	 w = w:reshape(w:size(1)*w:size(2), w:size(3), w:size(4))
      end
      wins[i] = image.display{image = w, padding=2, zoom=4, win = wins[i]}
   end
end

local current_version = 1
function saveNetwork(dir, iEpoch, networkp, network)
   if dir:sub(-1) ~= '/' then dir = dir..'/' end
   filename = dir .. 'model_' .. iEpoch
   local tosave = {}
   tosave.version = current_version
   tosave.networkp = networkp
   tosave.weights = {}
   tosave.weights[1], tosave.weights[2] = getWeights(network)
   torch.save(filename, tosave)
end

function checkVersion(loaded)
   if loaded.version ~= current_version then
      error('Input file has version '.. loaded.version.. ' but is required to have version '..current_version)
   end
end

function loadTrainerNetwork(filename)
   local loaded = torch.load(filename)
   checkVersion(loaded)
   local networkp = loaded.networkp
   local network = getTrainerNetwork(networkp)
   copyWeights(loaded.wrights, network)
   return network, networkp
end

function loadTesterNetwork(filename)
   local loaded = torch.load(filename)
   checkVersion(loaded)
   local networkp = loaded.networkp
   local network = getTesterNetwork(networkp)
   copyWeights(loaded.weights, network)
   return network, networkp
end