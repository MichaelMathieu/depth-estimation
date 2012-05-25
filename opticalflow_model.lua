require 'torch'
require 'xlua'
require 'nnx'
require 'SmartReshape'
require 'common'
require 'CascadingAddTable'
require 'OutputExtractor'
require 'inline'
require 'extractoutput'

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

function yx2xMulti(geometry, y, x)
   x = round(x)
   y = round(y)
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
	 print(x, y)
	 assert(false)
      end
      itarget = geometry.maxw*geometry.maxh
	 + (i-2)*(2*d*geometry.maxw + 2*(geometry.maxh-2*d)*d) + itarget
   end
   return itarget
end

function x2yxMulti(geometry, x)
   if type(x) == 'number' then
      return x2yxMultiNumber(geometry, x)
   else
      return x2yxMulti2(geometry, x)
      --[[
      local retx = torch.Tensor(x:size())
      local rety = torch.Tensor(x:size())
      for i = 1,x:size(1) do
	 for j = 1,x:size(2) do
	    rety[i][j], retx[i][j] = x2yxMultiNumber(geometry, x[i][j])
	 end
      end
      return rety, retx
      --]]
   end
end

function x2yxMulti2(geometry, x)
   local file = io.open("x2yxMulti2.c")
   local process = inline.load(file:read("*all"))
   file:close()
   local retx = torch.LongTensor():resizeAs(x)
   local rety = torch.LongTensor():resizeAs(x)
   print(rety:size())
   process(x, geometry.maxh, geometry.maxw, geometry.ratios, retx, rety)
   return rety, retx
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
   return (y+math.ceil(geometry.maxh/2)), (x+math.ceil(geometry.maxw/2))
end
function onebased2centered(geometry, y, x)
   return (y-math.ceil(geometry.maxh/2)), (x-math.ceil(geometry.maxw/2))
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
  
   function filter:getWeights()
      local weights = {}
      local iLayer = 1
      for i = 1,#self.modules do
	 if self.modules[i].weight then
	    weights['layer'..iLayer] = self.modules[i].weight
	    iLayer = iLayer + 1
	 end
      end
      return weights
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

   function prefilter:getWeights()
      local weights = {}
      if geometry.share_filters then
	 weights = self.modules[1].modules[3]:getWeights()
      else
	 for i = 1,#geometry.ratios do
	    local lweights = self.modules[i].modules[3]:getWeights()
	    for n,w in pairs(lweights) do
	       local name = 'scale'..geometry.ratios[i]..'_'..n
	       weights[name] = w
	    end
	 end
      end
      return weights
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
      model:add(parallel)
   end

   model:add(nn.SpatialMatching(geometry.maxh, geometry.maxw, false))
   model:add(nn.SmartReshape({-1, -2}, {-3, -4}))

   --model:add(nn.SmartReshape({-1,-2},-3))
   model:add(nn.Minus())
   if geometry.training_mode then
      model:add(nn.LogSoftMax())
      model:add(nn.SmartReshape(1,1,-2))
   else
      model:add(nn.SoftMax())
      model:add(nn.SmartReshape(geometry.hImg-geometry.hPatch2+1,
				geometry.wImg-geometry.wPatch2+1,-2))
   end

   function model:getWeights()
      if prefiltered then
	 return {}
      else
	 return self.modules[1].modules[1]:getWeights()
      end
   end

   return model
end

function getModelMultiscale(geometry, full_image, prefiltered)
   local model = nn.Sequential()
   if not geometry.training_mode then
      model:add(nn.Tic('model'))
   end
   if prefiltered == nil then prefiltered = false end
   assert(geometry.ratios[1] == 1)
   local rmax = geometry.ratios[#geometry.ratios]
   for i = 1,#geometry.ratios do
      local k = rmax - geometry.ratios[i]
      assert(math.mod(geometry.maxh * k, 2) == 0)
      assert(math.mod(geometry.maxw * k, 2) == 0)
   end
   local nChannelsIn
   if prefiltered then
      nChannelsIn = geometry.layers[#geometry.layers][4]
   else
      nChannelsIn = geometry.layers[1][1]
   end

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
   
   local matchers = {}
   for i = 1,#geometry.ratios do
      if geometry.share_filters then
	 matchers[i] = matcher:clone('weight', 'bias', 'gradWeight', 'gradBias')
      else
	 matchers[i] = matcher:clone()
      end
   end

   local pyramid = nn.SpatialPyramid(geometry.ratios, matchers,
				     geometry.wPatch2, geometry.hPatch2, 1, 1,
				     3, 2, 2, 1, prefiltered)
   model.pyramid = pyramid

   if not prefiltered then
      model:add(nn.JoinTable(1))
      model:add(nn.FunctionWrapper(function(self)
				      self.padder = nn.SpatialPadding(0,0,0,0)
				   end,
				   function(self, input)
				      -- doesn't work if the ratios have weird ratios
				      local r = geometry.ratios[#geometry.ratios]
				      local targeth = r*math.ceil(input:size(2)/r)
				      local targetw = r*math.ceil(input:size(3)/r)
				      self.padder.pad_b = targeth-input:size(2)
				      self.padder.pad_r = targetw-input:size(3)
				      if (self.padder.pad_b~=0) or (self.padder.par_r~=0) then
					 return self.padder:updateOutput(input)
				      else
					 return input
				      end
				   end,
				   function(self, input, gradOutput)
				      if (self.padder.pad_b~=0) or (self.padder.par_r~=0) then
					 return self.padder:updateGradInput(input, gradOutput)
				      else
					 return gradOutput
				      end
				   end))
   else
      --todo: this is not smart
      local parallel = nn.ParallelTable()
      for i = 1,#geometry.ratios do
	 parallel:add(nn.JoinTable(1))
      end
      model:add(parallel)
   end
   model:add(pyramid)
   if not geometry.training_mode then
      model:add(nn.Toc('model', 'after pyramid'))
   end

   local cascad_preproc = nn.ParallelTable()
   for i = 1,#geometry.ratios do
      local seq = nn.Sequential()
      cascad_preproc:add(seq)
      seq:add(nn.SmartReshape({-1,-2},{-3,-4}))
      seq:add(nn.Minus())
      seq:add(nn.SoftMax())
      seq:add(nn.SmartReshape(-1,geometry.maxh, geometry.maxw))
   end
   model:add(cascad_preproc)

   if not geometry.training_mode then
      model:add(nn.Toc('model', 'after cascad_preproc'))
   end

   local cascad = nn.CascadingAddTable(geometry.ratios, geometry.cascad_trainable_weights,
				      geometry.single_beta)
   model.cascad = cascad
   model:add(cascad)
   if not geometry.training_mode then
      model:add(nn.Toc('model', 'after cascad'))
   end
   
   local postprocessors = nn.ParallelTable()
   postprocessors:add(nn.SmartReshape(-1, {-2, -3}))
   for i = 2,#geometry.ratios do
      local d = round(geometry.maxw*(geometry.ratios[i]-geometry.ratios[i-1])/(2*geometry.ratios[i]))
      local remover1 = nn.Sequential()
      local remover2 = nn.Sequential()
      local remover3 = nn.Sequential()
      local remover4 = nn.Sequential()
      remover1:add(nn.Narrow(2, 1, d))
      remover2:add(nn.Narrow(2, d+1, geometry.maxh-2*d))
      remover2:add(nn.Narrow(3, 1, d))
      remover3:add(nn.Narrow(2, d+1, geometry.maxh-2*d))
      remover3:add(nn.Narrow(3, geometry.maxw-d+1, d))
      remover4:add(nn.Narrow(2, geometry.maxh-d+1, d))
      remover1:add(nn.SmartReshape(-1, {-2, -3}))
      remover2:add(nn.SmartReshape(-1, {-2, -3}))
      remover3:add(nn.SmartReshape(-1, {-2, -3}))
      remover4:add(nn.SmartReshape(-1, {-2, -3}))
      local removers = nn.ConcatTable()
      removers:add(remover1)
      removers:add(remover2)
      removers:add(remover3)
      removers:add(remover4)
      
      local middleRemover = nn.Sequential()
      middleRemover:add(removers)
      middleRemover:add(nn.JoinTable(2))
      postprocessors:add(middleRemover:clone())
   end
   
   model:add(postprocessors)
   model:add(nn.JoinTable(2))
   if not geometry.training_mode then
      model:add(nn.Toc('model', 'after complex reshaping'))
   end
   
   if geometry.training_mode then
      model:add(nn.SmartReshape(1,1,-2))
   else
      model:add(nn.SmartReshape(geometry.hImg, geometry.wImg,-2))
   end

   if geometry.training_mode then
      model:add(nn.Log2(1e-10))
   end

   function model:focus(x, y)
      if not x then
	 pyramid:focus()
      else
	 pyramid:focus(x, y, 1, 1)
      end
   end

   function model:getWeights()
      local weights = {}
      if geometry.cascad_trainable_weights then
	 weights['cascad'] = self.cascad:getWeight()
      end
      if not prefiltered then
	 local processors = self.pyramid.processors
	 if geometry.share_filters then
	    local lweights = processors[1].modules[1].modules[1].modules[3]:getWeights()
	    for n,w in pairs(lweights) do
	       weights[n] = w
	    end
	 else
	    for i = 1,#processors do
	       local lweights = processors[i].modules[1].modules[1].modules[3]:getWeights()
	       for n,w in pairs(lweights) do
		  local name = 'scale'..geometry.ratios[i]..'_'..n
		  weights[name] = w
	       end
	    end
	 end
      end
      return weights
   end
   
   return model
end

function prepareInput(geometry, patch1, patch2)
   assert(sameSize(patch1, patch2))
   if geometry.prefilter then
      assert(patch1:size(1) == geometry.layers[#geometry.layers][4])
   else
      if (geometry.layers[1][1] == 1) and (patch1:size(1) == 3) then
	 patch1 = image.rgb2y(patch1, patch2)
      end
      assert(patch1:size(1) == geometry.layers[1][1])
   end
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

function getOutputConfidences(geometry, input, threshold)
   if not threshold then
      local m, idx = input:max(3)
      m = m:select(3,1)
      idx = idx:select(3,1)
      local middleIndex = getMiddleIndex(geometry)
      local flatPixels = torch.LongTensor(m:size(1), m:size(2)):copy(m:eq(input[{{},{},middleIndex}]))
      idx = flatPixels*middleIndex + (-flatPixels+1):cmul(idx:reshape(idx:size(1),idx:size(2)))
      return idx, torch.Tensor(idx:size()):fill(1)
   else
      local imaxs = torch.LongTensor(input:size(1), input:size(2))
      local gds   = torch.LongTensor(input:size(1), input:size(2))
      extractoutput.extractOutput(input, 0.11, threshold, imaxs, gds)
      return imaxs, gds
   end
end

function getOutputConfidences2(geometry, input)
   local xmul = torch.Tensor(geometry.maxh*geometry.maxw)
   local ymul = torch.Tensor(geometry.maxh*geometry.maxw)
   for i = 1,geometry.maxh do
      for j = 1,geometry.maxw do
	 local k = yx2x(geometry, i, j)
	 ymul[k] = i
	 xmul[k] = j
      end
   end
   xmul = nn.Replicate(input:size(1)):forward(nn.Replicate(input:size(2)):forward(xmul))
   ymul = nn.Replicate(input:size(1)):forward(nn.Replicate(input:size(2)):forward(ymul))
   local x = input:clone():cmul(xmul):sum(3)[{{},{},1}]
   local y = input:cmul(ymul):sum(3)[{{},{},1}]
   return y, x, torch.Tensor(y:size()):fill(1)
end

function processOutput(geometry, output, process_full, threshold)
   local ret = {}
   if geometry.output_extraction_method == 'max' then
      ret.index, ret.confidences = getOutputConfidences(geometry, output, threshold)
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
   else
      assert(not geometry.multiscale)
      ret.y, ret.x, ret.confidences = getOutputConfidences2(geometry, output)
      ret.index = yx2x(geometry, (ret.y+0.5):floor(), (ret.x+0.5):floor())
      local yoffset, xoffset = centered2onebased(geometry, 0, 0)
      ret.y = ret.y - yoffset
      ret.x = ret.x - xoffset
   end

   if process_full then
      local hoffset, woffset
      hoffset = math.floor((geometry.hImg-ret.y:size(1))/2)
      woffset = math.floor((geometry.wImg-ret.y:size(2))/2)
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
	 if ret.confidences then
	    ret.full_confidences = torch.Tensor(geometry.hImg, geometry.wImg):zero()
	    ret.full_confidences:sub(1 + hoffset, ret.y:size(1) + hoffset,
				     1 + woffset, ret.y:size(2) + woffset):copy(
	       ret.confidences)
	 end
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

function prepareTarget(geometry, learning, targett)
   local itarget, target, xtarget, ytarget
   local halfh1 = math.ceil(geometry.maxh/2)-1
   local halfh2 = math.floor(geometry.maxh/2)
   local halfw1 = math.ceil(geometry.maxw/2)-1
   local halfw2 = math.floor(geometry.maxw/2)
   if (targett[1] < -halfh1) or (targett[1] > halfh2) or
      (targett[2] < -halfw1) or (targett[2] > halfw2) then
      xtarget = 0
      ytarget = 0
   else
      xtarget = targett[2]
      ytarget = targett[1]
   end
   if geometry.multiscale then
      itarget = yx2xMulti(geometry, ytarget, xtarget)
   else
      local xtargetO = xtarget + halfw1 + 1
      local ytargetO = ytarget + halfh1 + 1
      itarget = (ytargetO-1) * geometry.maxw + xtargetO
   end
   if learning.soft_targets then
      target = torch.Tensor(geometry.maxh*geometry.maxw)
      local invsigma2 = 1./learning.st_sigma2
      for y = -halfh1, halfh2 do
	 for x = -halfw1, halfw2 do
	    local d2 = (ytarget-y)*(ytarget-y) + (xtarget-x)*(xtarget-x)
	    local g = math.exp(-d2*invsigma2)
	    local i
	    if geometry.multiscale then
	       i = yx2xMulti(geometry, y, x)
	    else
	       local xO = x + halfw1 + 1
	       local yO = y + halfh1 + 1
	       i = (yO-1) * geometry.maxw + xO
	    end
	    target[i] = g
	 end
      end
   else
      target = itarget
   end
   return itarget, target
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
