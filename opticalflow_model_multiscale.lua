require 'torch'
require 'xlua'
require 'nnx'
require 'SmartReshape'
require 'common'
require 'CascadingAddTable'
require 'inline'
require 'extractoutput'

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

function getModelMultiscale(geometry, full_image, prefiltered)
   assert(geometry.output_extraction_method == 'max')
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