require 'torch'
require 'xlua'
require 'nnx'
require 'SmartReshape'
require 'common'
require 'CascadingAddTable'
require 'OutputExtractor'
require 'inline'
require 'extractoutput'
require 'opticalflow_model_multiscale'

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
      if i == 1 then
	 filter:add(nn.SpatialConvolution(geometry.layers[i][1], geometry.layers[i][4],
					  geometry.layers[i][2], geometry.layers[i][3]))
      elseif geometry.layers[i-1][4] == geometry.layers[i][1] then
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
   model:add(nn.Minus())
   
   model:add(nn.FunctionWrapper(function(self)
				   self.seq = nn.Sequential()
				   self.seq:add(nn.SmartReshape({-1, -2}, {-3, -4}))
				   self.seq:add(nn.SoftMax())
				   self.seq:add(nn.SmartReshape(1, 1, -2))
				end,
				function(self, input)
				   self.seq.modules[3].sizes[1] = input:size(1)
				   self.seq.modules[3].sizes[2] = input:size(2)
				   return self.seq:updateOutput(input)
				end,
				function(self, input, gradOutput)
				   return self.seq:updateGradInput(input, gradOutput)
				end))

   if geometry.output_extraction_method == 'mean' then
      model:add(nn.OutputExtractor(geometry.maxh, geometry.maxw))
   else
      assert(geometry.output_extraction_method == 'max')
      if geometry.training_mode then
	 model:add(nn.Log())
      end
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
   local y = input:clone():cmul(ymul):sum(3)[{{},{},1}]

   local imaxs = torch.LongTensor(input:size(1), input:size(2))
   --local gds   = torch.LongTensor(input:size(1), input:size(2))
   --extractoutput.extractOutput(input, 0.11, 0, imaxs, gds)
   local xgds   = torch.LongTensor(input:size(1), input:size(2))
   --local ygds   = torch.LongTensor(input:size(1), input:size(2))
   local inputmarg = input:reshape(input:size(1), input:size(2), geometry.maxh, geometry.maxw):sum(4):squeeze()
   extractoutput.extractOutput(inputmarg, 0.11, 0, imaxs, xgds)
   --extractoutput.extractOutput(input:sum(3), 0.11, 0, imaxs, ygds)
   gds = xgds

   return y, x, gds
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
      ret.index = ret.index:squeeze()
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


function postProcessImage(input, mask, winsize, method)
   local output = torch.Tensor(2, input[1]:size(1), input[1]:size(2)):zero()
   --local winsizeh1 = math.ceil(winsize/2)-1
   --local winsizeh2 = math.floor(winsize/2)
   --local win = torch.Tensor(2,winsize,winsize)
   inline.preamble [[
	 int comp(const void* a_,const void* b_) {
	    float a = *((float*)a_), b = *((float*)b_);
	    if (a==b) {
	       return 0;
	    } else {
	       if (a < b) {
		  return -1;
	       } else {
		  return 1;
	       }
	    }
	 }
   ]]
   local fmax = inline.load [[
	 const void* idfloat = luaT_checktypename2id(L, "torch.FloatTensor");
	 THFloatTensor* flow = (THFloatTensor*)luaT_checkudata(L, 1, idfloat);
	 THFloatTensor* mask = (THFloatTensor*)luaT_checkudata(L, 2, idfloat);
	 int k = lua_tointeger(L, 3);
	 THFloatTensor* ret = (THFloatTensor*)luaT_checkudata(L, 4, idfloat);
	 
	 int h = flow->size[1];
	 int w = flow->size[2];
	 long* fs = flow->stride;
	 float* flow_p = THFloatTensor_data(flow);
	 long* ms = mask->stride;
	 float* mask_p = THFloatTensor_data(mask);
	 long* rs = ret->stride;
	 float* ret_p = THFloatTensor_data(ret);
	 int halfk = k/2;

	 const int TMPSIZE = 256;
	 const int ROWSIZE = 16;
	 int tmp[TMPSIZE];

	 int i, j, ik, jk, l;
	 for (i = 0; i < h-k; ++i) {
	    for (j = 0; j < w-k; ++j) {
	       memset(tmp, 0, TMPSIZE*sizeof(int));
	       for (ik = i; ik < i+k; ++ik) {
		  for (jk = j; jk < j+k; ++jk) {
		     if (mask_p[ik*ms[0] + jk*ms[1] ]) {
			int vx = flow_p[fs[0] + ik*fs[1] + jk*fs[2] ];
			int vy = flow_p[ik*fs[1] + jk*fs[2] ];
			int v = vx+ROWSIZE*vy;
			++tmp[v];
		     }
		  }
	       }
	       int im = 0;
	       for (l = 0; l < TMPSIZE; ++l) {
		  if (tmp[l] > tmp[im])
		     im = l;
	       }
	       ret_p[rs[0] + (i+halfk)*rs[1] + (j+halfk)*rs[2] ] = im%%ROWSIZE;
	       ret_p[        (i+halfk)*rs[1] + (j+halfk)*rs[2] ] = im/ROWSIZE;
	    }
	 }
   ]]

   local fmed = inline.load [[
	 const void* idfloat = luaT_checktypename2id(L, "torch.FloatTensor");
	 THFloatTensor* flow = (THFloatTensor*)luaT_checkudata(L, 1, idfloat);
	 THFloatTensor* mask = (THFloatTensor*)luaT_checkudata(L, 2, idfloat);
	 int k = lua_tointeger(L, 3);
	 THFloatTensor* ret = (THFloatTensor*)luaT_checkudata(L, 4, idfloat);
	 
	 int h = flow->size[1];
	 int w = flow->size[2];
	 long* fs = flow->stride;
	 float* flow_p = THFloatTensor_data(flow);
	 long* ms = mask->stride;
	 float* mask_p = THFloatTensor_data(mask);
	 long* rs = ret->stride;
	 float* ret_p = THFloatTensor_data(ret);
	 int halfk = k/2;

	 const int TMPSIZE = 32;
	 const int ROWSIZE = 16;
	 float tmp[TMPSIZE];
	 float tmp2[TMPSIZE];

	 int i, j, ik, jk, l;
	 for (i = 0; i < h-k; ++i) {
	    for (j = 0; j < w-k; ++j) {
	       memset(tmp, 0, TMPSIZE*sizeof(int));
	       memset(tmp2, 0, TMPSIZE*sizeof(int));
	       int n = 0;
	       for (ik = i; ik < i+k; ++ik) {
		  for (jk = j; jk < j+k; ++jk) {
		     if (mask_p[ik*ms[0] + jk*ms[1] ]) {
			float vx = flow_p[fs[0] + ik*fs[1] + jk*fs[2] ];
			float vy = flow_p[ik*fs[1] + jk*fs[2] ];
			tmp[n] = vy;
			tmp2[n++] = vx;
		     }
		  }
	       }
	       qsort(tmp, n, sizeof(float), comp);
	       qsort(tmp2, n, sizeof(float), comp);
	       float im = tmp[n/2];
	       float im2 = tmp2[n/2];
	       ret_p[rs[0] + (i+halfk)*rs[1] + (j+halfk)*rs[2] ] = im2;
	       ret_p[        (i+halfk)*rs[1] + (j+halfk)*rs[2] ] = im;
	    }
	 }
   ]]
   inline.default_preamble()
   if method == 'max' then
      local inputR = (input+0.5):floor()
      m = inputR:min()
      fmax(inputR-m, mask, winsize, output)
      output = output+m
   else
      fmed(input, mask, winsize, output)
      output = output
   end

   --[[
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
   --]]
   return output
end
