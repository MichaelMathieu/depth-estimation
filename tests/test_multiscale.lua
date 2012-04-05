require 'groundtruth_opticalflow'
require 'image'
require 'opticalflow_model'

function asserteq(a, b)
   assert(type(a) == type(b))
   if type(a) == 'number' then
      assert(a == b)
   else
      assert(a:numel() == b:numel())
      assert((a:eq(b)):sum() == a:numel())
   end
end

geometry = {}
geometry.wImg=320
geometry.hImg=180
geometry.hKernel=8
geometry.wKernel=8
geometry.layers = {{3,geometry.hKernel, geometry.wKernel, geometry.hKernel*geometry.wKernel*3}}
geometry.hKernelGT=16
geometry.wKernelGT=16
geometry.maxhGT=16
geometry.maxwGT=16
geometry.multiscale=true
geometry.ratios={1,2}
geometry.maxh=8
geometry.maxw=8
geometry.wPatch2=geometry.maxw+geometry.wKernel-1
geometry.hPatch2=geometry.maxh+geometry.hKernel-1

--torch.manualSeed(1)

nSamples = 100

raw_data = loadDataOpticalFlow(geometry, 'data/', 10, '000000000', 1, false)
trainData = generateDataOpticalFlow(geometry, raw_data, nSamples,
				    'uniform_position', false)

model = getModelMultiscale(geometry, false)
--model:focus(1,1,1,1)
--print(model)
pyramid = model.modules[2].focused_pipeline

for i = 1,#pyramid.modules do
   local conv = pyramid.modules[i].modules[3].modules[1].modules[1].modules[3].modules[1]
   local weights = conv.weight
   for i = 1,geometry.hKernel do
      for j = 1,geometry.wKernel do
	 for k = 1,3 do
	    weights[(i-1)*geometry.wKernel*3+(j-1)*3+k]:zero()
	    weights[(i-1)*geometry.wKernel*3+(j-1)*3+k][k][i][j] = 1
	 end
      end
   end
end

-- test x2yx/yx2x
local mh = geometry.maxh*geometry.ratios[#geometry.ratios]
local mw = geometry.maxw*geometry.ratios[#geometry.ratios]
for i = -math.ceil(mh/2)+1, math.floor(mh/2) do
   for j = -math.ceil(mw/2)+1, math.floor(mw/2) do
      y, x = x2yxMulti(geometry, yx2xMulti(geometry, i, j))
      local tolerance
      for r = 1,#geometry.ratios do
	 if math.abs(i) < geometry.maxh*geometry.ratios[r] and
	    math.abs(j) < geometry.maxw*geometry.ratios[r] then
	    tolerance = geometry.ratios[r]
	 end
      end
      assert(math.abs(y-i) < tolerance and math.abs(x-j) < tolerance)
   end
end
local maxx = geometry.maxh*geometry.maxw
for i = 2,#geometry.ratios do
   maxx = maxx + geometry.maxh * geometry.maxw *
      (1 - math.pow(geometry.ratios[i-1]/geometry.ratios[i], 2))
end
for i = 1,maxx do
   assert(yx2xMulti(geometry, x2yxMulti(geometry, i)) == i)
end

for iSample = 1,nSamples do
   xlua.progress(iSample, nSamples)
   local sample = trainData:getElemFovea(iSample)
   local x = sample[1][2][2]
   local y = sample[1][2][1]
   local input = sample[1][1]
   model:focus(x, y)
   local targetCrit, target = prepareTarget(geometry, sample[2])
   local output = model:forward(input)
   assert(output:size(1) == maxx)
   processed_output = processOutput(geometry, output)

   -- check the groundtruth
   local l1x = math.ceil(geometry.wKernelGT/2)-1
   local l2x = math.floor(geometry.wKernelGT/2)
   local l1y = math.ceil(geometry.hKernelGT/2)-1
   local l2y = math.floor(geometry.hKernelGT/2)
   local px1 = input[1][{{}, {y-l1y,y+l2y}, {x-l1x,x+l2x}}]
   local px2 = input[2][{{}, {y+sample[2][1]-l1y, y+sample[2][1]+l2y},
			 {x+sample[2][2]-l1x, x+sample[2][2]+l2x}}]
   local score = (px1-px2):pow(2):sum()
   for i = -math.ceil(geometry.maxhGT/2)+1, math.floor(geometry.maxhGT/2) do
      for j = -math.ceil(geometry.maxwGT/2)+1, math.floor(geometry.maxwGT/2) do
	 px1 = input[1][{{}, {y-l1y,y+l2y}, {x-l1x,x+l2x}}]
	 px2 = input[2][{{}, {y+i-l1y, y+i+l2y}, {x+j-l1x, x+j+l2x}}]
	 assert((px1-px2):pow(2):sum() >= score)
      end
   end
   
   for i = 1,#pyramid.modules do
      -- check that the extracted patches are correctly centered
      local pyr_crop = pyramid.modules[i].modules[1].output
      local center_px = pyr_crop[{{}, math.ceil(pyr_crop:size(2)/2),
				  math.ceil(pyr_crop:size(3)/2)}]
      asserteq(center_px[{{1, 3}}], input[1][{{}, y, x}])
      asserteq(center_px[{{4, 6}}], input[2][{{}, y, x}])

      local filter_crop1b=pyramid.modules[i].modules[3].modules[1].modules[1].modules[1].output
      local filter_crop1 =pyramid.modules[i].modules[3].modules[1].modules[1].modules[2].output
      local filter_crop2 =pyramid.modules[i].modules[3].modules[1].modules[2].modules[1].output
      --important : since we want the patches to be centered on ceil-1 AFTER the matching,
      -- they have to be centered to floor before. (and the +1 are because of the 1-based)
      asserteq(filter_crop1[{{}, math.floor(filter_crop1:size(2)/2)+1,
			     math.floor(filter_crop1:size(3)/2)+1}],
	       filter_crop1b[{{}, math.floor(filter_crop1b:size(2)/2)+1,
			      math.floor(filter_crop1b:size(3)/2)+1}])
      if geometry.ratios[i] == 1 then
	 asserteq(filter_crop1[{{}, math.floor(filter_crop1:size(2)/2)+1,
				math.floor(filter_crop1:size(3)/2)+1}], center_px[{{1,3}}])
	 asserteq(filter_crop2[{{}, math.floor(filter_crop2:size(2)/2)+1,
				math.floor(filter_crop2:size(3)/2)+1}], center_px[{{4,6}}])
      end

      -- check the matching

      -- compute 'groundtruth' for that particular matching
      local l1x = math.ceil(geometry.wKernel/2)-1
      local l2x = math.floor(geometry.wKernel/2)
      local l1y = math.ceil(geometry.hKernel/2)-1
      local l2y = math.floor(geometry.hKernel/2)
      local px1 = filter_crop1
      local score = 1e25
      local ibest = 0
      local jbest = 0
      --for i = -math.ceil(geometry.maxhGT/2)+1, math.floor(geometry.maxhGT/2) do
      --for j = -math.ceil(geometry.maxwGT/2)+1, math.floor(geometry.maxwGT/2) do
      --px2 = input[2][{{}, {i-l1y, i+l2y}, {j-l1x, j+l2x}}]
      for i = 1, geometry.maxh do
	 for j = 1, geometry.maxw do
	    px2 = filter_crop2[{{}, {i, i+geometry.hKernel-1}, {j, j+geometry.wKernel-1}}]
	    if (px1-px2):pow(2):sum() < score then
	       ibest = i
	       jbest = j
	       score = (px1-px2):pow(2):sum()
	    end
	 end
      end

      -- actually check the matching
      local filter_output = pyramid.modules[i].output:squeeze()
      local _, m = filter_output:min(1)
      m = m:squeeze()
      local ymin = math.floor((m-1) / geometry.maxw)+1
      local xmin = math.mod(m-1, geometry.maxw)+1
      assert(ymin == ibest and xmin == jbest)
   end

   -- check cascading
   local filters_out = model.modules[3].output
   local cascad_out = model.modules[4].output
   asserteq(cascad_out[#cascad_out], filters_out[#cascad_out])
   local h = cascad_out[1]:size(1)
   local w = cascad_out[1]:size(2)
   local cy = math.ceil(h/2)
   local cx = math.ceil(w/2)
   local s = torch.Tensor(h, w)
   for i = 1,#cascad_out do
      s:zero()
      for ii = -cy+1,cy do
	 for jj = -cx+1,cx do
	    for j = i,#cascad_out do
	       local r = geometry.ratios[j] / geometry.ratios[i]
	       s[ii+cy][jj+cx] = s[ii+cy][jj+cx] +
		  filters_out[j][math.ceil(ii/r)+cy][math.ceil(jj/r)+cx]:squeeze()
	    end
	 end
      end

      s = s / (#cascad_out-i+1)
      assert(s[cy][cx] == cascad_out[i][cy][cx]:squeeze())
      asserteq(s, cascad_out[i]:squeeze())
   end
   
   -- check complex reshaping
   local reshaper = model.modules[5]
   asserteq(reshaper.modules[1].output, cascad_out[1])
   for i = 2,#geometry.ratios do
      local blockt = cascad_out[i]
      local liw = round(geometry.maxw*geometry.ratios[i-1]/geometry.ratios[i])
      local dw = round((geometry.maxw - liw)/2)
      local lih = round(geometry.maxh*geometry.ratios[i-1]/geometry.ratios[i])
      local dh = round((geometry.maxh - lih)/2)
      blockt:narrow(1, dh+1, lih):narrow(2, dw+1, liw):zero()
      local reshaped = reshaper.modules[i].output
      local block = torch.Tensor(blockt:size()):zero()
      local hb = block:size(3)
      local wb = block:size(4)
      block:narrow(1, 1, dh):copy(reshaped:narrow(1, 1, dh*geometry.maxw):reshape(dh, geometry.maxw, hb, wb))
      block:narrow(1, dh+1, lih):narrow(2, 1, dw):copy(reshaped:narrow(1, dh*geometry.maxw+1, lih*dw):reshape(lih, dw, hb, wb))
      block:narrow(1, dh+1, lih):narrow(2, dw+lih+1, dw):copy(reshaped:narrow(1, dh*geometry.maxw+lih*dw+1, lih*dw):reshape(lih, dw, hb, wb))
      block:narrow(1, dh+lih+1, dh):copy(reshaped:narrow(1, dh*geometry.maxw+2*lih*dw+1, dh*geometry.maxw):reshape(dh, geometry.maxw, hb, wb))
      asserteq(block, blockt)
   end

   -- check argmin
   local _, m = model.modules[6].output:min(1)
   asserteq(processed_output.index, m:squeeze())
   
   -- check x2yx
   
end