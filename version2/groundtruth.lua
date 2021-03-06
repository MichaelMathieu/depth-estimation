require 'torch'
require 'image'
require 'nnx'
require 'extractoutput'
require 'liuflow'

function unfold(img, wKer, hKer)
   local nfeats = wKer*hKer*img:size(1)
   local imgb = img:unfold(2, hKer, 1):unfold(3, wKer, 1)
   local h = imgb:size(2)
   local w = imgb:size(3)
   local imgc = torch.Tensor(nfeats, h, w)
   for i = 1,h do
      for j = 1,w do
	 imgc[{{},i,j}]:copy(imgb[{{},i,j,{},{}}]:reshape(nfeats))
      end
   end
   return imgc
end

function cross_correlation_pad_output(output, wWin, hWin, wKer, hKer)
   local xDim = 3
   local yDim = 2
   if output:nDimension() == 2 then
      xDim = 2
      yDim = 1
   end
   return nn.SpatialPadding(math.floor((wWin-1)/2) + math.floor((wKer-1)/2),
			    math.ceil((wWin-1)/2) + math.ceil((wKer-1)/2),
			    math.floor((hWin-1)/2) + math.floor((hKer-1)/2),
			    math.ceil((hWin-1)/2) + math.ceil((hKer-1)/2),
			    yDim, xDim)(output)
end

function adapt_mask(groundtruthp, mask)
   local wWin = groundtruthp.params.wWin
   local hWin = groundtruthp.params.hWin
   local wKer = groundtruthp.params.wKernel
   local hKer = groundtruthp.params.hKernel
   local h = mask:size(1)
   local w = mask:size(2)
   local newmask = torch.Tensor(mask:size()):zero()
   local lshift = math.floor((wWin-1)/2)+math.floor((wKer-1)/2)
   if lshift > 0 then
      newmask:sub(1, h, lshift+1, w):add(mask:sub(1, h, 1, w-lshift))
   end
   local rshift = math.ceil((wWin-1)/2)+math.ceil((wKer-1)/2)
   if rshift > 0 then
      newmask:sub(1, h, 1, w-rshift):add(mask:sub(1, h, 1+rshift, w))
   end
   local tshift = math.floor((hWin-1)/2)+math.floor((hKer-1)/2)
   if tshift > 0 then
      newmask:sub(tshift+1, h, 1, w):add(mask:sub(1, h-tshift, 1, w))
   end
   local bshift = math.ceil((hWin-1)/2)+math.ceil((hKer-1)/2)
   if bshift > 0 then
      newmask:sub(1, h-bshift, 1, w):add(mask:sub(1+bshift, h, 1, w))
   end
   newmask = torch.Tensor(newmask:size()):copy(newmask:gt(3.9))
   return newmask
end
   

function compute_groundtruth_cross_correlation(groundtruthp, img1, img2, mask)
   assert(groundtruthp.type == 'cross-correlation')
   local hWin = groundtruthp.params.hWin
   local wWin = groundtruthp.params.wWin
   local hKer = groundtruthp.params.hKernel or groundtruthp.params.hKer
   local wKer = groundtruthp.params.wKernel or groundtruthp.params.wKer
   mask = mask or torch.Tensor(img1[1]:size()):fill(1)
   mask = adapt_mask(groundtruthp, mask)
   -- CAREFUL this mask represents the pixels where the groundtruth is computed.
   -- it is not the same as the masks computed in radial_opticalflow_data, which are
   -- the pixels that can be used to compute the groundtruth (or anything)
   
   -- match
   local img1uf = unfold(img1, wKer, hKer)
   local img2uf = unfold(img2, wKer, hKer)
   local padder = nn.SpatialPadding(-math.floor((wWin-1)/2), -math.ceil((wWin-1)/2),
   			            -math.floor((hWin-1)/2), -math.ceil((hWin-1)/2))
   local matcher = nn.SpatialMatching(hWin, wWin, false)
   local output = matcher({padder(img1uf), img2uf})
   
   -- get min
   output = output:reshape(output:size(1), output:size(2), hWin*wWin)
   local m,idx = output:min(3)
   m = m[{{},{},1}]
   idx = torch.Tensor(idx[{{},{},1}]:size()):copy(idx)
   local middleidx = math.ceil(wWin/2) + wWin*(math.ceil(hWin/2)-1)
   local flat = m:eq(output[{{},{},middleidx}])
   flat = torch.Tensor(flat:size()):copy(flat)
   idx = flat*middleidx + (-flat+1):cmul(idx)

   -- get x, y
   local floored = ((idx-1)/wWin):floor()
   local flow = torch.Tensor(4, idx:size(1), idx:size(2))
   flow[1] = floored - math.floor((hWin-1)/2)              -- y
   flow[2] = idx-1 - floored*wWin - math.floor((wWin-1)/2) -- x
   
   --get confidences
   local scores = torch.Tensor(flow:size(2), flow:size(3))
   local imaxs = torch.LongTensor(flow:size(2), flow:size(3))
   extractoutput.extractOutput(output, scores, 0.21, imaxs)
   flow[3]:fill(1)                                     -- mask
   flow[4]:copy(scores)
   local flowp = cross_correlation_pad_output(flow, wWin, hWin, wKer, hKer)
   flowp[3]:cmul(mask)
   
   return flowp[{{1,2}}], flowp[3]
end


function cartesian_groundtruth_cc_testme()
   local function test(w, h, hKer, wKer, hWin, wWin, flowbase, noise)
      local im2 = torch.rand(30, h, w)
      --warp is confusing:
      -- im1(i,j) = im2(i+dx, j+dy)
      local im1 = image.warp(im2, flowbase, 'nearest', true) + torch.randn(30, h, w):mul(noise)
      local groundtruthp = {
	 type='cross-correlation',
	 params = {
	    hWin=hWin,
	    wWin=wWin,
	    hKer=hKer,
	    wKer=wKer
	 }
      }
      local flow = compute_cartesian_groundtruth_cross_correlation(groundtruthp, im1, im2)
      local diff = (flowbase[{{1,2}}]-flow[{{1,2}}])
      diff[1]:cmul(flow[3])
      diff[2]:cmul(flow[3])
      if diff:abs():sum() ~= 0 then
	 image.display(diff)
	 print('failed')
      end
   end
   local w = 42
   local h = 32
   local flowbaseEven = (torch.rand(2, h, w)*12-5):floor()
   test(w, h, 1, 1, 12, 15, flowbaseEven, 0)
   local flowbaseOdd = (torch.rand(2, h, w)*15-7):floor()
   test(w, h, 1, 1, 17, 15, flowbaseOdd, 0)
   local kh = 1
   local kw = 1
   local flowbaseKer = torch.Tensor(2,h,w)
   flowbaseKer[1]:fill(math.floor(torch.rand(1):squeeze()*17-8+0.5))
   flowbaseKer[2]:fill(math.floor(torch.rand(1):squeeze()*17-8+0.5))
   test(w, h, 3, 3, 17, 17, flowbaseKer, 0.5)
   local flowbaseKer2 = torch.Tensor(2,h,w)
   flowbaseKer2[1]:fill(math.floor(torch.rand(1):squeeze()*12-5+0.5))
   flowbaseKer2[2]:fill(math.floor(torch.rand(1):squeeze()*7-4+0.5))
   test(w, h, 5, 5, 17, 17, flowbaseKer, 1)
end

function compute_groundtruth_liu(groundtruthp, img1, img2)
   assert(groundtruthp.type == 'liu')
   local alpha = groundtruthp.params.alpha
   local ratio = groundtruthp.params.ratio
   local minWidth = groundtruthp.params.winWidth
   local nOuterFPIterations = groundtruthp.params.nOFPIters
   local nInnerFPIterations = groundtruthp.params.nIFPIters
   local nCGIterations = groundtruthp.params.nGCIters
   
   local resn, resa, warp, resx, resy = liuflow.infer{pair={img1, img2}, alpha=alpha,
						      ratio=ratio, minWidth=minWidth,
						      nOuterFPIterations=nOuterFPIterations,
						      nInnerFPIterations=nInnerFPIterations,
						      nCGIterations=nCGIterations}
   local flow = torch.Tensor(3, img1:size(2), img1:size(3))
   flow[1]:copy(resy)
   flow[2]:copy(resx)
   flow[3]:fill(1)
   return flow
end