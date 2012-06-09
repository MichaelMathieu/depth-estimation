require 'torch'
require 'cartesian2polar'
require 'image'
require 'nnx'
torch.setdefaulttensortype('torch.FloatTensor')

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

function compute_cartesian_groundtruth_cross_correlation(groundtruthp, img1, img2)
   assert(groundtruthp.type == 'cross-correlation')
   local hWin = groundtruthp.params.hWin
   local wWin = groundtruthp.params.wWin
   local hKer = groundtruthp.params.hKer
   local wKer = groundtruthp.params.wKer
   
   local img1uf = unfold(img1, wKer, hKer)
   local img2uf = unfold(img2, wKer, hKer)
   
   local padder = nn.SpatialPadding(-math.floor((wWin-1)/2), -math.ceil((wWin-1)/2),
   			            -math.floor((hWin-1)/2), -math.ceil((hWin-1)/2))
   local matcher = nn.SpatialMatching(hWin, wWin, false)
   local output = matcher({padder(img1uf), img2uf})
   output = output:reshape(output:size(1), output:size(2), hWin*wWin)
   local m,idx = output:min(3)
   m = m[{{},{},1}]
   idx = torch.Tensor(idx[{{},{},1}]:size()):copy(idx)
   local middleidx = math.ceil(wWin/2) + wWin*(math.ceil(hWin/2)-1)
   local flat = m:eq(output[{{},{},middleidx}])
   flat = torch.Tensor(flat:size()):copy(flat)
   idx = flat*middleidx + (-flat+1):cmul(idx)
   local floored = ((idx-1)/wWin):floor()
   local flow = torch.Tensor(3, idx:size(1), idx:size(2))
   flow[1] = floored - math.floor((hWin-1)/2)              -- y
   flow[2] = idx-1 - floored*wWin - math.floor((wWin-1)/2) -- x
   flow[3]:fill(1)                                         -- mask
   
   local flowp = cross_correlation_pad_output(flow, wWin, hWin, wKer, hKer)
   return flowp
end

function cartesian_groundtruth_cc_testme()
   torch.manualSeed(1)
   local function test(w, h, hKer, wKer, hWin, wWin, flowbase, noise)
      local im2 = torch.rand(30, h, w)
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