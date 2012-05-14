--package.path = "./?.lua;../?.lua;~/local/share/torch/lua/?.lua;~/local/share/torch/lua/?/init.lua;~/local/lib/torch/?.lua;~/local/lib/torch/?/init.lua"
--package.cpath = "./?.so;~/local/lib/torch/?.so;~/local/lib/torch/loadall.so"

require 'torch'
require 'opticalflow_model'
require 'opticalflow_model_io'
require 'openmp'
require 'image_camera'
require 'image_loader'
require 'sfm2'
require 'inline'

openmp.setDefaultNumThreads(2)

local input_model = 'model'
local camera_idx = 1

local loaded = loadModel(input_model, true, true)
local model = loaded.model
model.modules[4] = nn.SoftMax()
local filter = loaded.filter
local geometry = loaded.geometry
geometry.prefilter = true
local K = torch.Tensor(3,3)
K[1][1] = 293.824707
K[1][2] = 0.
K[1][3] = 310.435730
K[2][1] = 0.
K[2][2] = 300.631012
K[2][3] = 251.624924
K[3][1] = 0.
K[3][2] = 0.
K[3][3] = 1.
local distP = torch.Tensor(5)
distP[1] = -0.37994
distP[2] = 0.212737
distP[3] = 0.003098
distP[4] = 0.00087
distP[5] = -0.069770
local Kf = torch.FloatTensor(K:size()):copy(K)

local Khalf = Kf:clone():mul(0.5)
Khalf[3][3] = 1.0

--local cam = ImageCamera
--cam:init(geometry, camera_idx)
local cam = ImageLoader
cam:init(geometry, 'data/ardrone1', 1, 1)

local last_filtered = nil
local last_im = cam:getNextFrame()
last_im = sfm2.undistortImage(last_im, K, distP)
last_im_scaled = image.scale(last_im, geometry.wImg, geometry.hImg)
last_filtered = filter:forward(last_im_scaled):clone()

collectgarbage('stop')

function enlargeMask(mask, ix, iy)
   local f = inline.load [[
	 #define min(a,b) (((a)<(b))?(a):(b))
	 #define max(a,b) (((a)>(b))?(a):(b))
	 const void* iddouble = luaT_checktypename2id(L, "torch.DoubleTensor");
	 THDoubleTensor* mask = (THDoubleTensor*)luaT_checkudata(L, 1, iddouble);
	 int ix = lua_tonumber(L, 2);
	 int iy = lua_tonumber(L, 3);
	 
	 int h = mask->size[0];
	 int w = mask->size[1];
	 double* mask_p = THDoubleTensor_data(mask);
	 long* ms = mask->stride;
	 
	 int i, j, k;
	 for (i = 0; i < h; ++i) {
	    for (j = 0; j < w; ++j) {
	       if (mask_p[ms[0]*i + ms[1]*j] > 0.5) {
		  for (k = j; k < min(j+ix, w); ++k) {
		     mask_p[ms[0]*i + ms[1]*k] = 0.0;
		  }
		  break;
	       }
	    }
	    for (j = w-1; j >= 0; j -= 1) {
	       if (mask_p[ms[0]*i + ms[1]*j] > 0.5) {
		  for (k = j; k >= max(j-ix+1, 0); k -= 1) {
		     mask_p[ms[0]*i + ms[1]*k] = 0.0;
		  }
		  break;
	       }
	    }
	 }
	 for (j = 0; j < w; ++j) {
	    for (i = 0; i < h; ++i) {
	       if (mask_p[ms[0]*i + ms[1]*j] > 0.5) {
		  for (k = i; k < min(i+iy, h); ++k) {
		     mask_p[ms[0]*k + ms[1]*j] = 0.0;
		  }
		  break;
	       }
	    }
	    for (i = h-1; i >= 0; i -= 1) {
	       if (mask_p[ms[0]*i + ms[1]*j] > 0.5) {
		  for (k = i; k >= max(i-iy+1, 0); k -= 1) {
		     mask_p[ms[0]*k + ms[1]*j] = 0.0;
		  }
		  break;
	       }
	    }
	 }
	 #undef min
	 #undef max
   ]]
   f(mask, ix, iy)
   return mask
end

function nextFrameDepth()
   local im = cam:getNextFrame()
   im = sfm2.undistortImage(im, K, distP)
   local R,T,nFound,nInliers = sfm2.getEgoMotion(last_im, im, Kf, 400)
   print(T/T:norm())
   im_scaled = image.scale(im, geometry.wImg, geometry.hImg)
   --these 2 lines have to stay in this order, or filtered has to be cloned another way
   last_filtered, mask = sfm2.removeEgoMotion(last_filtered, Khalf, R)
   local filtered = filter:forward(im_scaled)

   dbg_last_im = last_im_scaled
   dbg_last_warped, dbg_mask = sfm2.removeEgoMotion(last_im_scaled, Khalf, R)

   local output
   if (nInliers/nFound < 0.2) then
      print("BAD IMAGE !!! " .. nInliers .. " " .. nFound .. " " .. nInliers/nFound)
      mask = torch.Tensor(im[1]:size()):zero()
      output = torch.Tensor(im[1]:size()):zero()
   else
      local input = prepareInput(geometry, last_filtered, filtered)
      local moutput = model:forward(input)
      local poutput = processOutput(geometry, moutput, true)
      output = poutput.full
      enlargeMask(mask,
		  math.ceil((geometry.wImg-poutput.y:size(2))/2),
		  math.ceil((geometry.hImg-poutput.y:size(1))/2))
      output = output[1]
      
      local mask2 = torch.Tensor(geometry.hImg, geometry.wImg):zero()
      mask2:narrow(
	 1, math.floor((geometry.hImg-mask:size(1))/2), mask:size(1)):narrow(
	 2, math.floor((geometry.wImg-mask:size(2))/2), mask:size(2)):copy(mask)
      mask = mask2
      
      mask:cmul(poutput.full_confidences)
      output:cmul(mask)
   end

   last_im = im
   last_im_scaled = im_scaled
   last_filtered = filtered
   --output = torch.FloatTensor():resize(filtered:size(2), filtered:size(3)):copy(filtered[1])
   --output = torch.FloatTensor():resize(im:size(2), im:size(3)):copy(im[1])*100
   output = output:contiguous()
   --im2 = torch.FloatTensor(warped_im[1]:size()):copy(warped_im[1])
   return im_scaled, dbg_last_im, dbg_last_warped, output, mask
end