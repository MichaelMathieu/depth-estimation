require 'torch'
require 'paths'
require 'xlua'
require 'common'
require 'image'
require 'sfm2'
require 'cartesian2polar'
require 'radial_opticalflow_groundtruth'

function load_image(root_directory, calibrationp, i)
   local rd = root_directory
   if rd:sub(-1) ~= '/' then rd = rd..'/' end
   local img
   if paths.filep(string.format("%simages/%09d.jpg", rd, i)) then
      img = image.load(string.format("%simages/%09d.jpg", rd, i))
   elseif paths.filep(string.format("%simages/%09d.png", rd, i)) then
      img = image.load(string.format("%simages/%09d.png", rd, i))
   else
      error(string.format("Image %simages/%09d.* does not exist.", rd, i))
   end
   img = sfm2.undistortImage(img, calibrationp.K, calibrationp.distortion)
   return img
end

function generate_groundtruth(filepath, groundtruthp, im1, im2, mask)
   if groundtruthp.type == 'cross-correlation' then
      print('Computing groundtruth '..filepath)
      local flow = compute_cartesian_groundtruth_cross_correlation(groundtruthp, im1, im2)
      flow[3]:cmul(mask)
      torch.save(filepath, flow)
   else
      error("Can't compute groundtruth of type "..groundtruth.type)
   end
end

function check_flow(groundtruthp, flow)
   if groundtruthp.type == 'cross-correlation' then
      if (flow:nDimension() ~= 3) or (flow:size(1) ~= 3) then
	 error('Flow has wrong size')
      end
      if (flow:size(2) ~= groundtruthp.hGT) or (flow:size(3) ~= groundtruthp.wGT) then
	 error('Flow image has wrong size')
      end
   elseif groundtruthp.type == 'liu' then
      if (flow:nDimension() ~= 3) or (flow:size(1) ~= 3) then
	 error('Flow has wrong size')
      end
      if (flow:size(2) ~= groundtruthp.hGT) or (flow:size(3) ~= groundtruthp.wGT) then
	 error('Flow image has wrong size')
      end
   else
      error("Can't check flow of type "..groundtruthp.type)
   end
end

function load_groundtruth(root_directory, groundtruthp, i, im1, im2, mask)
   local gtdir = root_directory
   if gtdir:sub(-1) ~= '/' then gtdir = gtdir..'/' end
   gtdir = gtdir .. "rectified_flow3/"
   gtdir = gtdir .. groundtruthp.wGT .. 'x' .. groundtruthp.hGT .. '/'
   local ext
   if groundtruthp.type == 'cross-correlation' then
      gtdir = gtdir .. groundtruthp.params.wWin .. 'x' .. groundtruthp.params.hWin .. 'x'
	 .. groundtruthp.params.wKer .. 'x' .. groundtruthp.params.hKer .. '/'
      gtdir = gtdir .. 'max/'
      ext = '.flow'
   elseif groundtruthp.type == 'liu' then
      gtdir = gtdir .. 'celiu/'
      ext = '.png'
   else
      error('Groundtruth '..groundtruthp.type..' not supported.')
   end
   gtdir = gtdir .. groundtruthp.delta .. '/'
   local gtpath = string.format("%s%09d%s", gtdir, i+groundtruthp.delta, ext)
   if not paths.filep(gtpath) then
      generate_groundtruth(gtpath, groundtruthp, im1, im2, mask)
   else
      local flow
      if ext == '.flow' then
	 flow = torch.load(gtpath)
      elseif ext == '.png' then
	 flow = image.load(gtpath)*255-128
	 flow[3]:fill(1)
      else
	 error('Unknown extension ' .. ext)
      end
      check_flow(groundtruthp, flow)
      flow[1]:cmul(flow[3])
      flow[2]:cmul(flow[3])
      return (flow[1]:cmul(flow[1]) + flow[2]:cmul(flow[2])):sqrt()
   end
end

local function rescale(im, w, h, mode)
   mode = mode or 'bilinear'
   if (im:size(2) ~= h) or (im:size(3) ~= w) then
      return image.scale(im, w, h, mode)
   else
      return im
   end
end

function load_training_raw_data(root_directory, networkp, groundtruthp, learningp, calibrationp)
   local Ksmall = calibrationp.K:clone()
   Ksmall[1]:mul(networkp.wImg/calibrationp.wImg)
   Ksmall[2]:mul(networkp.hImg/calibrationp.hImg)
   local data = {}
   --data.images = {}
   --data.prev_images = {}
   --data.prev_images_masks = {}
   --data.groundtruth = {}
   data.polar_images = {}
   data.polar_prev_images = {}
   data.polar_prev_images_masks = {}
   data.polar_groundtruth = {}
   local i = 1
   local previmg = nil
   for iImg = learningp.first_image+1,learningp.first_image+learningp.n_images-1, learningp.delta do
      img = load_image(root_directory, calibrationp, iImg)
      img = rescale(img, calibrationp.wImg, calibrationp.hImg)
      local prev_img
      if previmg == nil then
	 prev_img = load_image(root_directory, calibrationp, iImg-learningp.delta)
      else
	 prev_img = previmg
      end
      prev_img = rescale(prev_img, calibrationp.wImg, calibrationp.hImg)
      local R, T, nFound, nInliers, fundmat = sfm2.getEgoMotion{im1 = img, im2 = prev_img,
								K = calibrationp.K,
								maxPoints = calibrationp.sfm_max_points}
      local e1, e2 = sfm2.getEpipoles(fundmat)
      local rmax = math.max(math.floor(networkp.hImg/2),math.floor(networkp.wImg/2))
      local polarWarpMask = getC2PMask(networkp.wImg, networkp.hImg,
				       networkp.wInput, networkp.hInput,
				       e1[1]*networkp.wImg/calibrationp.wImg,
				       e1[2]*networkp.wImg/calibrationp.wImg,
				       0, 0, rmax)
      local polarWarpMaskPad = getC2PMask(networkp.wImg, networkp.hImg,
				       networkp.wInput, networkp.hInput,
				       e1[1]*networkp.wImg/calibrationp.wImg,
				       e1[2]*networkp.wImg/calibrationp.wImg,
				       math.floor((networkp.wKernel-1)/2),
				       math.ceil((networkp.wKernel-1)/2), rmax)
      if nInliers/nFound < calibrationp.bad_image_threshold then
	 previmg = nil
      else
	 noprev = img
	 img = rescale(img, networkp.wImg, networkp.hImg)
	 prev_img = rescale(prev_img, networkp.wImg, networkp.hImg)
	 local prev_img_mask
	 prev_img, prev_img_mask = sfm2.removeEgoMotion(prev_img, Ksmall, R)
	 local h = prev_img_mask:size(1)
	 local w = prev_img_mask:size(2)
	 prev_img_mask:sub(1,1,1,w):zero()
	 prev_img_mask:sub(h,h,1,w):zero()
	 prev_img_mask:sub(1,h,1,1):zero()
	 prev_img_mask:sub(1,h,w,w):zero()
	 local groundtruth = load_groundtruth(root_directory, groundtruthp, iImg, prev_img, img, prev_img_mask)
	 if i == 1 then
	    image.display{image=groundtruth, min=0, max=12}
	    image.display{image={prev_img, img}}
	 end
	 
	 data.polar_images[i] = cartesian2polar(img, polarWarpMaskPad)
	 data.polar_prev_images[i] = cartesian2polar(prev_img, polarWarpMaskPad)
	 data.polar_prev_images_masks[i] = cartesian2polar(prev_img_mask, polarWarpMaskPad)
	 data.polar_prev_images_masks[i] = torch.Tensor(data.polar_prev_images_masks[i]:size()):copy(data.polar_prev_images_masks[i]:gt(0))
	 data.polar_groundtruth[i] = cartesian2polar(groundtruth, polarWarpMask)
	 data.polar_groundtruth[i] = data.polar_groundtruth[i]*networkp.hInput/rmax
	 --data.polar_groundtruth[i] = (data.polar_groundtruth[i]+0.5):floor()
	 
	 i = i + 1
      end
   end
   return data
end

function generate_training_patches(raw_data, networkp, learningp)
   local patches = {}
   patches.images = raw_data.polar_images
   patches.prev_images = raw_data.polar_prev_images
   patches.patches = {}
   patches.flow = {}
   function patches:getPatch(i)
      return {self.prev_images[self.patches[i][1]]:sub(1, 3,
						       self.patches[i][2],self.patches[i][3]-1,
						       self.patches[i][4],self.patches[i][5]-1),
	      self.images[self.patches[i][1]]:sub(1, 3,
						  self.patches[i][2], self.patches[i][3]-1,
						  self.patches[i][4], self.patches[i][5]-1)}
   end
   function patches:size()
      return #self.patches
   end
   local wPatch = networkp.wKernel
   local hPatch = networkp.hKernel + networkp.hWin - 1
   local i = 1
   while i <= learningp.n_train_set do
      local iImg = randInt(1, #raw_data.polar_images + 1)
      local x = randInt(1, networkp.wInput - wPatch)
      local y = randInt(1, networkp.hInput - hPatch)
      local mask_patch = raw_data.polar_prev_images_masks[iImg]:sub(y, y+hPatch-1,
								    x, x+wPatch-1)
      if (mask_patch-1):sum() == 0 then
	 patches.patches[i] = {iImg, y, y+hPatch, x, x+wPatch}
	 patches.flow[i] = raw_data.polar_groundtruth[iImg][{y+math.ceil(networkp.hKernel/2)-1,
							     x+math.ceil(wPatch/2)-1}]
	 i = i + 1
      end
   end
   return patches
end