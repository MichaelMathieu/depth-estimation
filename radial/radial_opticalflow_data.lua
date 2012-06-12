require 'torch'
require 'paths'
require 'xlua'
require 'common'
require 'image'
require 'sfm2'
require 'cartesian2polar'
require 'radial_opticalflow_groundtruth'
require 'radial_opticalflow_polar'

function load_image(root_directory, dataset, calibrationp, i)
   local rd = root_directory
   local ds = dataset
   if rd:sub(-1) ~= '/' then rd = rd..'/' end
   if ds:sub(-1) ~= '/' then ds = ds..'/' end
   local img
   if paths.filep(string.format("%s%simages/%09d.jpg", rd, ds, i)) then
      img = image.load(string.format("%s%simages/%09d.jpg", rd, ds, i))
   elseif paths.filep(string.format("%s%simages/%09d.png", rd, ds, i)) then
      img = image.load(string.format("%s%simages/%09d.png", rd, ds, i))
   else
      error(string.format("Image %s%simages/%09d.* does not exist.", rd, ds, i))
   end
   img = sfm2.undistortImage(img, calibrationp.K, calibrationp.distortion)
   return img
end

function generate_groundtruth(gtdir, filepath, groundtruthp, im1, im2, mask)
   print('Computing '..groundtruthp.type..' groundtruth '..filepath)
   local flow
   if groundtruthp.type == 'cross-correlation' then
      flow = compute_cartesian_groundtruth_cross_correlation(groundtruthp, im1, im2)
   elseif groundtruthp.type == 'liu' then
      flow = compute_cartesian_groundtruth_liu(groundtruthp, im1, im2)
   else
      error("Can't compute groundtruth of type "..groundtruthp.type)
   end
   flow[3]:cmul(mask)
   os.execute('mkdir -p ' .. gtdir)
   torch.save(filepath, flow)
end

function check_flow(groundtruthp, flow)
   if groundtruthp.type == 'cross-correlation' then
      if (flow:nDimension() ~= 3) or (flow:size(1) < 3) then
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

function load_groundtruth(root_directory, dataset, groundtruthp, maxflow,
			  i, e2, im1, im2, mask)
   local gtdir = root_directory
   local ds = dataset
   if gtdir:sub(-1) ~= '/' then gtdir = gtdir..'/' end
   if ds:sub(-1) ~= '/' then ds = ds..'/' end
   gtdir = gtdir .. ds .. "rectified_flow3/"
   gtdir = gtdir .. groundtruthp.wGT .. 'x' .. groundtruthp.hGT .. '/'
   local ext = '.flow'
   if groundtruthp.type == 'cross-correlation' then
      gtdir = gtdir .. groundtruthp.params.wWin .. 'x' .. groundtruthp.params.hWin .. 'x'
	 .. groundtruthp.params.wKer .. 'x' .. groundtruthp.params.hKer .. '/'
      gtdir = gtdir .. 'max/'
   elseif groundtruthp.type == 'liu' then
      gtdir = gtdir .. 'celiu/'
   else
      error('Groundtruth '..groundtruthp.type..' not supported.')
   end
   gtdir = gtdir .. groundtruthp.delta .. '/'
   local gtpath = string.format("%s%09d%s", gtdir, i+groundtruthp.delta, ext)
   if not paths.filep(gtpath) then
      generate_groundtruth(gtdir, gtpath, groundtruthp, im1, im2, mask)
   end
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
   if groundtruthp.type == 'cross-correlation' then
      flow[3]:cmul(torch.Tensor(flow[3]:size()):copy(flow[4]:gt(15)))
   end
   local radial = torch.Tensor(2, flow:size(2), flow:size(3))
   radial[1]:copy(torch.ger(torch.linspace(0, flow:size(2)-1, flow:size(2)),
			    torch.Tensor(flow:size(3)):fill(1))-e2[2])
   radial[2]:copy(torch.ger(torch.Tensor(flow:size(2)):fill(1),
			    torch.linspace(0, flow:size(3)-1, flow:size(3)))-e2[1])
   local radialnorm = flownorm(radial)
   radial[1]:cdiv(radialnorm)
   radial[2]:cdiv(radialnorm)
   local projf = flow[1]:cmul(radial[1])+flow[2]:cmul(radial[2])
   local gds = projf:ge(0)
   gds = torch.Tensor(gds:size()):copy(gds)
   gds:cmul(flow[3])
   --gds = (gds+0.5):floor()
   projf:cmul(gds)
   local proj = (projf+0.5):floor()
   winp = image.display{image={proj, projf}, win=winp}
   
   local saturation = torch.Tensor(proj:size()):copy(proj:le(maxflow))
   proj:cmul(saturation)
   proj:add((-saturation+1)*(maxflow))
   gds:cmul(saturation)

   return proj:cmul(gds), gds
end

local function rescale(im, w, h, mode)
   mode = mode or 'bilinear'
   if (im:size(2) ~= h) or (im:size(3) ~= w) then
      return image.scale(im, w, h, mode)
   else
      return im
   end
end

local function getFstLastImages(root_directory, dataset)
   local rd = root_directory
   local ds = dataset
   if rd:sub(-1) ~= '/' then rd = rd..'/' end
   if ds:sub(-1) ~= '/' then ds = ds..'/' end
   local dir = rd .. ds .. 'images/'
   local imgs = ls2(dir, function(a) return tonumber(a:sub(1,-5)) ~= nil end)
   local fst = tonumber(imgs[1]:sub(1,-5))
   local last = tonumber(imgs[1]:sub(1,-5))
   for i = 1,#imgs do
      local a = tonumber(imgs[i]:sub(1,-5))
      if a < fst then
	 fst = a
      end
      if a > last then
	 last = a
      end
   end
   return fst, last
end

function load_dataset(root_directory, dataset, networkp, groundtruthp, learningp, calibrationp)
   local Ksmall = calibrationp.K:clone()
   Ksmall[1]:mul(networkp.wImg/calibrationp.wImg)
   Ksmall[2]:mul(networkp.hImg/calibrationp.hImg)
   local data = {}
   data.images = {}
   data.prev_images = {}
   --data.prev_images_masks = {}
   data.polar_images = {}
   data.polar_prev_images = {}
   data.polar_prev_images_masks = {}
   data.e1 = {}
   data.e2 = {}
   if groundtruthp ~= nil then
      data.groundtruth = {}
      data.polar_groundtruth = {}
      data.polar_groundtruth_masks = {}
   end

   local fst_img, last_img = getFstLastImages(root_directory, dataset)
   
   local i = 1
   local previmg = nil
   print('Loading images...')
   for iImg = fst_img+1,last_img,learningp.delta do
      xlua.progress(iImg-fst_img-1,last_img-fst_img)
      img = load_image(root_directory, dataset, calibrationp, iImg)
      img = rescale(img, calibrationp.wImg, calibrationp.hImg)
      local prev_img
      if previmg == nil then
	 prev_img = load_image(root_directory, dataset, calibrationp, iImg-learningp.delta)
      else
	 prev_img = previmg
      end
      prev_img = rescale(prev_img, calibrationp.wImg, calibrationp.hImg)
      local R, T, nFound, nInliers, fundmat =
	 sfm2.getEgoMotion{im1 = prev_img, im2 = img,
			   K = calibrationp.K,
			   maxPoints = calibrationp.sfm.max_points,
			   pointsQuality=calibrationp.sfm.points_quality,
			   ransacMaxDist=calibrationp.sfm.ransac_max_dist}
      local _, e2 = sfm2.getEpipoles(fundmat)
      e2 = e2*networkp.wImg/calibrationp.wImg
      data.e2[i] = e2
      local rmax = getRMax(networkp, e2)
      local polarWarpMask = getC2PMask(networkp.wImg, networkp.hImg,
				       networkp.wInput, networkp.hInput,
				       e2[1], e2[2], 0, 0, rmax)
      local polarWarpMaskPad = getC2PMask(networkp.wImg, networkp.hImg,
					  networkp.wInput, networkp.hInput,
					  e2[1], e2[2],
					  math.floor((networkp.wKernel-1)/2),
					  math.ceil((networkp.wKernel-1)/2), rmax)
      if nInliers/nFound < calibrationp.bad_image_threshold then
	 previmg = nil
      else
	 noprev = img
	 img = rescale(img, networkp.wImg, networkp.hImg)
	 prev_img = rescale(prev_img, networkp.wImg, networkp.hImg)
	 local prev_img_mask
	 prev_img, prev_img_mask = sfm2.removeEgoMotion(prev_img, Ksmall, R, 'bilinear')
	 local h = prev_img_mask:size(1)
	 local w = prev_img_mask:size(2)
	 prev_img_mask:sub(1,1,1,w):zero()
	 prev_img_mask:sub(h,h,1,w):zero()
	 prev_img_mask:sub(1,h,1,1):zero()
	 prev_img_mask:sub(1,h,w,w):zero()
	 data.polar_images[i] = cartesian2polar(img, polarWarpMaskPad)
	 data.polar_prev_images[i] = cartesian2polar(prev_img, polarWarpMaskPad)
	 data.polar_prev_images_masks[i] = cartesian2polar(prev_img_mask, polarWarpMaskPad)
	 data.polar_prev_images_masks[i] = torch.Tensor(data.polar_prev_images_masks[i]:size()):copy(data.polar_prev_images_masks[i]:gt(0))
	 data.prev_images[i] = prev_img
	 data.images[i] = img
	 
	 if groundtruthp ~= nil then
	    local groundtruth, gt_gds = load_groundtruth(root_directory, dataset, groundtruthp,
							 (networkp.hWin-1)*rmax/networkp.hInput,
							 iImg, e2, prev_img, img, prev_img_mask)
	 
	    data.groundtruth[i] = groundtruth
	    data.polar_groundtruth[i] = cartesian2polar(groundtruth, polarWarpMask)
	    data.polar_groundtruth_masks[i] = cartesian2polar(gt_gds, polarWarpMask)
	    data.polar_groundtruth[i] = data.polar_groundtruth[i]*networkp.hInput/rmax	    
	    --data.polar_groundtruth[i] = (data.polar_groundtruth[i]+0.5):floor()
	 end
	 
	 i = i + 1
      end
      collectgarbage()
   end
   return data
end

function load_data(root_directory, networkp, groundtruthp, learningp, calibrationp)
   local datasets = ls2(root_directory, function(a) return true end)
   local data = {}
   for i = 1,#datasets do
      local dataset = load_dataset(root_directory, datasets[i], networkp,
				   groundtruthp, learningp, calibrationp)
      for k, v in pairs(dataset) do
	 if data[k] == nil then data[k] = {} end
	 for j = 1,#v do
	    table.insert(data[k], dataset[k][j])
	 end
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
      local gt_mask_patch = raw_data.polar_groundtruth_masks[iImg]:sub(y, y+hPatch-1,
								       x, x+wPatch-1)
      mask_patch:cmul(gt_mask_patch)

      if mask_patch:lt(0.1):sum() == 0 then
	 patches.patches[i] = {iImg, y, y+hPatch, x, x+wPatch}
	 patches.flow[i] = raw_data.polar_groundtruth[iImg][{y+math.ceil(networkp.hKernel/2)-1,
							     x}]
	 i = i + 1
      end
   end
   return patches
end