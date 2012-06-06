require 'torch'
require 'paths'
require 'xlua'
require 'common'
require 'image'
require 'sfm2'
require 'cartesian2polar'

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

function load_groundtruth(root_directory, groundtruthp, i)
   local gtdir = root_directory
   if gtdir:sub(-1) ~= '/' then gtdir = gtdir..'/' end
   gtdir = gtdir .. "rectified_flow2/"
   gtdir = gtdir .. groundtruthp.wGT .. 'x' .. groundtruthp.hGT .. '/'
   local ext
   if groundtruthp.type == 'cross-correlation' then
      gtdir = gtdir .. groundtruthp.params.wWin .. 'x' .. groundtruthp.params.hWin .. 'x'
	 .. groundtruthp.params.wKer .. 'x' .. groundtruthp.params.hKer .. '/'
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
      error("Groundtruth file "..gtpath.." not found.")
   else
      if ext == '.flow' then
	 return torch.load(gtpath)
      elseif ext == '.png' then
	 local flow = image.load(gtpath)[{{1,2},{},{}}]
	 return flow*255-128
      else
	 error('Unknown extension ' .. ext)
      end
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
   Ksmall[1]:mul(networkp.wInput/calibrationp.wImg)
   Ksmall[2]:mul(networkp.hInput/calibrationp.hImg)
   local training_raw_data = {}
   training_raw_data.images = {}
   training_raw_data.prev_images = {}
   training_raw_data.prev_images_masks = {}
   training_raw_data.polar_images = {}
   training_raw_data.polar_prev_images = {}
   training_raw_data.polar_prev_images_masks = {}
   training_raw_data.groundtruth = {}
   local i = 1
   local noprev = true
   for iImg = learningp.first_image+1,learningp.n_images do
      img = load_image(root_directory, calibrationp, iImg)
      img = rescale(img, calibrationp.wImg, calibrationp.hImg)
      local prev_img
      if noprev then
	 prev_img = load_image(root_directory, calibrationp, iImg-1)
      else
	 prev_img = training_raw_data.images[i-1]
      end
      prev_img = rescale(prev_img, calibrationp.wImg, calibrationp.hImg)
      local R, T, nFound, nInliers = sfm2.getEgoMotion{im1 = img, im2 = prev_img,
						       K = calibrationp.K,
						       maxPoints = calibrationp.sfm_max_points}
      if nInliers/nFound < calibrationp.bad_image_threshold then
	 noprev = true
      else
	 noprev = false
	 training_raw_data.images[i] = rescale(img, networkp.wInput, networkp.hInput)
	 prev_img = rescale(prev_img, networkp.wInput, networkp.hInput)
	 training_raw_data.warped_prev_images[i], training_raw_data.warped_prev_images_masks[i] = sfm2.removeEgoMotion(prev_img, Ksmall, R)
	 

	 training_raw_data.groundtruth[i] = load_groundtruth(root_directory,
							     groundtruthp, iImg)
	 i = i + 1
      end
   end
   return training_raw_data
end

function generate_training_patches(raw_data, networkp, leanringp)
   local patches = {}
   local patches.patches = {}
   local patches.flow = {}
   local xmin = 1
   local xmax = networkp.wInput
   local ymin = 1
   local ymax = networkp.hInput
   local i = 1
   while i <= learningp.n_train_set do
      local iImg = randInt(1, #raw_data.images + 1)
      local x = randInt(xmin,xmax