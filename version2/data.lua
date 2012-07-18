require 'torch'
require 'paths'
require 'xlua'
require 'common'
require 'image'
require 'sfm2'
require 'opencv24'
require 'groundtruth'

function new_dataset(path, calibrationp, datap, groundtruthp)
   if path:sub(-1) ~= '/' then path = path .. '/' end
   local dataset = {}
   dataset.path = path
   dataset.calibrationp = calibrationp
   dataset.calibrationp.Ksmall = dataset.calibrationp.K:clone()/4
   dataset.calibrationp.Ksmall[3][3] = 1
   dataset.datap = datap
   dataset.groundtruthp = groundtruthp

   dataset.image_names_idx = {}
   dataset.image_idx_names = {}
   dataset.names = {}
   function dataset:add_subdir(dirname)
      if dirname:sub(-1) ~= '/' then dirname = dirname .. '/' end
      local names = ls2(dataset.path .. dirname .. 'images/',
			function(a) return tonumber(a:sub(1,-5)) ~= nil end)
      for i = 1,#names do
	 names[i] = dirname..'images/'..names[i]
	 dataset.image_names_idx[names[i]] = i
	 dataset.image_idx_names[i] = names[i]      
      end
      for i = 2,#names do
	 table.insert(dataset.names, names[i])
      end
   end
   
   function dataset:get_idx_from_name(name)
      return self.image_names_idx[name]
   end
   function dataset:get_name_by_idx(idx)
      return self.image_idx_names[idx]
   end
   function dataset:get_image_names()
      return self.names
   end
   function dataset:size()
      return #self.names
   end

   function dataset:get_full_image_by_name(name)
      local img
      if paths.filep(string.format("%s%s", self.path, name)) then
	 img = image.load(string.format("%s%s", self.path, name))
      else
	 error(string.format("Image %s%s does not exist.", self.path, name))
      end
      if (img:size(2) ~= self.calibrationp.hImg) or (img:size(3) ~= self.calibrationp.wImg) then
	 img = image.scale(img, self.calibrationp.wImg, self.calibrationp.hImg)
      end
      if calibrationp.distortion:abs():sum() ~= 0 then
	 img = sfm2.undistortImage(img, self.calibrationp.K, self.calibrationp.distortion)
      end
      return img
   end
   function dataset:get_full_image_by_idx(idx)
      return self:get_full_image_by_name(self:get_name_by_idx(idx))
   end
   
   dataset.images = {}
   function dataset:get_image_by_name(name)
      if not dataset.images[name] then
	 local img = self:get_full_image_by_name(name)
	 img = image.scale(img, self.datap.wImg, self.datap.hImg)
	 dataset.images[name] = img
      end
      return dataset.images[name]
   end
   function dataset:get_image_by_idx(idx)
      return self:get_image_by_name(self:get_name_by_idx(idx))
   end

   dataset.prev_images = {}
   dataset.masks = {}
   function dataset:get_prev_image_by_name(name)
      if not self.prev_images[name] then
	 local img1 = self:get_full_image_by_idx(self:get_idx_from_name(name)-1, true)
	 local img2 = self:get_full_image_by_name(name, true)
	 local mask = torch.Tensor(img1:size(2), img1:size(3)):fill(1)
	 if self.calibrationp.rectify then
	    local sfmparams = self.calibrationp.sfm
	    sfmparams.im1 = img1
	    sfmparams.im2 = img2
	    sfmparams.K = self.calibrationp.K
	    sfmparams.trackedPoints = opencv24.TrackPointsLK(sfmparams)
	    local R, T, nFound, nInliers, fundmat = sfm2.getEgoMotion(sfmparams)
	    img1 = image.scale(img1, self.datap.wImg, self.datap.hImg, 'bilinear')
	    img1, mask = sfm2.removeEgoMotion(img1, self.calibrationp.Ksmall, R, 'bilinear')
	 end
	 self.masks[name] = mask
	 self.prev_images[name] = img1
      end
      return self.prev_images[name]
   end
   function dataset:get_prev_image_by_idx(idx)
      return self:get_prev_image_by_name(self:get_name_by_idx(idx))
   end

   function dataset:get_mask_by_name(name)
      if not self.masks[name] then
	 self:get_prev_image_by_name(name)
      end
      return self.masks[name]
   end
   function dataset:get_mask_by_idx(idx)
      return self:get_mask_by_name(self:get_name_by_idx(idx))
   end

   dataset.gt = {}
   function dataset:get_gt_by_name(name)
      if not dataset.gt[name] then
	 local gtdir = self.path
	 if self.calibrationp.rectify then
	    gtdir = gtdir .. "rectified_flow4/"
	 else
	    gtdir = gtdir .. "flow"
	 end
	 gtdir = gtdir .. self.datap.wImg .. 'x' .. self.datap.hImg .. '/'
	 if self.groundtruthp.type == 'cross-correlation' then
	    gtdir = gtdir .. self.groundtruthp.params.wWin .. 'x'
	    gtdir = gtdir .. self.groundtruthp.params.hWin .. 'x'
	    gtdir = gtdir .. self.groundtruthp.params.wKernel .. 'x'
	    gtdir = gtdir .. self.groundtruthp.params.hKernel .. '/'
	    gtdir = gtdir .. 'max/'
	 elseif self.groundtruthp.type == 'liu' then
	    gtdir = gtdir .. 'celiu/'
	 else
	    error('Groundtruth '..self.groundtruthp.type..' not supported.')
	 end
	 local name2 = name
	 if (name2:sub(-4) == '.jpg') or (name2:sub(-4) == '.png') then
	    name2 = name2:sub(1,-5)
	 end
	 local gtpath = string.format("%s%s.flow", gtdir, name2)
	 
	 if not paths.filep(gtpath) then
	    local im1 = self:get_prev_image_by_name(name)
	    local im2 = self:get_image_by_name(name)
	    local mask = self:get_mask_by_name(name)
	    local flow, conf = generate_groundtruth(self.groundtruthp, im1, im2, mask)
	    torch.save(gtpath, {flow, conf})
	 end
	 local flowraw = torch.load(gtpath)
	 if type(flowraw) ~= 'table' then
	    flowraw = {flowraw[{{1,2}}], flowraw[3]}
	    torch.save(gtpath, flowraw)
	 end
	 local flow = flowraw[1]
	 local conf = flowraw[2]
	 return flow, conf
      end
      return dataset.gt[name]
   end
   function dataset:get_gt_by_idx(idx)
      return self:get_gt_by_name(self:get_name_by_idx(idx))
   end

   function dataset:get_patches(nSamples)
      local patches = {}
      local wPatch = datap.wKernel + datap.wWin - 1
      local hPatch = datap.hKernel + datap.hWin - 1
      local wOffset = math.ceil(datap.wKernel/2) + math.ceil(datap.wWin/2) - 2
      local hOffset = math.ceil(datap.hKernel/2) + math.ceil(datap.hWin/2) - 2
      local i = 1
      local names = self:get_image_names()
      while i <= nSamples do
	 xlua.progress(i, nSamples)
	 local iImg = randInt(1, #names+1)
	 local name = names[iImg]
	 local x = randInt(1, datap.wImg - wPatch)
	 local y = randInt(1, datap.hImg - hPatch)
	 local mask_val = self:get_mask_by_name(name)[{{y,y+hOffset-1},{x,x+wOffset-1}}]
	 mask_val:add(-1)
	 local flow, conf = self:get_gt_by_name(name)
	 flow = flow[{{},y+hOffset, x+wOffset}]:clone()
	 conf = conf[{y+hOffset, x+wOffset}]
	 if (mask_val:abs():sum() < 0.1) and (conf > 0.5) then
	    patches[i] = {
	       patch1 = function()
			   return self:get_prev_image_by_name(name):sub(1,3,
									y,y+hPatch-1,
									x,x+wPatch-1)
			end,
	       patch2 = function()
			   return self:get_image_by_name(name):sub(1,3,
								   y,y+hPatch-1,
								   x,x+wPatch-1)
			end,
	       target = flow,
	       targetCrit = (flow[1]+math.ceil(datap.hWin/2)-1)*datap.wWin + flow[2]+math.ceil(datap.wWin/2)-1 + 1
	    }
	    i = i + 1
	 end
	 if i % 10 == 0 then
	    collectgarbage()
	 end
      end
      return patches
   end

   return dataset
end

function generate_groundtruth(groundtruthp, im1, im2, mask)
   --print('Computing '..groundtruthp.type..' groundtruth '..filepath)
   local flow, conf
   if groundtruthp.type == 'cross-correlation' then
      flow, conf = compute_groundtruth_cross_correlation(groundtruthp, im1, im2, mask)
   elseif groundtruthp.type == 'liu' then
      flow, conf = compute_groundtruth_liu(groundtruthp, im1, im2)
   else
      error("Can't compute groundtruth of type "..groundtruthp.type)
   end
   return flow, conf
end