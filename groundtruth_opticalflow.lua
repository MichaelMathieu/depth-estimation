require 'torch'
require 'paths'
require 'xlua'
require 'common'
require 'image'
require 'common'
require 'opencv'
require 'opticalflow_model'
require 'motion_correction'
require 'sfm2'

function findMax(geometry, of)
   local maxs, imax = of:reshape(of:size(1)*of:size(2), of:size(3),of:size(4)):max(1)
   local yc, xc = centered2onebased(geometry, 0, 0)
   local xmax = torch.Tensor(imax:size(2), imax:size(3)):fill(yc)
   local ymax = torch.Tensor(imax:size(2), imax:size(3)):fill(xc)
   for i = 1,imax:size(2) do
      for j = 1,imax:size(3) do
	 if maxs[1][i][j] ~= -1 then
	    local y, x = x2yx(geometry, imax[1][i][j])
	    ymax[i][j] = y
	    xmax[i][j] = x
	 end
      end
   end
   return ymax,xmax
end

function getOpticalFlowFast(geometry, image1, image2)
   local geometryGT = {}
   geometryGT.wPatch2 = geometry.maxwGT + geometry.wKernelGT - 1
   geometryGT.hPatch2 = geometry.maxhGT + geometry.hKernelGT - 1
   geometryGT.hImg = geometry.hImg
   geometryGT.wImg = geometry.wImg
   geometryGT.maxw = geometry.maxwGT
   geometryGT.maxh = geometry.maxhGT
   geometryGT.maxwGT = geometry.maxwGT
   geometryGT.maxhGT = geometry.maxhGT
   geometryGT.hKernel = geometry.hKernelGT
   geometryGT.wKernel = geometry.wKernelGT
   geometryGT.layers = geometry.layers
   geometryGT.multiscale = false
   geometryGT.training_mode = true
   
   local maxh = geometry.maxhGT
   local maxw = geometry.maxwGT
   local nfeats = geometry.hKernelGT*geometry.wKernelGT*image1:size(1)
   local input = prepareInput(geometryGT,
			      image.scale(image1, geometry.wImg, geometry.hImg),
			      image.scale(image2, geometry.wImg, geometry.hImg))

   local input1 = input[1]:unfold(2, geometry.hKernelGT, 1):unfold(3, geometry.wKernelGT, 1)
   local h1 = input1:size(2)
   local w1 = input1:size(3)
   local input1b = torch.Tensor(nfeats, h1, w1)
   for i = 1,h1 do
      for j = 1,w1 do
	 input1b:select(2,i):select(2,j):copy(input1:select(2,i):select(2,j):reshape(nfeats))
      end
   end
   
   local input2 = input[2]:unfold(2, geometry.hKernelGT, 1):unfold(3, geometry.wKernelGT, 1)
   local h2 = input2:size(2)
   local w2 = input2:size(3)
   local input2b = torch.Tensor(nfeats, h2, w2)
   for i = 1,h2 do
      for j = 1,w2 do
	 input2b:select(2,i):select(2,j):copy(input2:select(2,i):select(2,j):reshape(nfeats))
      end
   end
   
   local net = nn.SpatialMatching(maxh, maxw, false)
   local output = net:forward({input1b, input2b})
   output = -output
   print(output:size())
   output = output:reshape(output:size(1), output:size(2), maxh*maxw)
   local output2 = processOutput(geometryGT, output, true)
   return output2.full[1], output2.full[2]
end

function getOpticalFlow(geometry, image1, image2)
   local halfmaxh = math.ceil(geometry.maxhGT/2)-1
   local halfmaxw = math.ceil(geometry.maxwGT/2)-1
   local halfhKernel = math.ceil(geometry.hKernelGT/2)-1
   local halfwKernel = math.ceil(geometry.wKernelGT/2)-1
   of = torch.Tensor(geometry.maxhGT, geometry.maxwGT, image1:size(2), image1:size(3)):fill(-1)
   for i = 1, image1:size(2) - geometry.hKernelGT - geometry.maxhGT + 2 do
      xlua.progress(i, image1:size(2) - geometry.hKernelGT - geometry.maxhGT + 2)
      for j = 1, image1:size(3) - geometry.wKernelGT - geometry.maxwGT + 2 do
	 local win = image1:sub(1, image1:size(1),
				i+halfmaxh, i+halfmaxh+geometry.hKernelGT-1,
				j+halfmaxw, j+halfmaxw+geometry.wKernelGT-1)
	 local tomul = image2:sub(1, image2:size(1),
				  i, i+geometry.maxhGT+geometry.hKernelGT-2,
				  j, j+geometry.maxwGT+geometry.wKernelGT-2)
	 local unfolded = tomul:unfold(2, geometry.hKernelGT, 1):unfold(3, geometry.wKernelGT,1)
	 local norm2win = win:dot(win)
	 for k = 1, geometry.maxhGT do
	    for l = 1, geometry.maxwGT do
	       local win2 = unfolded:select(2,k):select(2,l)
	       of[k][l][i+halfmaxh+halfhKernel][j+halfmaxw+halfwKernel] = win:dot(win2)/(math.sqrt(norm2win*win2:dot(win2)))
	    end
	 end
      end
   end
   --'of' contains now the expected result of the newtork
   assert(false)-- output not coherent anymore (it is one-based)
   return findMax(geometry, of)
end

function loadImageOpticalFlow(geometry, dirbasename, imagebasename, previmagebasename,
			      delta, groundtruth)
   local ext = '.jpg'
   local imagepath = dirbasename .. 'images/' .. imagebasename .. ext
   if not paths.filep(imagepath) then
      ext = '.png'
      imagepath = dirbasename .. 'images/' .. imagebasename .. ext
      if not paths.filep(imagepath) then
	 print("Image " .. imagepath .. " not found.")
	 return nil
      end
   end
   local im = image.scale(image.load(imagepath), geometry.wImg, geometry.hImg)
   if not previmagebasename then
      return im
   end
   local flowdir = dirbasename .. 'flow/' .. geometry.wImg .. 'x' .. geometry.hImg
   local flowfilename
   local flow = nil
   
   if groundtruth == 'liu' then
      flowdir = flowdir .. '/celiu'
      os.execute('mkdir -p ' .. flowdir)
      flowfilename = flowdir .. '/' .. imagebasename .. '.png'
      if paths.filep(flowfilename) then
         flowpng = image.loadPNG(flowfilename)*255-128
         if (flowpng:size(2) ~= geometry.hImg) or (flowpng:size(3) ~= geometry.wImg) then
       flow = nil
       print("Flow in file " .. flowfilename .. " has wrong size. Recomputing...")
         end
         flow = torch.Tensor(2, flowpng:size(2), flowpng:size(3)):fill(1)
         flow[1] = flowpng[1]
         flow[2] = flowpng[2]
      else
         print("Flow " .. flowfilename .. " not found.")
         return nil
      end
      
   elseif groundtruth == 'cross-correlation' then
      flowdir = flowdir .. '/' .. geometry.maxhGT .. 'x' ..geometry.maxwGT .. 'x'
      flowdir = flowdir .. geometry.hKernelGT .. 'x' ..geometry.wKernelGT .. '/' .. delta
      os.execute('mkdir -p ' .. flowdir)
      flowfilename = flowdir .. '/' .. imagebasename .. '.flow'
      if paths.filep(flowfilename) then
         flow = torch.load(flowfilename)
         if (flow:size(2) ~= geometry.hImg) or (flow:size(3) ~= geometry.wImg) then
	    flow = nil
	    print("Flow in file " .. flowfilename .. " has wrong size. Recomputing...")
         end
      end
      if not flow then
	 local previmagepath = dirbasename .. 'images/' .. previmagebasename .. ext
	 print('Computing groundtruth optical flow for images '..imagepath..' and '..previmagepath)
         if not paths.filep(previmagepath) then
	    print("Image " .. previmagepath .. " not found.")
	    return nil
	 end
	 local previmage = image.scale(image.load(previmagepath), geometry.wImg, geometry.hImg)
	 local yflow, xflow = getOpticalFlowFast(geometry, previmage, im)
	 flow = torch.Tensor(2, xflow:size(1), xflow:size(2)):fill(1)
	 flow[1]:copy(yflow)
	 flow[2]:copy(xflow)
	 torch.save(flowfilename, flow)
      end
   else
      error('groundtruth must be either liu or cross-correlation')
   end
   

   return im, flow
end

function loadRectifiedImageOpticalFlow(geometry, dirbasename, imagebasename,
				       previmagebasename, delta, groundtruth)
   if groundtruth == 'liu' then error('liu rectified : not implemented') end
   local imagepath = dirbasename .. 'images/' .. imagebasename .. '.jpg'
   if not paths.filep(imagepath) then
      print("Image " .. imagepath .. " not found.")
      return nil
   end
   local im = image.scale(image.load(imagepath), geometry.wImg, geometry.hImg)
   if not previmagebasename then
      return im
   end

   local rectimagepath = dirbasename .. 'rectified_images/' .. imagebasename .. '.jpg'
   if not paths.filep(imagepath) then
      print("Image " .. rectimagepath .. " not found.")
      return nil
   end
   local im_rect = image.scale(image.load(imagepath), geometry.wImg, geometry.hImg)

   local flowdir = dirbasename .. 'rectified_flow/' .. geometry.wImg .. 'x' .. geometry.hImg
   flowdir = flowdir .. '/' .. geometry.maxhGT .. 'x' ..geometry.maxwGT .. 'x'
   flowdir = flowdir .. geometry.hKernelGT .. 'x' ..geometry.wKernelGT .. '/' .. delta
   os.execute('mkdir -p ' .. flowdir)
   local flowfilename = flowdir .. '/' .. imagebasename .. '.flow'
   local flow = nil
   if paths.filep(flowfilename) then
      flow = torch.load(flowfilename)
      if (flow:size(2) ~= geometry.hImg) or (flow:size(3) ~= geometry.wImg) then
         flow = nil
         print("Flow in file " .. flowfilename .. " has wrong size. Recomputing...")
      end
   end

   --TODO there is an error here: the corrected image should be the FIRST one
   error("cf code")
   if not flow then
      local previmagepath = dirbasename .. 'images/' .. previmagebasename .. '.jpg'
      print('Computing groundtruth optical flow for images '..imagepath..' and '..previmagepath)
      if not paths.filep(previmagepath) then
	 print("Image " .. previmagepath .. " not found.")
	 return nil
      end
      local previmage = image.scale(image.load(previmagepath), geometry.wImg, geometry.hImg)
      local yflow, xflow = getOpticalFlowFast(geometry, previmage, im_rect)
      flow = torch.Tensor(2, xflow:size(1), xflow:size(2)):fill(1)
      flow[1]:copy(yflow)
      flow[2]:copy(xflow)
      torch.save(flowfilename, flow)
   end

   return im, flow, im_rect

end

function loadRectifiedImageOpticalFlow2(correction, geometry, learning, dirbasename,
					imagebasename, previmagebasename, delta)
   if learning.groundtruth ~= 'cross-correlation' then
      error('loadRectifiedImageOpticalFlow2: groundtruth must be cross-correlation')
   end
   local ext = '.jpg'
   local impath = dirbasename .. 'images/' .. imagebasename .. ext
   if not paths.filep(impath) then
      ext = '.png'
      impath = dirbasename .. 'images/' .. imagebasename .. ext
      if not paths.filep(impath) then
	 print("Image " .. impath .. " not found.")
	 return nil
      end
   end

   local im = image.scale(image.load(impath), correction.wImg, correction.hImg)
   im = sfm2.undistortImage(im, correction.K, correction.distP)
   if not previmagebasename then
      return image.scale(im, geoemtry.wImg, geometry.hImg)
   end

   local previmpath = dirbasename .. 'images/' .. previmagebasename .. ext
   if not paths.filep(previmpath) then
      print("Image " .. previmpath .. " not found.")
      return nil
   end
   local prev_im = image.scale(image.load(previmpath), correction.wImg, correction.hImg)
   prev_im = sfm2.undistortImage(prev_im, correction.K, correction.distP)
   
   local R, T = sfm2.getEgoMotion{im1=prev_im, im2=im, K=correction.K, maxPoints=500}
   local warped_im, warped_mask = sfm2.removeEgoMotion(prev_im, correction.K, R)

   im = image.scale(im, geometry.wImg, geometry.hImg)
   prev_im = image.scale(prev_im, geometry.wImg, geometry.hImg)
   warped_im = image.scale(warped_im, geometry.wImg, geometry.hImg)
   
   local flowdir = dirbasename .. 'rectified_flow2/' .. geometry.wImg .. 'x' .. geometry.hImg
   flowdir = flowdir .. '/' .. geometry.maxhGT .. 'x' .. geometry.maxwGT .. 'x'
   flowdir = flowdir .. geometry.hKernelGT .. 'x' .. geometry.wKernelGT .. '/' .. delta
   sys.execute('mkdir -p ' .. flowdir)
   local flowfilename = flowdir .. '/' .. imagebasename .. '.flow'
   local flow = nil
   if paths.filep(flowfilename) then
      flow = torch.load(flowfilename)
      if (flow:size(2) ~= geometry.hImg) or (flow:size(3) ~= geometry.wImg) then
         flow = nil
         print("Flow in file " .. flowfilename .. " has wrong size. Recomputing...")
      end
   end

   if not flow then
      print('Computing groundtruth optical flow for images '..impath..' and '..previmpath)
      local yflow, xflow = getOpticalFlowFast(geometry, warped_im, im)
      flow = torch.Tensor(2, xflow:size(1), xflow:size(2)):fill(1)
      flow[1]:copy(yflow)
      flow[2]:copy(xflow)
      torch.save(flowfilename, flow)
      print("ok")
   end

   return last_im, warped_im, warped_mask, im, flow
end

function loadDataOpticalFlowCCLiu(correction, geometry, learning, dirbasename)
   local imagesdir = dirbasename .. 'images'
   raw_data = {}
   raw_data.images = {}
   raw_data.flow = {}

   local imagepaths_raw = {}
   local flowpaths = {}
   local ls = ls2(imagesdir, function(a) return a:sub(-4) == '.jpg' or a:sub(-4) == '.png' end)
   for i = 1,#ls do
      local linebase,_ = ls[i]:sub(1,-5)
      table.insert(imagepaths_raw, linebase)
   end
   local imagepaths = {}
   local iLine = learning.first_image+1 --images are numbered from 0
   for i = 1,learning.num_images do
      imagepaths[i] = imagepaths_raw[iLine]
      iLine = iLine + learning.delta
   end

   if correction.motion_correction == 'mc' then
      raw_data.rectified_images = {}
      raw_data.H = {}

      file = torch.DiskFile(dirbasename .. 'rectified_data_H', 'r')
      raw_data.H = file:readObject()
   elseif correction.motion_correction == 'sfm' then
      raw_data.warped_images = {}
      raw_data.warped_masks = {}
   end

   local im = loadImageOpticalFlow(geometry, dirbasename, imagepaths[1], nil, nil)
   table.insert(raw_data.images, im)
   
   if learning.groundtruth == 'liu' then
      print("Using Liu groundtruth...")
   end

   for i = 2,math.min(#imagepaths, learning.num_images) do
      if correction.motion_correction == 'sfm' then
	 local last_im, warped_im, warped_mask, im, flow = loadRectifiedImageOpticalFlow2(
	    correction, geometry, learning, dirbasename, imagepaths[i],
	    imagepaths[i-1], learning.delta)
         raw_data.images       [i]   = im
         raw_data.flow         [i-1] = flow
         raw_data.warped_images[i-1] = warped_im
         raw_data.warped_masks [i-1] = warped_mask
      elseif correction.motion_correction then
         local im, flow, im_rect = loadRectifiedImageOpticalFlow(
	    geometry, dirbasename, imagepaths[i], imagepaths[i-1],
	    learning.delta, learning.groundtruth)
         table.insert(raw_data.images, im)
         table.insert(raw_data.flow, flow)   
         table.insert(raw_data.rectified_images, im_rect)
      else
         local im, flow = loadImageOpticalFlow(
	    geometry, dirbasename, imagepaths[i],
	    imagepaths[i-1], learning.delta, learning.groundtruth)
         table.insert(raw_data.images, im)
         table.insert(raw_data.flow, flow)
      end
      
   end
   return raw_data
end

function loadDataOpticalFlowCVlibs(geometry, learning, dirbasename)
   if not cvlibs_dataset then
      require 'cvlibs_dataset'
   end
   local ret = {}
   for i = learning.first_image,(learning.first_image+learning.num_images),learning.delta do
      local flowobj = cvlibs_dataset.readFlowObject(dirbasename, i)
      table.insert(ret, flowobj)
   end
   return ret
end

function loadDataOpticalFlow(correction, geometry, learning, dirbasename)
   if (learning.groundtruth == 'liu') or (learning.groundtruth == 'cross-correlation') then
      return loadDataOpticalFlowCCLiu(correction, geometry, learning, dirbasename)
   elseif learning.groundtruth == 'cvlibs' then
      return loadDataOpticalFlowCVlibs(geometry, learning, dirbasename)
   else
      error('loadDataOpticalFlow: learning.groundtruth must be either liu, cvlibs or cross-correlation')
   end
end

function check_borders(index, xPatch, yPatch, geometry)
   local im_index = index-1
   
   local wpt = torch.Tensor(2)
   local chpt = torch.Tensor(2)
   local invH = torch.inverse(raw_data.H[im_index]:sub(1,2,1,2))

   local w_imgs = geometry.wImg
   local h_imgs = geometry.hImg
   local wPatch = geometry.wPatch2
   local hPatch = geometry.hPatch2

   local x = xPatch + wPatch/2
   local y = yPatch + hPatch/2

   for i=0,1 do
      for j=0,1 do
         wpt[1] = x - w_imgs/2 + wPatch*(i-0.5) - raw_data.H[im_index][1][3]
         wpt[2] = y - h_imgs/2 + hPatch*(j-0.5) - raw_data.H[im_index][2][3]
         
         chpt[1] = invH[1]:dot(wpt) + w_imgs/2
         if chpt[1]<1 or chpt[1]>w_imgs then
            -- print('')
            -- print('oulier!')
            -- print('xPatch ' .. xPatch .. ' yPatch ' .. yPatch)
            -- print('x ' .. x .. ' y ' .. y)
            -- print('wpt.x ' .. wpt[1] .. ' wpt.y ' .. wpt[2])
            -- print('chpt.x ' .. chpt[1] .. ' chpt.y ' .. chpt[2])
            return false
         end

         chpt[2] = invH[2]:dot(wpt) + h_imgs/2
         if chpt[2]<1 or chpt[2]>h_imgs then
            -- print('')
            -- print('oulier!')
            -- print('xPatch ' .. xPatch .. ' yPatch ' .. yPatch)
            -- print('x ' .. x .. ' y ' .. y)
            -- print('wpt.x ' .. wpt[1] .. ' wpt.y ' .. wpt[2])
            -- print('chpt.x ' .. chpt[1] .. ' chpt.y ' .. chpt[2])
            return false
         end
      end
   end
   return true
end

function generateDataOpticalFlowCCLiu(correction, geometry, learning, raw_data, nSamples)
   local dataset = {}
   dataset.raw_data = raw_data
   dataset.patches = torch.Tensor(nSamples, 6)
   dataset.targets = torch.Tensor(nSamples, 2)
   function dataset:size()
      return nSamples
   end
   setmetatable(dataset, {__index = function(self, index)
				       local coords = self.patches[index]
				       local image1, image2
				       if geometry.motion_correction == 'sfm' then
					  image1 = self.raw_data.warped_images[coords[1]]
					  image2 = self.raw_data.images[coords[2]]
				       elseif geometry.motion_correction == 'mc' then
					  image1 = self.raw_data.images[coords[1]]
					  image2 = self.raw_data.rectified_images[coords[1]]
				       else
					  image1 = self.raw_data.images[coords[1]]
					  image2 = self.raw_data.images[coords[2]]
				       end
				       local patch1 = image1:sub(1, image1:size(1),
								 coords[3], coords[4],
								 coords[5], coords[6])
				       local patch2 = image2:sub(1, image2:size(1),
								 coords[3], coords[4],
								 coords[5], coords[6])
				       return {{patch1, patch2}, self.targets[index]}
				    end})

   local hoffset = math.ceil(geometry.maxhGT/2) + math.ceil(geometry.hKernelGT/2) - 2+1
   local woffset = math.ceil(geometry.maxwGT/2) + math.ceil(geometry.wKernelGT/2) - 2+1
   function dataset:getElemFovea(index)
      local coords = self.patches[index]
      return {{{self.raw_data.images[coords[1]], self.raw_data.images[coords[2]]},
	       {coords[3]+hoffset, coords[5]+woffset}}, self.targets[index]}
   end

   local iSample = 1
   while iSample <= nSamples do
      --modProgress(iSample, nSamples, 100)
      local iImg = randInt(2, #raw_data.images+1)
      
      local yPatch = randInt(1, geometry.hImg-geometry.maxhGT-geometry.hKernelGT-1)
      local xPatch = randInt(1, geometry.wImg-geometry.maxwGT-geometry.wKernelGT-1)

      local yCenter = yPatch+hoffset
      local xCenter = xPatch+woffset
      
      local yFlow = raw_data.flow[iImg-1][1][yCenter][xCenter]
      local xFlow = raw_data.flow[iImg-1][2][yCenter][xCenter]
      
      dataset.patches[iSample][1] = iImg-1
      dataset.patches[iSample][2] = iImg
      dataset.patches[iSample][3] = yPatch
      dataset.patches[iSample][4] = yPatch+geometry.hPatch2-1
      dataset.patches[iSample][5] = xPatch
      dataset.patches[iSample][6] = xPatch+geometry.wPatch2-1
      
      dataset.targets[iSample][1] = yFlow
      dataset.targets[iSample][2] = xFlow
      
      if geometry.motion_correction == 'mc' then
	 if check_borders(iImg, xPatch, yPatch, geometry) then
	    iSample = iSample+1
	 end
      elseif geometry.motion_correction == 'sfm' then
	 local hk = math.ceil(geometry.hKernel/2)
	 local wk = math.ceil(geometry.wKernel/2)
	 if (raw_data.warped_masks[iImg-1][yCenter-hk][xCenter-wk] > 0.5) and
	    (raw_data.warped_masks[iImg-1][yCenter+hk][xCenter-wk] > 0.5) and
	    (raw_data.warped_masks[iImg-1][yCenter+hk][xCenter+wk] > 0.5) and
	    (raw_data.warped_masks[iImg-1][yCenter-hk][xCenter+wk] > 0.5) then
	    iSample = iSample+1
	 end
      else
	 iSample = iSample+1
      end
   end

   return dataset
end

function generateDataOpticalFlowCVlibs(geometry, learning, raw_data, nSamples)
   assert(not geometry.motion_correction)
   local dataset = {}
   dataset.raw_data = raw_data
   dataset.patches = torch.Tensor(nSamples, 5)
   dataset.targets = torch.Tensor(nSamples, 2)
   function dataset:size()
      return nSamples
   end
   setmetatable(dataset, {__index = function(self, index)
				       local coords = self.patches[index]
				       local image1 = self.raw_data[coords[1]].image1
				       local image2 = self.raw_data[coords[1]].image2
				       local patch1 = image1:sub(1, image1:size(1),
								 coords[2], coords[3],
								 coords[4], coords[5])
				       local patch2 = image2:sub(1, image2:size(1),
								 coords[2], coords[3],
								 coords[4], coords[5])
				       return {{patch1, patch2}, self.targets[index]}
				    end})

   local hoffset = math.ceil(geometry.maxhGT/2) + math.ceil(geometry.hKernelGT/2) - 2+1
   local woffset = math.ceil(geometry.maxwGT/2) + math.ceil(geometry.wKernelGT/2) - 2+1
   function dataset:getElemFovea(index)
      local coords = self.patches[index]
      return {{{self.raw_data[coords[1]].image1, self.raw_data[coords[1]].image2},
	       {coords[2]+hoffset, coords[4]+woffset}}, self.targets[index]}
   end

   local iSample = 1
   while iSample <= nSamples do
      --modProgress(iSample, nSamples, 100)
      local iImg = randInt(1, #raw_data+1)

      local good = false
      while not good do
	 local yPatch = randInt(1, geometry.hImg-geometry.maxhGT-geometry.hKernelGT-1)
	 local xPatch = randInt(1, geometry.wImg-geometry.maxwGT-geometry.wKernelGT-1)

	 good = (raw_data[iImg].flow_noc_mask[yPatch+hoffset][xPatch+woffset] > 0.5)
	 if good then
	    
	    local yFlow = raw_data[iImg].flow_noc[1][yPatch+hoffset][xPatch+woffset]
	    local xFlow = raw_data[iImg].flow_noc[2][yPatch+hoffset][xPatch+woffset]
	    
	    dataset.patches[iSample][1] = iImg
	    dataset.patches[iSample][2] = yPatch
	    dataset.patches[iSample][3] = yPatch+geometry.hPatch2-1
	    dataset.patches[iSample][4] = xPatch
	    dataset.patches[iSample][5] = xPatch+geometry.wPatch2-1
      
	    dataset.targets[iSample][1] = yFlow
	    dataset.targets[iSample][2] = xFlow
      
	    iSample = iSample+1
	 end
      end
   end

   return dataset   
end

function generateDataOpticalFlow(correction, geometry, learning, raw_data, nSamples)
   if (learning.groundtruth == 'liu') or (learning.groundtruth == 'cross-correlation') then
      return generateDataOpticalFlowCCLiu(correction, geometry, learning, raw_data, nSamples)
   elseif learning.groundtruth == 'cvlibs' then
      return generateDataOpticalFlowCVlibs(geometry, learning, raw_data, nSamples)
   else
      error('generateDataOpticalFlow: learning.groundtruth must be either liu, cvlibs or cross-correlation')
   end
end