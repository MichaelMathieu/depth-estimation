require 'torch'
require 'paths'
require 'image'
require 'sys'
require 'xlua'
require 'opencv'
require 'motion_correction'

raw_data = {}
H = {}
warpImg = {}
w_imgs = 640
h_imgs = 360
patchesMedianDepth = {}
nClasses = 2
maxDepth = 0
cutDepth = 0
numberOfBins = 0
use_motion_correction = true

function loadCameras(dirbasename)
   local filename = dirbasename .. 'depths/cameras'
   if not paths.filep(filename) then
      print('File' .. filename .. ' not found. Can\'t read camera positions.')
      return nil
   end
   local file = torch.DiskFile(filename, 'r')
   file:quiet()
   local ret = {}
   while true do
      local cam = {}
      cam.file = file:readString('*l')
      cam.f = file:readDouble()
      cam.k1 = file:readDouble()
      cam.k2 = file:readDouble()
      cam.R = torch.Tensor(3,3)
      for i = 1,3 do
	 for j = 1,3 do
	    cam.R[i][j] = file:readDouble()
	 end
      end
      cam.t = torch.Tensor(3)
      for i = 1,3 do
	 cam.t[i] = file:readDouble()
      end
      if file:hasError() then
	 return ret
      else
	 table.insert(ret, cam)
      end
   end
   file:close()
end

function loadImage(dirbasename, filebasename)
   local imfilename = dirbasename .. 'images/' .. filebasename .. '.jpg'
   local depthfilename = dirbasename .. 'depths/' .. filebasename .. '.mat'
   if not paths.filep(imfilename) then
      print('File ' .. imfilename .. ' not found. Skipping...')
      return
   end
   if not paths.filep(depthfilename) then
      print('File ' .. depthfilename .. ' not found. Skipping...')
      return
   end
   local im = image.loadJPG(imfilename)
   local h_im = im:size(2)
   local w_im = im:size(3)
   im = image.scale(im, w_imgs, h_imgs)
   local file_depth = torch.DiskFile(depthfilename, 'r')
   local nPts = file_depth:readInt()
   local depthPoints = torch.Tensor(nPts, 3)
   for i = 1,nPts do
      -- todo : indices are wrong by one because of the indexing from 1
      depthPoints[i][1] = file_depth:readInt() * h_imgs / h_im
      depthPoints[i][2] = file_depth:readInt() * w_imgs / w_im
      depthPoints[i][3] = file_depth:readDouble()
   end
   table.insert(raw_data, {im, depthPoints})
   file_depth:close()
end

function getClass(depth)
   local step = 2*cutDepth/nClasses
   local class = math.ceil(depth/step)
   if class > nClasses then
      return nClasses
   end
   return class
end

function randInt(a, b) --a included, b excluded
   return math.floor(torch.uniform(a, b))
end

function randomPermutation(n)
   local ret = torch.Tensor(n)
   for i = 1,n do
      local rnd = randInt(1, i+1)
      ret[i] = ret[rnd]
      ret[rnd] = i
   end
   return ret
end

-- Get the median of a table.
function median(t)
  local temp={}

  -- deep copy table so that when we sort it, the original is unchanged
  -- also weed out any non numbers
  for k,v in pairs(t) do
    if type(v) == 'number' then
      table.insert( temp, v )
    end
 end

  if (#temp == 0) then
     print("error: median : empty table")
  end

  table.sort( temp )

  -- If we have an even number of table elements or odd.
  if math.fmod(#temp,2) == 0 then
    -- return mean value of middle two elements
    return ( temp[#temp/2] + temp[(#temp/2)+1] ) / 2
  else
    -- return middle elements
    return temp[math.ceil(#temp/2)]
  end
end

function preSortData(wPatch, hPatch, use_median)
   -- Merge the patches of all the images
   print("Merging the data from all the images...")
   local numberOfPatches = 0
   for iImg = 1,#raw_data do
      numberOfPatches = numberOfPatches + raw_data[iImg][2]:size(1)
   end
   local patches = torch.Tensor(numberOfPatches, 4)
   local iPatch = 1
   maxDepth = 0
   for iImg = 1,#raw_data-1 do
      xlua.progress(iImg, #raw_data-1)
      for iPatchInImg = 1,raw_data[iImg][2]:size(1) do
	 patches[iPatch][1] = iImg
	 patches[iPatch][2] = raw_data[iImg][2][iPatchInImg][1]
	 patches[iPatch][3] = raw_data[iImg][2][iPatchInImg][2]
	 patches[iPatch][4] = raw_data[iImg][2][iPatchInImg][3]
	 if (patches[iPatch][4] > maxDepth) then
	    maxDepth = patches[iPatch][4]
	 end
	 iPatch = iPatch + 1
      end
   end
   numberOfBins = math.ceil(maxDepth)
   print("maxDepth is " .. maxDepth)

   if use_motion_correction then
      print("Computing motion correction H...")
      for iImg = 1,#raw_data-1 do
         xlua.progress(iImg, #raw_data-1)
         local ptsin = opencv.GoodFeaturesToTrack{image=raw_data[iImg+1][1], count=50}
         local ptsout = opencv.TrackPyrLK{pair={raw_data[iImg+1][1],raw_data[iImg][1]},
                           points_in=ptsin}
         H[iImg] = lsq_trans(ptsin, ptsout, w_imgs/2, h_imgs/2)
         --warpImg[iImg] = opencv.WarpAffine(raw_data[iImg+1][1],H[iImg])
      end
   end

   -- Compute histogram
   --print("Calculating patches median depth...")
   print("Computing the histogram of depths...")
   local ySorted,sorti
   if use_median then
      ySorted,sorti = torch.sort(patches:select(2,3), 1)
   end
   
   for iBin = 1,numberOfBins do
      patchesMedianDepth[iBin] = {}
   end
   
   local firstIndex = true
   local lastPatchIndex = 1
   local i
   local usedPatchesNumber = 0
   xlua.progress(0, numberOfPatches)
   for origi = 1,numberOfPatches do
      if (math.mod(origi, 1000) == 0) then
	 xlua.progress(origi, numberOfPatches)
      end
      if use_median then
	 i = sorti[origi]
      else
	 i = origi
      end
      local imo = patches[i][1]
      local yo = patches[i][2]
      local xo = patches[i][3]
      if (yo-hPatch/2 >= 1) and (yo+hPatch/2-1 <= h_imgs) and
         (xo-wPatch/2 >= 1) and (xo+wPatch/2-1 <= w_imgs) then
    
      usedPatchesNumber = usedPatchesNumber + 1

	 local currentPatchMedianDepth = 0
	 if use_median then
	    local currentPatchPts = {}
	    for origj = lastPatchIndex,numberOfPatches do
	       local j = sorti[origj]
	       local x = patches[j][3]
	       if x>=xo+wPatch/2 then
		  firstIndex = true
		  break
	       end
	       if x>=xo-wPatch/2 then
		  if firstIndex then
		     lastPatchIndex = origj
		     firstIndex = false
		  end
		  local y = patches[j][2]
		  if (y>=yo-hPatch/2) and (y<=yo+hPatch-1/2) then
		     local depth = patches[j][4]
		     table.insert(currentPatchPts, depth)		  
		  end
	       end
	    end
	    currentPatchMedianDepth = median(currentPatchPts)
	 else
	    currentPatchMedianDepth = patches[i][4]
	 end

	 local binIndex = math.ceil(currentPatchMedianDepth)
	 table.insert(patchesMedianDepth[binIndex], {currentPatchMedianDepth, imo, yo, xo})
	 
      end
   end
   print("")
   nPerClass = torch.Tensor(nClasses):zero()
   
   local numberOfSamples = 0
   for i = 1,numberOfBins do
      numberOfSamples = numberOfSamples + #(patchesMedianDepth[i])
      if (numberOfSamples>usedPatchesNumber/2) then
         cutDepth = i
         break
      end
   end
   print("cutDepth is " .. cutDepth)

   for i = 1,numberOfBins do
      --print("Bin " .. i .. " has " .. #(patchesMedianDepth[i]) .. " elements")
      nPerClass[getClass(i-0.1)] = nPerClass[getClass(i-0.1)] + #(patchesMedianDepth[i])
   end
   for i = 1,nClasses do
      print("Class " .. i .. " has " .. nPerClass[i] .. " elements")
   end
end

function generateData(nSamples, wPatch, hPatch, is_train, use_2_pics)
   local dataset = {}
   if use_2_pics then
      dataset.patches = torch.Tensor(nSamples, 2, hPatch, wPatch)
   else
      print("Using one pic")
      dataset.patches = torch.Tensor(nSamples, 1, hPatch, wPatch)
   end
   dataset.targets = torch.Tensor(nSamples, nClasses):zero()
   dataset.permutation = randomPermutation(nSamples)
   setmetatable(dataset, {__index = function(self, index_)
				       local index = self.permutation[index_]
				       return {self.patches[index], self.targets[index]}
				    end})
   function dataset:size()
      return nSamples
   end   
   
   print("Sampling patches...")
   local binStep = math.floor(2*cutDepth/nClasses)
   local nPerClass = torch.Tensor(nClasses):zero()
   local nGood = 1
   while nGood <= nSamples do
      local randomClass = randInt(1,nClasses+1)
      local sizeOfBin = 0
      local randomBinIndex = 0
      while sizeOfBin == 0 do
         randomBinIndex = randInt((randomClass-1)*binStep+1, (randomClass)*binStep+1)
         sizeOfBin = table.getn(patchesMedianDepth[randomBinIndex])
      end
      local randomPatchIndex = randInt(1, sizeOfBin+1)
      local patch_descr = patchesMedianDepth[randomBinIndex][randomPatchIndex]
      local im_index = math.ceil(patch_descr[2])
      local y = math.ceil(patch_descr[3])
      local x = math.ceil(patch_descr[4])
      local patch = image.rgb2y(raw_data[im_index][1]:sub(1, 3,
							  y-hPatch/2, y+hPatch/2-1,
							  x-wPatch/2, x+wPatch/2-1))
      dataset.patches[nGood][1]:copy(patch)
      if use_2_pics then
         if use_motion_correction then
            patch2 = torch.Tensor(wPatch,hPatch)
            for i=-wPatch/2,wPatch/2-1 do
            for j=-hPatch/2,hPatch/2-1 do
               pt = torch.Tensor(3)
               wpt = torch.Tensor(2) 
               pt[1] = x+i
               pt[2] = x+j
               pt[3] = 1
               wpt[1] = math.ceil(H[im_index][1]:dot(pt))
               wpt[2] = math.ceil(H[im_index][2]:dot(pt))
               if wpt[1]<1 then wpt[1] = 1 end
               if wpt[1]>h_imgs then wpt[1] = h_imgs end
               if wpt[2]<1 then wpt[2] = 1 end
               if wpt[2]>w_imgs then wpt[2] = w_imgs end
               patch2[i+wPatch/2+1][j+hPatch/2+1] = image.rgb2y(raw_data[im_index+1][1]:sub(1,3,
                                                      wpt[1],wpt[1],wpt[2],wpt[2]))
            end
            end
         else
            local patch2 = image.rgb2y(raw_data[im_index+1][1]:sub(1, 3,
                           y-hPatch/2, y+hPatch/2-1,
                           x-wPatch/2, x+wPatch/2-1))
         end
         dataset.patches[nGood][2]:copy(patch2)
      end
      local class = getClass(patchesMedianDepth[randomBinIndex][randomPatchIndex][1])
      nPerClass[class] = nPerClass[class] + 1
      dataset.targets[nGood][class] = 1
      nGood = nGood + 1
   end
   
   print("Done :")
   for i = 1,nClasses do
      print("Class " .. i .. " has " .. nPerClass[i] .. " patches")
   end
   
   return dataset
end   

function loadData(nImgs, delta, geometry, root_dir)
   --print("Loading images")
   local directories = {}
   local nDirs = 0
   local findIn = 'find -L ' .. root_dir .. ' -name images'
   for i in io.popen(findIn):lines() do
      nDirs = nDirs + 1
      directories[nDirs] = string.gsub(i, "images", "")
   end
   --local imagesPerDir = math.floor(nImgs/nDirs)
   local imagesPerDir = nImgs
   for j=1,nDirs do
      print("")
      print("Loading " .. imagesPerDir .. " images from " .. directories[j])
      
      local blacklist = {}
      local nBl
      local bl = torch.DiskFile(directories[j] .. 'images/blacklist.txt',r,true)
      if (bl == nil) then
         nBl = 0
      else
         nBl = bl:readInt()
         for iBl = 0, nBl-1 do blacklist[iBl] = bl:readInt() end
      end
      print('- ' .. nBl .. ' images in blacklist')
      
      for i = 0,imagesPerDir-1 do
         xlua.progress(imagesPerDir*(j-1)+i+1, nImgs*nDirs)
         local imageId = i*delta
         
         local isInBl = false
         for iBl = 0, nBl-1 do
            if (blacklist[iBl] == imageId) then
               isInBl = true
               break
            end
         end
         if (isInBl) then
            print("")
            print('Skipping image ' .. string.format("%09d", imageId))
         else
            loadImage(directories[j],string.format("%09d", imageId))
         end
      end
   end
   print("Pre-sorting images")
   preSortData(geometry[1], geometry[2], false) --temporarly not using median to improve speed
end