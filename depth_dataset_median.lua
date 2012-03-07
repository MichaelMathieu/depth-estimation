require 'torch'
require 'paths'
require 'sys'
require 'xlua'

require 'load_data'
require 'common'
require 'groundtruth_descrete'

raw_data = {}
w_imgs = 640
h_imgs = 360
patchesMedianDepth = {}
numberOfBins = 0
maxDepth = 0

function preSortData(wPatch, hPatch, use_median, use_continuous)
   -- Merge the patches of all the images
   print("Merging the data from all the images...")
   local numberOfPatches = 0
   for iImg = 1,#raw_data do
      numberOfPatches = numberOfPatches + raw_data[iImg][2]:size(1)
   end
   local patches = torch.Tensor(numberOfPatches, 4)
   local iPatch = 1
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

   -- Compute histogram
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
      modProgress(origi, numberOfPatches, 1000);
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

   if not use_continuous then
      depthDescretizer:computeCutDepth(patchesMedianDepth, usedPatchesNumber)
   end
end

function generateData(nSamples, wPatch, hPatch, is_train, use_2_pics, use_continuous)
   local dataset = {}
   if use_2_pics then
      dataset.patches = torch.Tensor(nSamples, 2, hPatch, wPatch)
   else
      print("Using one pic")
      dataset.patches = torch.Tensor(nSamples, 1, hPatch, wPatch)
   end
   if use_continuous then
      dataset.targets = torch.Tensor(nSamples, 1):zero()
   else
      dataset.targets = torch.Tensor(nSamples, depthDescretizer.nClasses):zero()
   end
   dataset.permutation = randomPermutation(nSamples)
   setmetatable(dataset, {__index = function(self, index_)
				       local index = self.permutation[index_]
				       return {self.patches[index], self.targets[index]}
				    end})
   function dataset:size()
      return nSamples
   end
   
   print("Sampling patches...")
   local nGood = 1
   while nGood <= nSamples do
      local randomBinIndex = 0
      if use_continuous then
	 local sizeOfBin = 0
	 while sizeOfBin == 0 do
	    print('Empty bin : ' .. randomBinIndex .. ' with continuous output. This will bias the normalization')
	    randomBinIndex = randInt(1, numberOfBins+1)
	    sizeOfBin = table.getn(patchesMedianDepth[randomBinIndex])
	 end
      else
	 randomBinIndex = depthDescretizer:randomBin(patchesMedianDepth)
      end

      local randomPatchIndex = randInt(1, table.getn(patchesMedianDepth[randomBinIndex])+1)
      local patch_descr = patchesMedianDepth[randomBinIndex][randomPatchIndex]
      local im_index = patch_descr[2]
      local y = math.ceil(patch_descr[3])
      local x = math.ceil(patch_descr[4])
      local patch = image.rgb2y(raw_data[im_index][1]:sub(1, 3,
							  y-hPatch/2, y+hPatch/2-1,
							  x-wPatch/2, x+wPatch/2-1))
      dataset.patches[nGood][1]:copy(patch)
      if use_2_pics then
	 local patch2 = image.rgb2y(raw_data[im_index+1][1]:sub(1, 3,
								y-hPatch/2, y+hPatch/2-1,
								x-wPatch/2, x+wPatch/2-1))
	 dataset.patches[nGood][2]:copy(patch2)
      end
      if use_continuous then
	 dataset.targets[nGood][1] = patch_descr[1]
      else
	 dataset.targets[nGood][depthDescretizer:getClass(patchesMedianDepth[randomBinIndex][randomPatchIndex][1])] = 1
      end
      nGood = nGood + 1
   end
   
   if not use_continuous then
      nPerClass = torch.Tensor(depthDescretizer.nClasses):zero()
      for i = 1,nGood-1 do
	 for j = 1,depthDescretizer.nClasses do
	    if dataset.targets[i][j] == 1 then
	       nPerClass[j] = nPerClass[j] + 1
	    end
	 end
      end
      print("Done :")
      for i = 1,depthDescretizer.nClasses do
	 print("Class " .. i .. " has " .. nPerClass[i] .. " patches")
      end
   end
   
   return dataset
end   

function loadData(nImgs, delta, geometry, root_dir, use_continuous)
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
            table.insert(raw_data, loadImage(directories[j],string.format("%09d", imageId)))
         end
      end
   end
   print("Pre-sorting images")
   preSortData(geometry[1], geometry[2], false, use_continuous)
end