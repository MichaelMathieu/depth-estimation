require 'torch'
require 'paths'
require 'sys'
require 'xlua'

require 'common'
require 'groundtruth_descrete'

raw_data = {}
w_imgs = 640
h_imgs = 360
depthHistogram = {}
numberOfBins = 0
maxDepth = 0

function preSortDataDescrete(wPatch, hPatch, use_median)
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
      depthHistogram[iBin] = {}
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
	 table.insert(depthHistogram[binIndex], {currentPatchMedianDepth, imo, yo, xo})
      end
   end
   print("")

   depthDescretizer:computeCutDepth(depthHistogram, usedPatchesNumber)
end

function generateDataDescrete(nSamples, wPatch, hPatch, is_train, use_2_pics)
   local dataset = {}
   if use_2_pics then
      dataset.patches = torch.Tensor(nSamples, 2, hPatch, wPatch)
   else
      print("Using one pic")
      dataset.patches = torch.Tensor(nSamples, 1, hPatch, wPatch)
   end
   dataset.targets = torch.Tensor(nSamples, depthDescretizer.nClasses):zero()
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
      local randomBinIndex = depthDescretizer:randomBin(depthHistogram)
      local randomPatchIndex = randInt(1, table.getn(depthHistogram[randomBinIndex])+1)
      local patch_descr = depthHistogram[randomBinIndex][randomPatchIndex]
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

      dataset.targets[nGood][depthDescretizer:getClass(depthHistogram[randomBinIndex][randomPatchIndex][1])] = 1
      nGood = nGood + 1
   end
   
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
   
   return dataset
end