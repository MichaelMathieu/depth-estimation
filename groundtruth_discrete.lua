require 'torch'
require 'paths'
require 'sys'
require 'xlua'
require 'image'

require 'common'
require 'motion_correction'

depthDiscretizer = {}
depthDiscretizer.nClasses = -1
depthDiscretizer.cutDepth = 0
depthDiscretizer.binStep = 0

function depthDiscretizer:getClass(depth)
   local step = 2*self.cutDepth/self.nClasses
   local class = math.ceil(depth/step)
   if class > self.nClasses then
      return self.nClasses
   end
   return class
end

function depthDiscretizer:computeCutDepth(histogram, nUsedPatches)
   local nPerClass = torch.Tensor(self.nClasses):zero()
      
   local numberOfSamples = 0
   for i = 1,numberOfBins do
      numberOfSamples = numberOfSamples + #(histogram[i])
      if numberOfSamples > nUsedPatches/2 then
	 self.cutDepth = i
	 break
      end
   end
   print("cutDepth is " .. self.cutDepth)
   
   for i = 1,numberOfBins do
      --print("Bin " .. i .. " has " .. #(patchesMedianDepth[i]) .. " elements")
      nPerClass[self:getClass(i-0.1)] = nPerClass[self:getClass(i-0.1)] + #(histogram[i])
   end
   for i = 1,self.nClasses do
      print("Class " .. i .. " has " .. nPerClass[i] .. " elements")
   end

   self.binStep = math.floor(2*self.cutDepth/self.nClasses)
end

function depthDiscretizer:randomBin(histogram)
   local randomClass = randInt(1, self.nClasses+1)
   local randomBinIndex
   local sizeOfBin = 0
   while sizeOfBin == 0 do
      randomBinIndex = randInt((randomClass-1)*self.binStep+1, randomClass*self.binStep+1)
      sizeOfBin = table.getn(histogram[randomBinIndex])
   end
   return randomBinIndex
end

raw_data = {}
w_imgs = 640
h_imgs = 360
depthHistogram = {}
numberOfBins = 0
maxDepth = 0
H = {}
warpImg = {}

function preSortDataDiscrete(wPatch, hPatch, use_median, use_motion_correction)
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

   if use_motion_correction then
      print("Computing motion cor rection H...")
      for iImg = 1,#raw_data-1 do
         xlua.progress(iImg, #raw_data-1)
         local ptsin = opencv.GoodFeaturesToTrack{image=raw_data[iImg][1], count=50}
         local ptsout = opencv.TrackPyrLK{pair={raw_data[iImg][1],raw_data[iImg+1][1]},
                           points_in=ptsin}
         local dx, dy, dtheta
         H[iImg],dx,dy,dtheta = lsq_trans(ptsin, ptsout, w_imgs/2, h_imgs/2)
         local inputImg = raw_data[iImg+1][1]:clone()
         warpImg[iImg] = opencv.WarpAffine(inputImg,H[iImg])
      end
   end

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

   depthDiscretizer:computeCutDepth(depthHistogram, usedPatchesNumber)
end

function generateDataDiscrete(nSamples, wPatch, hPatch, is_train, use_2_pics, use_motion_correction)
   local dataset = {}
   if use_2_pics then
      dataset.patches = torch.Tensor(nSamples, 2, hPatch, wPatch)
   else
      print("Using one pic")
      dataset.patches = torch.Tensor(nSamples, 1, hPatch, wPatch)
   end
   dataset.targets = torch.Tensor(nSamples, depthDiscretizer.nClasses):zero()
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
      local randomBinIndex = depthDiscretizer:randomBin(depthHistogram)
      local randomPatchIndex = randInt(1, table.getn(depthHistogram[randomBinIndex])+1)
      local patch_descr = depthHistogram[randomBinIndex][randomPatchIndex]
      local im_index = patch_descr[2]
      local y = math.ceil(patch_descr[3])
      local x = math.ceil(patch_descr[4])
      local patch = image.rgb2y(raw_data[im_index][1]:sub(1, 3,
							  y-hPatch/2, y+hPatch/2-1,
							  x-wPatch/2, x+wPatch/2-1))
      dataset.patches[nGood][1]:copy(patch)
      dataset.targets[nGood][depthDiscretizer:getClass(depthHistogram[randomBinIndex][randomPatchIndex][1])] = 1
      if use_2_pics then
         if use_motion_correction then

            patch2 = image.rgb2y(warpImg[im_index]:sub(1, 3,
                              y-hPatch/2, y+hPatch/2-1,
                              x-wPatch/2, x+wPatch/2-1))
            dataset.patches[nGood][2]:copy(patch2)
            
            local wpt = torch.Tensor(2)
            local chpt = torch.Tensor(2)
            local invH = torch.inverse(H[im_index]:sub(1,2,1,2))
            
            for i=0,1 do
               for j=0,1 do
                  wpt[1] = x - w_imgs/2 + wPatch*(i-0.5) - H[im_index][1][3]
                  wpt[2] = y - h_imgs/2 + hPatch*(j-0.5) - H[im_index][2][3]
                  
                  chpt[1] = invH[1]:dot(wpt) + w_imgs/2
                  if chpt[1]<1 or chpt[1]>w_imgs then
                     if nGood>1 then nGood = nGood-1 end
                     break
                  end

                  chpt[2] = invH[2]:dot(wpt) + h_imgs/2
                  if chpt[2]<1 or chpt[2]>h_imgs then
                     if nGood>1 then nGood = nGood-1 end
                     break
                  end

               end
            end

         else
            patch2 = image.rgb2y(raw_data[im_index+1][1]:sub(1, 3,
                              y-hPatch/2, y+hPatch/2-1,
                              x-wPatch/2, x+wPatch/2-1))
            dataset.patches[nGood][2]:copy(patch2)
         end
      end
      
      nGood = nGood + 1
   end
   
   nPerClass = torch.Tensor(depthDiscretizer.nClasses):zero()
   for i = 1,nGood-1 do
      for j = 1,depthDiscretizer.nClasses do
	 if dataset.targets[i][j] == 1 then
	    nPerClass[j] = nPerClass[j] + 1
	 end
      end
   end
   print("Done :")
   for i = 1,depthDiscretizer.nClasses do
      print("Class " .. i .. " has " .. nPerClass[i] .. " patches")
   end
   
   return dataset
end