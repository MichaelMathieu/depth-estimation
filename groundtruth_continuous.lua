require 'torch'
require 'xlua'
require 'common'

function getBinFromDepth(nBins, depth)
   if depth < 0 then
      return 1
   end
   if depth >= 1 then
      return nBins
   else
      return math.floor(depth*nBins)+1
   end
end

function isInFrame(geometry, y, x)
   return (x-geometry.wPatch/2 >= 1) and (y-geometry.hPatch/2 >= 1) and
          (x+geometry.wPatch/2 <= geometry.wImg) and (y+geometry.hPatch/2 <= geometry.hImg)
end

function preSortDataContinuous(geometry, raw_data, maxDepth, nBins, keep_only_tracked_pts)
   local nPatches = 0
   for iImg = 1,#raw_data do
      nPatches = nPatches + raw_data[iImg][2]:size(1)
   end

   local data = {}
   data.patches = torch.Tensor(nPatches, 5) -- patch: (iImg, y, x, depth, nextOccurence)
   data.histogram = {} -- contains only usable patches (if keep_only_tracked_pts)
   data.perId = {}
   data.images = torch.Tensor(#raw_data, geometry.hImg, geometry.wImg)

   -- find min and max depths
   local minDepth = 1e20
   local maxDepthData = 0
   for iImg = 1,#raw_data do
      for iPatchInImg = 1,raw_data[iImg][2]:size(1) do
	 local depth = raw_data[iImg][2][iPatchInImg][3]
	 if depth < minDepth then
	    minDepth = depth
	 end
	 if depth > maxDepthData then
	    maxDepthData = depth
	 end
      end
   end
   maxDepth = math.min(maxDepth, maxDepthData)
   
   -- get patches geometry (iImg, y, x, depth) and fill data.perId and data.images
   local iPatch = 1
   for iImg = 1,#raw_data do
      xlua.progress(iImg, #raw_data)
      for iPatchInImg = 1,raw_data[iImg][2]:size(1) do
	 local y = round(raw_data[iImg][2][iPatchInImg][1]) + 1
	 local x = round(raw_data[iImg][2][iPatchInImg][2]) + 1
	 local depth = (raw_data[iImg][2][iPatchInImg][3] - minDepth) / (maxDepth-minDepth)
	 local id = raw_data[iImg][2][iPatchInImg][4]
	 if isInFrame(geometry, y, x) and depth <= 1 then
	    data.patches[iPatch][1] = iImg
	    data.patches[iPatch][2] = y
	    data.patches[iPatch][3] = x
	    data.patches[iPatch][4] = depth
	    data.patches[iPatch][5] = -1
	    if data.perId[id] == nil then
	       data.perId[id] = {}
	    end
	    table.insert(data.perId[id], iPatch)
	    iPatch = iPatch + 1
	 end
      end
      data.images[iImg] = image.rgb2y(raw_data[iImg][1])
   end
   nPatches = iPatch-1
   data.patches = data.patches:narrow(1, 1, nPatches)

   -- get patches next occurences (nextOccurence)
   for iId,perId in pairs(data.perId) do
      for iOcc = 1,(#perId-1) do
	 local iPatch = perId[iOcc]
	 local iPatchNext = perId[iOcc+1]
	 if data.patches[iPatchNext][1] == data.patches[iPatch][1]+1 then
	    data.patches[iPatch][5] = iPatchNext
	 end
      end
   end

   -- fill data.histogram
   --   fill histogram
   for i = 1,nBins do
      data.histogram[i] = {}
   end
   for iPatch = 1,nPatches do
      local iCurrentPatch = iPatch
      local goodPatch = true
      if keep_only_tracked_pts then
	 for i = 1,geometry.nImgsPerSample-1 do
	    iCurrentPatch = data.patches[iCurrentPatch][5]
	    if iCurrentPatch == -1 then
	       goodPatch = false
	       break
	    end
	 end
      end
      if goodPatch then
	 local iBin = getBinFromDepth(nBins, data.patches[iPatch][4])
	 table.insert(data.histogram[iBin], iPatch)
      end
   end
   --   prune small far classes
   -- todo

   print("Histogram:")
   print(data.histogram)
   -- check for empty bins
   for iBin = 1,nBins do
      if #(data.histogram[iBin]) == 0 then
	 print('Error: data.histogram[' .. iBin .. '] is empty. Use more data or larger bins.')
	 return nil
      end
   end

   return data
end

function generateContinuousDataset(geometry, data, nSamples)
   local dataset = {}
   dataset.patches = torch.Tensor(nSamples, geometry.nImgsPerSample,
				  geometry.hPatch, geometry.wPatch)
   dataset.targets = torch.Tensor(nSamples, 1):zero()
   function dataset:size()
      return nSamples
   end
   setmetatable(dataset, {__index = function(self, index)
				       return {self.patches[index], self.targets[index]}
				    end})

   for iSample = 1,nSamples do
      local iBin = randInt(1, #data.histogram+1)
      local iPatch = data.histogram[iBin][randInt(1, #data.histogram[iBin]+1)]
      dataset.targets[iSample][1] = data.patches[iPatch][4]

      local y = data.patches[iPatch][2]
      local x = data.patches[iPatch][3]
      local iImg = data.patches[iPatch][1]

      for iImgIndex = 1,geometry.nImgsPerSample do
	 dataset.patches[iSample][iImgIndex] =
	    data.images[iImg]:sub(y-geometry.hPatch/2, y+geometry.hPatch/2-1,
				  x-geometry.wPatch/2, x+geometry.wPatch/2-1)
	 iImg = iImg + 1
      end
   end
   
   return dataset
end

function generateContinuousDatasetOpticalFlow(geometry, data, nSamples)
   assert(geometry.nImgsPerSample == 2) -- optical flow is between 2 images
   local dataset = {}
   dataset.patches = torch.Tensor(nSamples, geometry.nImgsPerSample,
				  geometry.hPatch, geometry.wPatch)
   dataset.targets = torch.Tensor(nSamples, 2):zero()
   function dataset:size()
      return nSamples
   end
   setmetatable(dataset, {__index = function(self, index)
				       return {self.patches[index], self.targets[index]}
				    end})

   meandist = torch.Tensor(2):zero()
   for iSample = 1,nSamples do
      local iBin = randInt(1, #data.histogram+1)
      local iPatch1 = data.histogram[iBin][randInt(1, #data.histogram[iBin]+1)]
      local iImg1 = data.patches[iPatch1][1]
      local y1 = data.patches[iPatch1][2]
      local x1 = data.patches[iPatch1][3]
      local iPatch2 = data.patches[iPatch1][5]
      local iImg2 = data.patches[iPatch2][1]
      local y2 = data.patches[iPatch2][2]
      local x2 = data.patches[iPatch2][3]

      dataset.patches[iSample][1] =
	 data.images[iImg1]:sub(y1-geometry.hPatch/2, y1+geometry.hPatch/2-1,
				x1-geometry.wPatch/2, x1+geometry.wPatch/2-1)
      dataset.patches[iSample][2] =
	 data.images[iImg2]:sub(y1-geometry.hPatch/2, y1+geometry.hPatch/2-1,
				x1-geometry.wPatch/2, x1+geometry.wPatch/2-1)
      -- normalize distances in [0..1] (outsiderso won't be predictable)
      dataset.targets[iSample][1] = (x2-x1) / geometry.wPatch + 0.5
      dataset.targets[iSample][2] = (y2-y1) / geometry.hPatch + 0.5
      meandist[1] = meandist[1] + math.abs(x2-x1)
      meandist[2] = meandist[2] + math.abs(y2-y1)
      if (dataset.targets[iSample][1] < 0) or (dataset.targets[iSample][1] > 1) or
         (dataset.targets[iSample][2] < 0) or (dataset.targets[iSample][2] > 1)  then
	 print("Won't be able to predict that optical flow : too large " .. iSample .. " " ..
	       x2-x1 .. " " .. y2-y1)
      end
   end
   meandist = meandist/nSamples
   print('Mean optical flow in pixels (x, y):')
   print('(' .. meandist[1] .. ', ' .. meandist[2] .. ')')
   
   return dataset
end