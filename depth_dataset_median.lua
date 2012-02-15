require 'torch'
require 'paths'
require 'image'
require 'sys'
require 'xlua'

raw_data = {}
w_imgs = 640
h_imgs = 360
patchesPerClass = {}
patchesMedianDepth = {}
nClasses = 2

function loadImage(filebasename)
   local imfilename = 'data/images/' .. filebasename .. '.jpg'
   local depthfilename = 'data/depths/' .. filebasename .. '.mat'
   --local imfilename = filebasename .. '.jpg'
   --local depthfilename = filebasename .. '.mat'
   assert(paths.filep(imfilename))
   assert(paths.filep(depthfilename))
   local im = image.loadJPG(imfilename)
   local h_im = im:size(2)
   local w_im = im:size(3)
   im = image.scale(im, w_imgs, h_imgs)
   local file_depth = torch.DiskFile(depthfilename, 'r')
   local nPts = file_depth:readInt()
   local depthPoints = torch.Tensor(nPts, 3)
   for i = 1,nPts do
      depthPoints[i][1] = file_depth:readInt() * h_imgs / h_im
      depthPoints[i][2] = file_depth:readInt() * w_imgs / w_im
      depthPoints[i][3] = file_depth:readDouble()
   end
   table.insert(raw_data, {im, depthPoints})
   
end

function getClass(depth)
   local maxDepth = 16
   
   local step = maxDepth/nClasses
   local class = math.ceil(depth/step)
   
   if class>nClasses then return nClases end
   return class
end

function preSortData()
   for iClass = 1,nClasses do
      patchesPerClass[iClass] = {}
   end

	for iImg = 1,(table.getn(raw_data)-1) do
		xlua.progress(iImg, (table.getn(raw_data)-1))
      local im = raw_data[iImg][1]
      local patches = raw_data[iImg][2]
      for iPatch = 1,patches:size(1) do
			local y = patches[iPatch][1]
			local x = patches[iPatch][2]
			local depth = patches[iPatch][3]
			table.insert(patchesPerClass[getClass(depth)], {iImg, y, x})
      end
   end
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

  table.sort( temp )

  -- If we have an even number of table elements or odd.
  if math.fmod(#temp,2) == 0 then
    -- return mean value of middle two elements
    return ( temp[#temp/2] + temp[(#temp/2)+1] ) / 2
  else
    -- return middle element
    return temp[math.ceil(#temp/2)]
  end
end

function generateData(nSamples, w, h, is_train, use_2_pics)
   dataset = {}
   if use_2_pics then
      dataset.patches = torch.Tensor(nSamples, 2, h, w)
   else
		print("Using one pic")
      dataset.patches = torch.Tensor(nSamples, 1, h, w)
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
   
	--[[
   for i = 1,nClasses do
      assert(table.getn(patchesPerClass[i]) > nSamples/nClasses)
   end
   --]]
   
   
   
   print("Calculating patches median depth...")
   local currentPatchPts = {}
   patches = raw_data[1][2]
   local numberOfPatches = patches:size(1)
   --local numberOfPatches = 2000

   y,sorti = torch.sort(patches, 1)
   
   local numberOfBins = math.ceil(y[numberOfPatches][3])
   for iBin = 1,numberOfBins do
      patchesMedianDepth[iBin] = {}
   end

   local firstIndex = true
   local lastPatchIndex = 1
   for origi = 1,numberOfPatches do
		i = sorti[origi][2]
      xlua.progress(i, numberOfPatches)
		local yo = patches[i][1]
		local xo = patches[i][2]
		if (yo-h/2 >= 1) and (yo+h/2-1 <= h_imgs) and (xo-w/2 >= 1) and (xo+w/2-1 <= w_imgs) then
			for origj = lastPatchIndex,numberOfPatches do
            j = sorti[origj][2]
            local x = patches[j][2]
            if x>xo+w/2 then
               firstIndex = true
               break
            end
				if x>=xo-w/2 then
               if firstIndex then
                  lastPatchIndex = j-1
                  firstIndex = false
               end
               local y = patches[j][1]
   				
   				if (y>=yo-h/2+1) and (y<=yo+h/2) then
   					local depth = patches[j][3]
   					table.insert(currentPatchPts, depth)
   					
   				end
            end
			end
			
			local currentPatchMedianDepth = median(currentPatchPts)
			local binIndex = math.ceil(currentPatchMedianDepth)
			table.insert(patchesMedianDepth[binIndex], {currentPatchMedianDepth, yo, xo})
			
			for k in pairs(currentPatchPts) do currentPatchPts[k]=nil end
			
		end
   end
   
   print("Sampling patches...")
   nGood = 1
   while nGood <= nSamples do
		local randomBinIndex = randInt(1,numberOfBins)
		local sizeOfBin = table.getn(patchesMedianDepth[randomBinIndex])
		if sizeOfBin>0 then
			local randomPatchIndex = randInt(1, sizeOfBin)
			local y = math.ceil(patchesMedianDepth[randomBinIndex][randomPatchIndex][2])
			local x = math.ceil(patchesMedianDepth[randomBinIndex][randomPatchIndex][3])
			local patch = image.rgb2y(raw_data[1][1]:sub(1, 3, y-h/2+1, y+h/2, x-w/2+1, x+w/2))
			dataset.patches[nGood][1]:copy(patch)
			if use_2_pics then
				local patch2 = image.rgb2y(raw_data[2][1]:sub(1, 3, y-h/2, y+h/2-1, x-w/2, x+w/2-1))
				dataset.patches[nGood][2]:copy(patch2)
			end
			local class = getClass(patchesMedianDepth[randomBinIndex][randomPatchIndex][1])
			dataset.targets[nGood][class] = 1
			nGood = nGood + 1
		end
   end
   
   print("Done")
   
   return dataset
end   

function loadData(nImgs, delta)
   print("Loading images")
   for i = 0,nImgs-1 do
      xlua.progress(i+1, nImgs)
      loadImage(string.format("%09d", i*delta))
   end
   --print("Pre-sorting images")
	--preSortData()
end

--loadData(2,1)
--generateData(1000, 32, 32, true, true)
