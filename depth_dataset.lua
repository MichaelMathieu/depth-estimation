require 'torch'
require 'paths'
require 'image'
require 'sys'
require 'xlua'

raw_data = {}
w_imgs = 640
h_imgs = 360
patchesPerClass = {}
nClasses = 2

function loadImage(filebasename)
   local imfilename = 'data/images/' .. filebasename .. '.jpg'
   local depthfilename = 'data/depths/' .. filebasename .. '.mat'
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
   assert(nClasses == 2)
   if depth < 7 then
      return 1
   else
      return 2
   end
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

function generateData(nSamples, w, h, is_train, use_2_pics)
   dataset = {}
   if use_2_pics then
      dataset.patches = torch.Tensor(nSamples, 2, h, w)
   else
      dataset.patches = torch.Tensor(nSamples, 1, h, w)
   end
   dataset.targets = torch.Tensor(nSamples, nClasses):zero()
   setmetatable(dataset, {__index = function(self, index)
				       return {self.patches[index], self.targets[index]}
				    end})
   function dataset:size()
      return nSamples
   end

   for i = 1,nClasses do
      assert(table.getn(patchesPerClass[i]) > nSamples/nClasses)
   end

   local nGood = 1
   local dbg = torch.Tensor(2):zero()
   local dbg2 = torch.Tensor(2):zero()
   while nGood <= nSamples do
      local class = randInt(1, nClasses+1)
      dbg[class] = dbg[class] + 1
      local iPatch = randInt(1, table.getn(patchesPerClass[class])+1)
      local patch = patchesPerClass[class][iPatch]
      local im = patch[1]
      local y = patch[2]
      local x = patch[3]
      if (y-h/2 >= 1) and (y+h/2-1 <= h_imgs) and (x-w/2 >= 1) and (x+w/2-1 <= w_imgs) then
	 dbg2[class] = dbg2[class] + 1
	 local patch = image.rgb2y(raw_data[im][1]:sub(1, 3, y-h/2, y+h/2-1, x-w/2, x+w/2-1))
	 dataset.patches[nGood][1]:copy(patch)
	 if use_2_pics then
	    local patch2 = image.rgb2y(raw_data[im+1][1]:sub(1, 3, y-h/2, y+h/2-1, x-w/2, x+w/2-1))
	    dataset.patches[nGood][2]:copy(patch2)
	 end
	 dataset.targets[nGood][class] = 1
	 nGood = nGood + 1
      end
   end
   
   return dataset
end   

function loadData(nImgs, delta)
   print("Loading images")
   for i = 1,nImgs do
      xlua.progress(i, nImgs)
      loadImage(string.format("%09d", i*delta+1010))
   end
   print("Pre-sorting images")
   preSortData()
end
--generateData(1000, 32, 32, true, true)