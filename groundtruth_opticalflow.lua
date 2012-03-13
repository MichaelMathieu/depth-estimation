require 'torch'
require 'xlua'
require 'common'
require 'image'
require 'common'
require 'opticalflow_model'

function findMax(geometry, of)
   _, imax = of:reshape(of:size(1), of:size(2), of:size(3)*of:size(4)):max(3)
   xmax = torch.Tensor(imax:size(1), imax:size(2))
   ymax = torch.Tensor(imax:size(1), imax:size(2))
   for i = 1,imax:size(1) do
      for j = 1,imax:size(2) do
	 ymax[i][j] = math.floor((imax[i][j][1]-1)/geometry.maxw)+1
	 xmax[i][j] = math.mod(imax[i][j][1]-1, geometry.maxw)+1
      end
   end
   return ymax,xmax
end

function getOpticalFlow(geometry, image1, image2)
   of = torch.Tensor(image1:size(2)-geometry.hKernel-geometry.maxh+2,
		      image1:size(3)-geometry.wKernel-geometry.maxw+2,
		      geometry.maxh, geometry.maxw)
   for i = 1, image1:size(2) - geometry.hKernel - geometry.maxh + 2 do
      xlua.progress(i, image1:size(2) - geometry.hKernel - geometry.maxh + 2)
      for j = 1, image1:size(3) - geometry.wKernel - geometry.maxw + 2 do
	 local win = image1:sub(1, image1:size(1),
				i+math.floor(geometry.maxh/2),
				i+math.floor(geometry.maxh/2)+geometry.hKernel-1,
				j+math.floor(geometry.maxw/2),
				j+math.floor(geometry.maxw/2)+geometry.wKernel-1)
	 local tomul = image2:sub(1, image2:size(1),
				  i, i+geometry.maxh+geometry.hKernel-2,
				  j, j+geometry.maxw+geometry.wKernel-2)
	 local unfolded = tomul:unfold(2, geometry.hKernel, 1):unfold(3, geometry.wKernel, 1)
	 local norm2win = win:dot(win)
	 for k = 1, geometry.maxh do
	    for l = 1, geometry.maxw do
	       local win2 = unfolded:select(2,k):select(2,l)
	       of[i][j][k][l] = win:dot(win2)/(math.sqrt(norm2win*win2:dot(win2)))
	    end
	 end
      end
   end
   --'of' contains now the expected result of the newtork
   return findMax(geometry, of)
end

function loadDataOpticalFlow(geometry, dirbasename, nImgs, delta)
   nImgs = nImgs or 100
   local imagesdir = dirbasename .. 'images'
   local flowdir = dirbasename .. 'flow/' .. delta
   local findIm = 'cd ' .. imagesdir .. ' && ls -LB'
   local findFlow = 'mkdir -p ' .. flowdir .. ' && cd ' .. flowdir .. ' && ls -LB'
   raw_data = {}
   raw_data.images = {}
   raw_data.flow = {}
   local imagepaths = {}
   local flowpaths = {}
   local iLine = 0
   for line in io.popen(findIm):lines() do
      if math.mod(iLine, delta) == 0 then
	 table.insert(imagepaths, {line, imagesdir .. '/' .. line})
      end
      iLine = iLine + 1
   end
   for line in io.popen(findFlow):lines() do
      flowpaths[line] = flowdir .. '/' .. line
   end
   for i = 1,math.min(#imagepaths, nImgs) do
      table.insert(raw_data.images,
		   image.scale(image.loadJPG(imagepaths[i][2]), geometry.wImg, geometry.hImg))
      if i ~= 1 then
	 local found_flow = false
	 local flowname = imagepaths[i][1]:gsub('.jpg', '.png')
	 if flowpaths[flowname] then
	    local flow = image.loadPNG(flowpaths[flowname])
	    if (flow:size(2) ~= geometry.hImg - geometry.hKernel - geometry.maxh + 2) or
	       (flow:size(3) ~= geometry.wImg - geometry.wKernel - geometry.maxw + 2) then
	       print(flow:size(2) .. " " .. geometry.hImg)
	       print(flow:size(3) .. " " .. geometry.wImg)
	       print('flow in file ' .. flowname .. ' has wrong size. Recomputing.')
	       local ibackup = 1
	       local backupname = flowname .. '_backup' .. ibackup
	       while flowpaths[backupname] ~= nil do
		  ibackup = ibackup + 1
		  backupname = flowname .. '_backup' .. ibackup
	       end
	       print('Backing up the old file (' .. backupname .. ')')
	       io.popen('mv ' .. flowpaths[flowname] .. ' ' .. flowdir .. '/' .. backupname)
	    else
	       found_flow = true
	       table.insert(raw_data.flow, (flow:narrow(1, 1, 2)*255+0.5):floor())
	    end
	 end
	 if not found_flow then
	    print('Computing groundtruth optical flow for image ' .. imagepaths[i][1])
	    local yflow, xflow = getOpticalFlow(geometry, raw_data.images[#raw_data.images-1],
						raw_data.images[#raw_data.images])
	    local flow = torch.Tensor(3, xflow:size(1), xflow:size(2)):fill(1)
	    flow[1]:copy(xflow/255)
	    flow[2]:copy(yflow/255)
	    image.savePNG(flowdir .. '/' .. flowname, flow)
	    table.insert(raw_data.flow, (flow:narrow(1, 1, 2)*255+0.5):floor())
	 end
      end
   end

   raw_data.histogram = {}
   for i = 1,geometry.maxh do
      raw_data.histogram[i] = {}
      for j = 1,geometry.maxw do
	 raw_data.histogram[i][j] = {}
      end
   end
   for iImg = 1,#raw_data.flow do
      for i = 1,raw_data.flow[iImg]:size(2) do
	 for j = 1,raw_data.flow[iImg]:size(3) do
	    table.insert(raw_data.histogram[raw_data.flow[iImg][2][i][j]][raw_data.flow[iImg][1][i][j]], {iImg+1, i, j})
	 end
      end
   end

   return raw_data
end

function generateDataOpticalFlow(geometry, raw_data, nSamples)
   local dataset = {}
   dataset.patches = torch.Tensor(nSamples, 2, raw_data.images[1]:size(1),
				  geometry.hPatch2, geometry.wPatch2)
   dataset.targets = torch.Tensor(nSamples, 2)
   function dataset:size()
      return nSamples
   end
   setmetatable(dataset, {__index = function(self, index)
				       return {self.patches[index], self.targets[index]}
				    end})

   local iFlow = 1
   local iSample = 1
   while iSample <= nSamples do
      modProgress(iSample, nSamples, 100)
      local yFlow, xFlow = x2yx(geometry, iFlow)
      local candidates = raw_data.histogram[yFlow][xFlow]
      if #candidates ~= 0 then
	 local iCandidate = randInt(1, #candidates+1)
	 local iImg = candidates[iCandidate][1]
	 local yPatch = candidates[iCandidate][2]
	 local xPatch = candidates[iCandidate][3]

	 local patch1 = raw_data.images[iImg-1]:sub(1, raw_data.images[iImg-1]:size(1),
						    yPatch, yPatch+geometry.hPatch2-1,
						    xPatch, xPatch+geometry.wPatch2-1)
	 local patch2 = raw_data.images[iImg]:sub(1, raw_data.images[iImg]:size(1),
						  yPatch, yPatch+geometry.hPatch2-1,
						  xPatch, xPatch+geometry.wPatch2-1)
	 dataset.patches[iSample][1] = patch1
	 dataset.patches[iSample][2] = patch2
	 dataset.targets[iSample][1] = yFlow
	 dataset.targets[iSample][2] = xFlow
	 iSample = iSample + 1
      end
      iFlow = iFlow + 1
      if iFlow > geometry.maxh*geometry.maxw then
	 iFlow = 1
      end
   end
   return dataset
end




--[[
local geometry = {}
geometry.wImg = 320
geometry.hImg = 180
geometry.wPatch2 = 32
geometry.hPatch2 = 32
geometry.wKernel = 16
geometry.hKernel = 16
geometry.maxw = geometry.wPatch2 - geometry.wKernel + 1
geometry.maxh = geometry.hPatch2 - geometry.hKernel + 1
geometry.wPatch1 = geometry.wPatch2 - geometry.maxw + 1
geometry.hPatch1 = geometry.hPatch2 - geometry.maxh + 1
geometry.nChannelsIn = 3
geometry.nFeatures = 10

raw_data = loadDataOpticalFlow(geometry, 'data/', 10, 2)
for i = 1,17 do
   print(raw_data.histogram[i])
end
--]]