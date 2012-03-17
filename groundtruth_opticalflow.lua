require 'torch'
require 'paths'
require 'xlua'
require 'common'
require 'image'
require 'common'
require 'opticalflow_model'

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

--[[
function getOpticalFlowFast(geometry, image1, image2)
   local halfmaxh = math.ceil(geometry.maxh/2)-1
   local halfmaxw = math.ceil(geometry.maxw/2)-1
   local halfhKernel = math.ceil(geometry.hKernel/2)-1
   local halfwKernel = math.ceil(geometry.wKernel/2)-1
   local matching = nn.SpatialMatching(geometry.maxh, geometry.maxw, true)
   local unfolded1 = image1:unfold(2, geometry.hKernel, 1):unfold(3, geometry.wKernel, 1)
   unfolded1:resize(unfolded1:size(1), unfolded1:size(2), unfolded1:size(3),
		    unfolded1:size(4)*unfolded1:size(5))
   local input1 = Torch.tensor(unfolded1:size(1)*unfolded1:size(4),
			       unfolded1:size(2), unfolded1:size(3))
   for i = 1,unfolded1:size(2) do
      for j = 1,unfolded1:size(3) do
	 for k = 1,unfolded1:size(1) do
	    input1:sub(select(2,i):select(2,j) = unfolded1:
   local unfolded2 = image2:unfold(2, geometry.hKernel, 1):unfold(3, geometry.wKernel, 1)
   unfolded2:resize(unfolded2:size(1), unfolded2:size(2), unfolded2:size(3),
		    unfolded2:size(4)*unfolded2:size(5))
   local input2 = Torch.tensor(unfolded2:size(1)*unfolded2:size(4),
			       unfolded2:size(2), unfolded2:size(3))

   of = torch.Tensor(geometry.maxh, geometry.maxw, image1:size(2), image1:size(3)):fill(-1)
   for i = 1, image1:size(2) - geometry.hKernel - geometry.maxh + 2 do
      xlua.progress(i, image1:size(2) - geometry.hKernel - geometry.maxh + 2)
      for j = 1, image1:size(3) - geometry.wKernel - geometry.maxw + 2 do
	 local win = image1:sub(1, image1:size(1),
				i+halfmaxh, i+halfmaxh+geometry.hKernel-1,
				j+halfmaxw, j+halfmaxw+geometry.wKernel-1)
	 local tomul = image2:sub(1, image2:size(1),
				  i, i+geometry.maxh+geometry.hKernel-2,
				  j, j+geometry.maxw+geometry.wKernel-2)
	 local unfolded = tomul:unfold(2, geometry.hKernel, 1):unfold(3, geometry.wKernel, 1)
	 local norm2win = win:dot(win)
	 for k = 1, geometry.maxh do
	    for l = 1, geometry.maxw do
	       local win2 = unfolded:select(2,k):select(2,l)
	       of[k][l][i+halfmaxh+halfhKernel][j+halfmaxw+halfwKernel] = win:dot(win2)/(math.sqrt(norm2win*win2:dot(win2)))
	    end
	 end
      end
   end
   --'of' contains now the expected result of the newtork
   return findMax(geometry, of)
end
--]]

function getOpticalFlow(geometry, image1, image2)
   local halfmaxh = math.ceil(geometry.maxh/2)-1
   local halfmaxw = math.ceil(geometry.maxw/2)-1
   local halfhKernel = math.ceil(geometry.hKernel/2)-1
   local halfwKernel = math.ceil(geometry.wKernel/2)-1
   of = torch.Tensor(geometry.maxh, geometry.maxw, image1:size(2), image1:size(3)):fill(-1)
   for i = 1, image1:size(2) - geometry.hKernel - geometry.maxh + 2 do
      xlua.progress(i, image1:size(2) - geometry.hKernel - geometry.maxh + 2)
      for j = 1, image1:size(3) - geometry.wKernel - geometry.maxw + 2 do
	 local win = image1:sub(1, image1:size(1),
				i+halfmaxh, i+halfmaxh+geometry.hKernel-1,
				j+halfmaxw, j+halfmaxw+geometry.wKernel-1)
	 local tomul = image2:sub(1, image2:size(1),
				  i, i+geometry.maxh+geometry.hKernel-2,
				  j, j+geometry.maxw+geometry.wKernel-2)
	 local unfolded = tomul:unfold(2, geometry.hKernel, 1):unfold(3, geometry.wKernel, 1)
	 local norm2win = win:dot(win)
	 for k = 1, geometry.maxh do
	    for l = 1, geometry.maxw do
	       local win2 = unfolded:select(2,k):select(2,l)
	       of[k][l][i+halfmaxh+halfhKernel][j+halfmaxw+halfwKernel] = win:dot(win2)/(math.sqrt(norm2win*win2:dot(win2)))
	    end
	 end
      end
   end
   --'of' contains now the expected result of the newtork
   return findMax(geometry, of)
end

function loadImageOpticalFlow(geometry, dirbasename, imagebasename, previmagebasename, delta)
   local imagepath = dirbasename .. 'images/' .. imagebasename .. '.jpg'
   if not paths.filep(imagepath) then
      print("Image " .. imagepath .. " not found.")
      return nil
   end
   local im = image.scale(image.loadJPG(imagepath), geometry.wImg, geometry.hImg)
   if not previmagebasename then
      return im
   end
   local flowdir = dirbasename .. 'flow/' .. geometry.wImg .. 'x' .. geometry.hImg
   flowdir = flowdir .. '/' .. geometry.maxh .. 'x' ..geometry.maxw .. 'x'
   flowdir = flowdir .. geometry.hKernel .. 'x' ..geometry.wKernel .. '/' .. delta
   os.execute('mkdir -p ' .. flowdir)
   local flowfilename = flowdir .. '/' .. imagebasename .. '.png'
   local flow = nil
   if paths.filep(flowfilename) then
      flow = image.loadPNG(flowfilename)
      if (flow:size(2) ~= geometry.hImg) or (flow:size(3) ~= geometry.wImg) then
	 flow = nil
	 print("Flow in file " .. flowfilename .. " has wrong size. Recomputing...")
      end
   end
   if not flow then
      local previmagepath = dirbasename .. 'images/' .. previmagebasename .. '.jpg'
      print('Computing groundtruth optical flow for images '..imagepath..' and '..previmagepath)
         if not paths.filep(previmagepath) then
	    print("Image " .. previmagepath .. " not found.")
	    return nil
	 end
      local previmage = image.scale(image.loadJPG(previmagepath), geometry.wImg, geometry.hImg)
      local yflow, xflow = getOpticalFlow(geometry, previmage, im)
      flow = torch.Tensor(3, xflow:size(1), xflow:size(2)):fill(1)
      flow[1]:copy(xflow/255)
      flow[2]:copy(yflow/255)
      image.savePNG(flowfilename, flow)
   end
   flow = (flow:narrow(1, 1, 2)*255+0.5):floor()
   return im, flow
end

function loadDataOpticalFlow(geometry, dirbasename, nImgs, first_image, delta)
   local imagesdir = dirbasename .. 'images'
   local findIm = 'cd ' .. imagesdir .. ' && ls -LB'
   raw_data = {}
   raw_data.images = {}
   raw_data.flow = {}
   local imagepaths_raw = {}
   local flowpaths = {}
   for line in io.popen(findIm):lines() do
      local linebase,_ = line:gsub('.jpg', '')
      if linebase .. '.jpg' == line then
	 table.insert(imagepaths_raw, linebase)
      end
   end
   local imagepaths = {}
   local iLine = first_image+1 --images are numbered from 0
   for i = 1,nImgs do
      imagepaths[i] = imagepaths_raw[iLine]
      iLine = iLine + delta
   end

   local im = loadImageOpticalFlow(geometry, dirbasename, imagepaths[1], nil, nil)
   table.insert(raw_data.images, im)
   for i = 2,math.min(#imagepaths, nImgs) do
      local im, flow = loadImageOpticalFlow(geometry, dirbasename, imagepaths[i],
					    imagepaths[i-1], delta)
      table.insert(raw_data.images, im)
      table.insert(raw_data.flow, flow)
   end

   local hoffset = math.ceil(geometry.maxh/2) + math.ceil(geometry.hKernel/2) - 2
   local woffset = math.ceil(geometry.maxw/2) + math.ceil(geometry.wKernel/2) - 2
   raw_data.histogram = {}
   for i = 1,geometry.maxh do
      raw_data.histogram[i] = {}
      for j = 1,geometry.maxw do
	 raw_data.histogram[i][j] = {}
      end
   end
   for iImg = 1,#raw_data.flow do
      for i = 1,geometry.hImg-geometry.hPatch2 do
	 for j = 1,geometry.wImg-geometry.wPatch2 do
	    table.insert(raw_data.histogram[raw_data.flow[iImg][1][i+hoffset][j+woffset]][raw_data.flow[iImg][2][i+hoffset][j+woffset]], {iImg+1, i, j})
	 end
      end
   end

   return raw_data
end

function generateDataOpticalFlow(geometry, raw_data, nSamples, method)
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

   if method == 'uniform_flow' then
      local iFlow = 1
      local iSample = 1
      local thres_n_candidates = (geometry.hImg-geometry.hPatch2)*(geometry.wImg-geometry.wPatch2) * #raw_data.flow / 20
      while iSample <= nSamples do
	 modProgress(iSample, nSamples, 100)
	 local yFlow, xFlow = x2yx(geometry, iFlow)
	 local candidates = raw_data.histogram[yFlow][xFlow]
	 if #candidates > thres_n_candidates then
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
   elseif method == 'uniform_position' then
      local hoffset = math.ceil(geometry.maxh/2) + math.ceil(geometry.hKernel/2) - 2
      local woffset = math.ceil(geometry.maxw/2) + math.ceil(geometry.wKernel/2) - 2
      for iSample = 1,nSamples do
	 modProgress(iSample, nSamples, 100)
	 local iImg = randInt(2, #raw_data.images+1)
	 local yPatch = randInt(1, geometry.hImg-geometry.hPatch2-1)
	 local xPatch = randInt(1, geometry.wImg-geometry.wPatch2-1)
	 local yFlow = raw_data.flow[iImg-1][1][yPatch+hoffset][xPatch+woffset]
	 local xFlow = raw_data.flow[iImg-1][2][yPatch+hoffset][xPatch+woffset]

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
      end
   else
      assert(false)
   end

   return dataset
end



--[[
local geometry = {}
geometry.wImg = 32
geometry.hImg = 32
geometry.wPatch2 = 6
geometry.hPatch2 = 6
geometry.wKernel = 1
geometry.hKernel = 1
geometry.maxw = geometry.wPatch2 - geometry.wKernel + 1
geometry.maxh = geometry.hPatch2 - geometry.hKernel + 1
geometry.wPatch1 = geometry.wPatch2 - geometry.maxw + 1
geometry.hPatch1 = geometry.hPatch2 - geometry.maxh + 1
geometry.nChannelsIn = 3
geometry.nFeatures = 10
geometry.soft_target = true
--]]
--[[
raw_data = loadDataOpticalFlow(geometry, 'data/', 2, 0, 2)
im1 = raw_data.images[1]
im2 = raw_data.images[2]
flow = raw_data.flow[1]
--]]

--[[
im1 = torch.randn(3, 16, 16)
im2 = torch.Tensor(im1:size())
im2:sub(1,3,1,10,1,16):copy(im1:sub(1,3,1,10,1,16))
im2:sub(1,3,11,16,1,5):copy(im1:sub(1,3,10,15,1,5))
im2:sub(1,3,11,16,6,16):copy(im1:sub(1,3,10,15,4,14))
--]]
--[[
im1 = torch.randn(3, 32, 32)
im1:zero()
for i = 1,32 do
   for j = 1,32 do
      im1[1][i][j] = i
      im1[2][i][j] = j
      im1[3][i][j] = 1
   end
end
im2 = torch.Tensor(im1:size())
im2:sub(1,3,1,10,1,32):copy(im1:sub(1,3,1,10,1,32))
im2:sub(1,3,11,32,1,10):copy(im1:sub(1,3,10,31,1,10))
im2:sub(1,3,11,32,11,32):copy(im1:sub(1,3,10,31,9,30))
--]]
--[[
im1 = torch.randn(3, 64, 64)
im2 = torch.Tensor(im1:size())
im2:sub(1,3,1,20,1,61):copy(im1:sub(1,3,2,21,3,63))
im2:sub(1,3,1,20,62,64):copy(im1:sub(1,3,2,21,62,64))
im2:sub(1,3,21,64,1,32):copy(im1:sub(1,3,21,64,1,32))
im2:sub(1,3,21,64,33,64):copy(im1:sub(1,3,21,64,30,61))
--]]
--[[
image.display{im1, im2}
yflow,xflow = getOpticalFlow(geometry, im1, im2)
flow = torch.Tensor(2, yflow:size(1), yflow:size(2))
flow[1]:copy(yflow)
flow[2]:copy(xflow)

require 'opticalflow_model'
input = prepareInput(geometry, im1, im2)
model = getModel(geometry, true)
flow_model = processOutput(geometry, model:forward(input)).full
flow = flow_model

print(flow)
image.display(flow)
a = torch.Tensor(im1:size()):fill(0)
for i = 1,im1:size(2) do
   for j = 1,im1:size(3) do
      local y = flow[1][i][j]
      local x = flow[2][i][j]
      if (x ~= 0) and (y ~= 0) then
	 y, x = onebased2centered(geometry, y, x)
	 for k = 1,im1:size(1) do
	    a[k][i][j] = im1[k][i][j] - im2[k][i+y][j+x]
	 end
      end
   end
end

--image.display(a)
image.display(image.scale(a, 128, 128, 'simple'))
--]]