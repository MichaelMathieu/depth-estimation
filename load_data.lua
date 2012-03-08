require 'torch'
require 'paths'
require 'sys'
require 'xlua'
require 'image'

function loadCameras(dirbasename)
   local filename = dirbasename .. 'depths/cameras'
   if not paths.filep(filename) then
      print('File' .. filename .. ' not found. Can\'t read camera positions.')
      return nil
   end
   local file = torch.DiskFile(filename, 'r')
   local version = file:readString('*l')
   if version ~= 'cameras version 1' then
      print('File ' .. filename .. ': wrong version')
      return nil
   end
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
   local version = file_depth:readString('*l')
   if version ~= 'depths version 2' then
      print('File ' .. depthfilename .. ': wrong version. Skipping...')
      return
   end
   local nPts = file_depth:readInt()
   local depthPoints = torch.Tensor(nPts, 4)
   for i = 1,nPts do
      -- todo : indices are wrong by one because of the indexing from 1
      depthPoints[i][4] = file_depth:readInt()
      depthPoints[i][1] = file_depth:readInt() * h_imgs / h_im
      depthPoints[i][2] = file_depth:readInt() * w_imgs / w_im
      depthPoints[i][3] = file_depth:readDouble()
   end
   file_depth:close()
   return {im, depthPoints}
end

function loadData(nImgs, delta, root_dir)
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
end