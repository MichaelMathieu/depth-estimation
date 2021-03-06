require 'image'
require 'paths'
require 'groundtruth_opticalflow'

ImageLoader = {}

function ImageLoader:init(geometry, path, fst, delta)
   self.geometry = geometry
   self.nextFrame = fst-1
   self.delta = delta
   self.path = path
   if self.path:sub(-1) ~= '/' then
      self.path = self.path .. '/'
   end
end

function ImageLoader:getNextFrame()
   self.nextFrame = self.nextFrame + 1
   print(self.nextFrame)
   local impath = string.format("%simages/%09d.jpg", self.path, self.nextFrame)
   if not paths.filep(impath) then
      impath = string.format("%simages/%09d.png", self.path, self.nextFrame)
      if not paths.filep(impath) then
	 return nil
      end
   end
   return image.load(impath)
   --local im = image.scale(image.load(impath), self.geometry.wImg, self.geometry.hImg)
   --return im
end

function ImageLoader:getCurrentGT()
   if self.nextFrame >= 1 then
      local pimpath = string.format("%09d",self.nextFrame-1)
      local impath = string.format("%09d", self.nextFrame)
      local _, gt = loadImageOpticalFlow(self.geometry, self.path, impath, pimpath, 1,
					 'liu')
					 --'cross-correlation')
      return gt
   else
      return torch.Tensor(2, self.geometry.wImg, self.geometry.hImg)
   end
end