require 'image'
require 'paths'

ImageLoader = {}

function ImageLoader:init(geometry, path, fst, delta)
   self.geometry = geometry
   self.nextFrame = fst
   self.delta = delta
   self.path = path
   if self.path:sub(-1) ~= '/' then
      self.path = self.path .. '/'
   end
end

function ImageLoader:getNextFrame()
   local impath = string.format("%s%09d.jpg", self.path, self.nextFrame)
   if not paths.filep(impath) then
      return nil
   end
   local im = image.scale(image.load(impath), self.geometry.wImg, self.geometry.hImg)
   self.nextFrame = self.nextFrame + 1
   return im
end