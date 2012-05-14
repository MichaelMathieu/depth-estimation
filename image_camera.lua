require 'torch'
require 'sys'
require 'image'
require 'paths'
--useOpenCV = true
require 'camera'

ImageCamera = {}

function ImageCamera:init(geometry, idx)
   self.geometry = geometry
   self.camera = image.Camera{idx = idx, width = 640, height = 480, nbuffers=1}
   self.camera:setTVStandard('NTSC')
end

function ImageCamera:getNextFrame()
   --self.camera:forward()
   --self.camera:forward()
   --self.camera:forward()
   self.camera:forward()
   local frame = self.camera:forward()
   local target_h = self.geometry.hImg*frame:size(3)/self.geometry.wImg
   local diff_h = frame:size(2) - target_h
   frame = frame:narrow(2, diff_h/2, target_h)
   --return image.scale(frame, self.geometry.wImg, self.geometry.hImg, 'simple')
   return frame
end