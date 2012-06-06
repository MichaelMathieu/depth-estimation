require 'image'

function getC2PMask(wsrc, hsrc, wdst, hdst)
   local mask = torch.Tensor(2, hdst, wdst)
   --TODO inliner
   local rmax = math.min(math.floor(hsrc/2), math.floor(wsrc/2)) - 1
   local xcenter = wsrc/2
   local ycenter = hsrc/2
   for i = 1,hdst do
      for j = 1,wdst do
	 local r = i*rmax/hdst
	 local theta = j/wdst*2*math.pi
	 mask[1][i][j] = r * math.sin(theta) + ycenter + 1
	 mask[2][i][j] = r * math.cos(theta) + xcenter + 1
      end
   end
   return mask
end

function cartesian2polar(img, mask)
   return image.warp(img, mask, 'bilinear', false)
end

function cartesian2polar_testme()
   local mask = getC2PMask(512, 512, 200, 200)
   print(mask:min())
   print(mask:max())
   image.display(image.lena())
   local im = torch.Tensor()
   image.warp(im, image.lena(), mask, 'bilinear', false)
   image.display(im)
end

cartesian2polar_testme()