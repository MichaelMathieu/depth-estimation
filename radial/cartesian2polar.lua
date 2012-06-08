require 'image'
require 'inline'

function getC2PMask(wsrc, hsrc, wdst, hdst, xcenter, ycenter, lpadding, rpadding, rmax)
   lpadding = lpadding or 0
   rpadding = rpadding or 0
   local padded = torch.FloatTensor(2, hdst, wdst+lpadding+rpadding)
   local mask = padded:sub(1,2, 1,hdst, 1+lpadding,wdst+lpadding)
   rmax = rmax or math.min(math.floor(hsrc/2), math.floor(wsrc/2)) - 1
   xcenter = xcenter or wsrc/2
   ycenter = ycenter or hsrc/2
   local kr = rmax/hdst
   local ktheta = 2*math.pi/wdst

   buildMask = inline.load [[
	 const void* idtensor = luaT_checktypename2id(L, "torch.FloatTensor");
	 THFloatTensor* mask = (THFloatTensor*)luaT_checkudata(L, 1, idtensor);
	 float xcenter = lua_tonumber(L, 2);
	 float ycenter = lua_tonumber(L, 3);
	 float kr = lua_tonumber(L, 4);
	 float ktheta = lua_tonumber(L, 5);
	 
	 int hdst = mask->size[1];
	 int wdst = mask->size[2];
	 long* ms = mask->stride;
	 float* mask_p = THFloatTensor_data(mask);
	 
	 int i, j;
	 float r, theta;
	 for (i = 0; i < hdst; ++i) {
	    for (j = 0; j < wdst; ++j) {
	       r = kr * (float)i;
	       theta = ktheta * (float)j;
	       mask_p[      + i*ms[1] + j*ms[2] ] = r * sin(theta) + ycenter;
	       mask_p[ms[0] + i*ms[1] + j*ms[2] ] = r * cos(theta) + xcenter;
	    }
	 }
   ]]
   buildMask(mask, xcenter, ycenter, kr, ktheta)
   if lpadding > 0 then
      padded:sub(1,2, 1,hdst, 1,lpadding):copy(mask:sub(1,2, 1,hdst, wdst-lpadding+1,wdst))
   end
   if rpadding > 0 then
      padded:sub(1,2, 1,hdst, wdst+1+lpadding,wdst+lpadding+rpadding):copy(mask:sub(1,2, 1,hdst, 1,rpadding))
   end
   return padded
end

function getP2CMask(wsrc, hsrc, wdst, hdst, xcenter, ycenter, rmax)
   local mask = torch.FloatTensor(2, hdst, wdst)
   rmax = rmax or math.min(math.floor(hdst/2), math.floor(wdst/2)) - 1
   xcenter = xcenter or wdst/2
   ycenter = ycenter or hdst/2
   local pi2 = 2*math.pi
   local kx = wsrc/pi2
   local ky = hsrc/rmax;

   buildMask = inline.load [[
	 const void* idtensor = luaT_checktypename2id(L, "torch.FloatTensor");
	 THFloatTensor* mask = (THFloatTensor*)luaT_checkudata(L, 1, idtensor);
	 float xcenter = lua_tonumber(L, 2);
	 float ycenter = lua_tonumber(L, 3);
	 float kx = lua_tonumber(L, 4);
	 float ky = lua_tonumber(L, 5);
	 float pi2 = lua_tonumber(L, 6);
	 
	 int hdst = mask->size[1];
	 int wdst = mask->size[2];
	 long* ms = mask->stride;
         float* mask_p = THFloatTensor_data(mask);
	 
	 int i, j;
	 float x, y;
	 for (i = 0; i < hdst; ++i) {
	    for (j = 0; j < wdst; ++j) {
	       x = (float)j - xcenter;
	       y = (float)i - ycenter;
	       mask_p[      + i*ms[1] + j*ms[2] ] = sqrt(x*x+y*y)*ky;
	       mask_p[ms[0] + i*ms[1] + j*ms[2] ] = fmod(atan2(y, x) + pi2, pi2)*kx;
	    }
	 }
   ]]
   buildMask(mask, xcenter, ycenter, kx, ky, pi2)
   return mask
end

function cartesian2polar(img, mask)
   return image.warp(img, mask, 'bilinear', false)
end

function cartesian2polar_testme()
   local lena = image.scale(image.lena(), 452, 231)
   --image.display(lena)
   local mask = getC2PMask(lena:size(3), lena:size(2), 400, 250)
   local im = torch.Tensor()
   image.warp(im, lena, mask, 'bilinear', false)
   --image.display(im)
   local mask2 = getP2CMask(400, 250, lena:size(3), lena:size(2))
   --image.display(cartesian2polar(im, mask2))
   image.display{image=lena-cartesian2polar(im, mask2), min=0, max=1}
end