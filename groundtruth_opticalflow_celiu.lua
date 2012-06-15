require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
require 'image'
require 'common'
require 'liuflow'
require 'xlua'

function getFlow(im1, im2)
   local alpha = 0.005
   local ratio = 0.75
   local minWidth = 30
   local nOuterFPIterations = 5
   local nInnerFPIterations = 1
   local SORIterations = 40
   local resn, resa, warp, resx, resy = liuflow.infer{pair={im1, im2}, alpha=alpha,
						      ratio=ratio,
						      minWidth=minWidth,
						      nOuterFPIterations=nOuterFPIterations,
						      nInnerFPIterations=nInnerFPIterations,
						      nCGIterations=SORIterations}
   win3=image.display{image=liuflow.field2rgb(resn, resa), win=win3, min=0, max=1}
   return resx, resy
end

local use_rectified = true
local delta = 1
local use_full_image = false
local imdir = '/home/nyu/robot/depth-estimation/data/no-risk/part3/'
local w = 320

local lst = ls2(imdir..'images', function (a) return true end)
nImg = #lst

for i = 30,nImg do
   xlua.progress(i, nImg)
   if use_rectified then
      im1 = image.load(string.format("%srectified_images/%09d.jpg", imdir, i))
   else
      im1 = image.load(string.format("%sundistorted_images/%09d.jpg", imdir, i))
   end
   im2 = image.load(string.format("%sundistorted_images/%09d.jpg", imdir, i+delta))
   
   
   local h = im1:size(2)*w/im1:size(3)
   if not use_full_image then
      im1 = image.scale(im1, w, h)
      im2 = image.scale(im2, w, h)
   end

   wim2=image.display{image={im1-im2, im1, im2}, win=wim2, min=0, max=1}

   local vx, vy = getFlow(im1, im2)
   if use_full_image then
      vx = image.scale(vx, w, h, 'bilinear')
      vy = image.scale(vy, w, h, 'bilinear')
      vx = w/im1:size(3)*vx
      vy = h/im1:size(2)*vy
      vx = (vx+0.5):floor()
      vy = (vy+0.5):floor()
   end

   win=image.display{image={vx, vy}, win=win, min=-8, max=8}

   local flow = torch.Tensor(3, h, w)
   flow[1]:copy((vy+128)/255)
   flow[2]:copy((vx+128)/255)
   flow[3]:zero()

   local output
   if use_rectified then
      output = string.format('%srectified_flow2/%dx%d/celiu/%d/', imdir, w, h, delta)
   else
      output = string.format('%sflow/%dx%d/celiu/%d/', imdir, w, h, delta)
   end
   sys.execute('mkdir -p '..output)
   image.save(string.format("%s%09d.png", output, i), flow)
end