require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
package.path = "./?.lua;../?.lua;" .. package.path
package.cpath = "./?.so;../?.so;" .. package.cpath
require 'xlua'
require 'sys'
require 'openmp'
require 'image'
require 'radial_opticalflow_data'
require 'radial_opticalflow_network'
require 'radial_opticalflow_filtering'
require 'radial_opticalflow_display'
require 'draw'

torch.manualSeed(1)

op = xlua.OptionParser('%prog [options]')
-- general
op:option{'-nt', '--num-threads', action='store', dest='nThreads', default=2,
	  help='Number of threads used'}

-- input
op:option{'-rd', '--root-directory', action='store', dest='root_directory',
	  default='data/no-risk/part1/', help='Root dataset directory'}
op:option{'-fi', '--first-image', action='store', dest='first_image', default=0,
	  help='Index of first image used'}
op:option{'-d', '--delta', action='store', dest='delta', default=1,
	  help='Delta between two consecutive frames'}
op:option{'-ni', '--num-input-images', action='store', dest='num_input_images',
	  default=10, help='Number of annotated images used'}
op:option{'-cal', '--caligration', dest='calibration_file', default='rectified_gopro.cal',
	  action='store', help='Calibration parameters file'}
op:option{'-i', '--network', action='store', dest='network_file',
	  default=nil, help='Path to the saved network'}

opt = op:parse()
opt.nTherads = tonumber(opt.nThreads)
opt.first_image = tonumber(opt.first_image)
opt.delta = tonumber(opt.delta)
opt.num_input_images = tonumber(opt.num_input_images)
if opt.root_directory:sub(-1) ~= '/' then opt.root_directory = opt.root_directory .. '/' end

openmp.setDefaultNumThreads(opt.nThreads)

local network, networkp = loadTesterNetwork(opt.network_file)

local alpha_polar = 1.

local calibrationp = torch.load(opt.calibration_file)
calibrationp.sfm.max_points = 1000
calibrationp.sfm.points_quality = 0.0001
calibrationp.sfm.ransac_max_dist = .2

local datap = {}
datap.first_image = opt.first_image
datap.delta = opt.delta
datap.n_images = opt.num_input_images

local function rescale(im, w, h, mode)
   --if im:size(2) > w*2 then
   --im = image.scale(im, w*2, h*2, mode)
   --end 
   mode = mode or 'bilinear'
   if (im:size(2) ~= h) or (im:size(3) ~= w) then
      return image.scale(im, w, h, mode)
   else
      return im
   end
end

local Ksmall = calibrationp.K:clone()
Ksmall[1]:mul(networkp.wImg/calibrationp.wImg)
Ksmall[2]:mul(networkp.hImg/calibrationp.hImg)

local prev_img = load_image(opt.root_directory, '.', calibrationp, datap.first_image)
prev_img = rescale(prev_img, calibrationp.wImg, calibrationp.hImg)
prev_scaled = rescale(prev_img, networkp.wImg, networkp.hImg)

e2p = nil


for i = 2,datap.n_images do
   local timer = torch.Timer()
   local iImg = (i-1)*datap.delta+datap.first_image
   print(iImg)
   local img = load_image(opt.root_directory, '.', calibrationp, iImg)
   print("load   : "..timer:time().real)
   img = rescale(img, calibrationp.wImg, calibrationp.hImg)
   print("rescale: "..timer:time().real)
   --[[
   local R,T,nFound,nInliers,fundmat, inliers =
      sfm2.getEgoMotion{im1 = prev_img, im2 = img, K = calibrationp.K,
			maxPoints = calibrationp.sfm.max_points,
			pointsQuality=calibrationp.sfm.points_quality,
			ransacMaxDist=calibrationp.sfm.ransac_max_dist,
			pointsMinDistance=20,
			getInliers = true}
   T = calibrationp.K * T
   local e = T/T[3]
   --]]

   local R2,T2,nFound2,nInliers2,fundmat2, inliers2 =
      sfm2.getEgoMotion2{im1 = prev_img, im2 = img, K = calibrationp.K,
			 maxPoints = calibrationp.sfm.max_points,
			 pointsQuality=calibrationp.sfm.points_quality,
			 ransacMaxDist=0.02,
			 pointsMinDistance=50,
			 getInliers = true}
   print('nfound2: ' .. nFound2)
   T2 = calibrationp.K * T2
   local e2 = T2/T2[3]


   local e = e2
   local R = R2
   local inliers = inliers2
   --local e2 = e
   --local R2 = R
   --local inliers2 = inliers
   --[[
   local Rb = torch.mm(torch.mm(calibrationp.K, sfm2.inverse(R2)), sfm2.inverse(calibrationp.K))
   --local Rb = R
      
   local im_cpy = prev_img:clone()
   for i = 1,inliers2:size(1) do
      local t = torch.Tensor(3, 1):fill(1)
      t[1][1] = inliers2[i][1]
      t[2][1] = inliers2[i][2]
      t = torch.mm(Rb, t)
      t = t/t[3][1]
      local u = torch.Tensor(3, 1):fill(1)
      u[1][1] = inliers2[i][3]
      u[2][1] = inliers2[i][4]
      t = u - (u-t)*100
      draw.line(im_cpy, t[1][1], t[2][1], u[1][1], u[2][1], 1, 0, 1)
   end
   draw.point(im_cpy, e[1],  e[2],  10, 1, 0, 0)
   draw.point(im_cpy, e2[1], e2[2], 10, 0, 0, 1)
   w42 = image.display{image = image.scale(im_cpy, im_cpy:size(3)/2, im_cpy:size(2)/2),
		       win=w42}
   --]]
   
   e2 = e2*networkp.wImg/calibrationp.wImg
   --e2[1] = networkp.wImg/2
   --e2[2] = networkp.hImg/2
   if e2p ~= nil then
      local alpha = 1.0
      if (e2[1] >= 2) and (e2[2] >= 2) and (e2[1] <= networkp.wImg-1) and (e2[2] < networkp.hImg-1) then
	 e2 = e2*alpha+e2p*(1-alpha)
      else
	 e2 = e2p
      end
   end
   e2p = e2
   print("sfm    : "..timer:time().real)
   local rmax = getRMax(networkp.hImg, networkp.wImg, e2)
   local polarWarpMaskPad = getC2PMask(networkp.wImg, networkp.hImg,
				       networkp.wInput, networkp.hInput,
				       e2[1], e2[2],
				       math.floor((networkp.wKernel-1)/2),
				       math.ceil((networkp.wKernel-1)/2), rmax, alpha_polar)
   print("mask1  : "..timer:time().real)
   img_scaled = rescale(img, networkp.wImg, networkp.hImg)
   print("rescale: "..timer:time().real)
   prev_warped, prev_img_mask = sfm2.removeEgoMotion(prev_scaled, Ksmall, R2, 'bilinear')
   polar_img = cartesian2polar(img_scaled, polarWarpMaskPad)
   polar_prev = cartesian2polar(prev_warped, polarWarpMaskPad)
   print("warps  : "..timer:time().real)

   local img_disp = img_scaled:clone()
   local prev_disp = prev_warped:clone()
   draw.point(prev_disp, e2[1], e2[2], 3, 1, 0, 0)
   draw.point(img_disp,  e2[1], e2[2], 3, 1, 0, 0)
   win2 = image.display{image={polar_prev, polar_img}, win=win2}
   print("debug  : "..timer:time().real)
      
   local output = network:forward({polar_prev, polar_img})
   local _, idx = output:min(3)
   idx = torch.Tensor(idx:squeeze():size()):copy(idx)
   idx:add(-1)
   print("forward: "..timer:time().real)
   
   local scaled_idx = torch.Tensor(idx:size())
   for i = 1,scaled_idx:size(1) do
      scaled_idx[i]:copy((idx[i]+i-1):pow(alpha_polar) - (i-1)^(alpha_polar))
   end
   win_idx = image.display{image={scaled_idx, idx}, win=win_idx}
   local p2cmask = getP2CMaskOF(networkp, e2, apha_polar)
   local cartidx = cartesian2polar(scaled_idx, p2cmask)
   --scaled_idx = idx

   local center = e2
   winflow = image.display{image=cartidx, win=winflow}

   local center2 = center * getKOutput(networkp)
   depth, confs = flow2depth(networkp, cartidx, center2, 0.65)
   colordepth = depth2color(depth, confs)
   colordepth = padOutput(networkp, colordepth)   
   win4b = image.display{image=colordepth+img_scaled, win=win4b}

   timer:reset()
   local flowcv = sfm2.getOpticalFlow(prev_scaled, img_scaled)
   print("opencv: "..timer:time().real)
   cartidx = (flowcv[1]:cmul(flowcv[1]) + flowcv[2]:cmul(flowcv[2])):sqrt()
   local depth, confs = flow2depth(networkp, cartidx, center, 0.65)
   local colordepth = depth2color(depth, confs)
   win4 = image.display{image=colordepth+img_scaled, win=win4}
   
   win3 = image.display{image={prev_disp, img_disp}, win=win3}
   --win5 = image.display{image=colordepth2+img_scaled, win=win5}
   
   prev_img = img
   prev_scaled = img_scaled
end