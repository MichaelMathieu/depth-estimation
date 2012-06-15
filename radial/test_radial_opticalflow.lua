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

local calibrationp = torch.load(opt.calibration_file)

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
   local R,T,nFound,nInliers,fundmat =
      sfm2.getEgoMotion{im1 = prev_img, im2 = img, K = calibrationp.K,
			maxPoints = calibrationp.sfm.max_points,
			pointsQuality=calibrationp.sfm.points_quality,
			ransacMaxDist=calibrationp.sfm.ransac_max_dist}
   
   local _, e2 = sfm2.getEpipoles(fundmat)
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
   local rmax = getRMax(networkp, e2)
   local polarWarpMaskPad = getC2PMask(networkp.wImg, networkp.hImg,
				       networkp.wInput, networkp.hInput,
				       e2[1], e2[2],
				       math.floor((networkp.wKernel-1)/2),
				       math.ceil((networkp.wKernel-1)/2), rmax)
   print("mask1  : "..timer:time().real)
   img_scaled = rescale(img, networkp.wImg, networkp.hImg)
   print("rescale: "..timer:time().real)
   prev_warped, prev_img_mask = sfm2.removeEgoMotion(prev_scaled, Ksmall, R, 'bilinear')
   polar_img = cartesian2polar(img_scaled, polarWarpMaskPad)
   polar_prev = cartesian2polar(prev_warped, polarWarpMaskPad)
   print("warps  : "..timer:time().real)

   local img_disp = img_scaled:clone()
   local prev_disp = prev_warped:clone()
   draw.point(prev_disp, e2[1], e2[2], 3, 1, 0, 0)
   draw.point(img_disp,  e2[1], e2[2], 3, 1, 0, 0)
   --win2 = image.display{image={polar_prev, polar_img}, win=win2}
   print("debug  : "..timer:time().real)
      
   local output = network:forward({polar_prev, polar_img})
   local _, idx = output:min(3)
   idx = torch.Tensor(idx:squeeze():size()):copy(idx)
   idx:add(-1)
   print("forward: "..timer:time().real)

   local p2cmask = getP2CMaskOF(networkp, e2)
   local cartidx = cartesian2polar(idx, p2cmask)

   local depth, confs = flow2depth(networkp, cartidx, nil, 0.65)
   local colordepth = depth2color(depth, confs)
   colordepth = padOutput(networkp, colordepth)
   --win = image.display{image=colordepth, win=win}
   --win2 = image.display{image=cartidx, win=win2}
   
   --[[
   p2cmask = getP2CMask(networkp.wInput, networkp.hInput, networkp.wImg, networkp.hImg,
			e2[1], e2[2], rmax)
   local idx2 = torch.Tensor(networkp.hInput, idx:size(2))
   idx2:sub(1, output:size(1)):copy(idx)
   cartidx = cartesian2polar(idx2, p2cmask)
   depth, confs = flow2depth(networkp, cartidx, nil, 0.65)
   local colordepth2 = depth2color(depth, confs)--]]

   win4 = image.display{image=colordepth+img_scaled, win=win4}
   win3 = image.display{image={prev_disp, img_disp}, win=win3}
   --win5 = image.display{image=colordepth2+img_scaled, win=win5}
   

   prev_img = img
   prev_scaled = img_scaled
end