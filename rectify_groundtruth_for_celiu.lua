require 'torch'
require 'image'
require 'groundtruth_opticalflow'
require 'sfm2'
require 'sys'

local datapath = 'data/no-risk/part7/'
local ext = 'jpg'
local w = 320.
local camera = 'gopro'

local K = torch.Tensor(3,3):zero()
local distP = torch.Tensor(5)
if camera == 'gopro' then
   K[1][1] = 602.663208
   K[2][2] = 603.193289
   K[1][3] = 641.455200
   K[2][3] = 344.950836
   K[3][3] = 1.0
   distP = torch.FloatTensor(5)
   distP[1] = -0.355740
   distP[2] = 0.142684
   distP[3] = 0.000469
   distP[4] = 0.000801
   distP[5] = -0.027673
else
   K[1][1] = 293.824707
   K[1][2] = 0.
   K[1][3] = 310.435730
   K[2][1] = 0.
   K[2][2] = 300.631012
   K[2][3] = 251.624924
   K[3][1] = 0.
   K[3][2] = 0.
   K[3][3] = 1.
   distP[1] = -0.37994
   distP[2] = 0.212737
   distP[3] = 0.003098
   distP[4] = 0.00087
   distP[5] = -0.069770
end
local Kf = torch.FloatTensor(K:size()):copy(K)

if datapath:sub(-1) ~= '/' then datapath = datapath .. '/' end
local i = 1
local previmname = string.format("%09d.%s", i, ext)
local previm = image.load(datapath..'images/'..previmname)
if camera ~= 'gopro' then
   previm = sfm2.undistortImage(previm, K, distP)
end
i = i+1
local imname = string.format("%09d.%s", i, ext)

local h = previm:size(2)*w/previm:size(3)

sys.execute('mkdir -p '..datapath..'undistorted_images/')
sys.execute('mkdir -p '..datapath..'rectified_images/')

while sys.filep(datapath..'images/'..imname) do
   print(imname)
   local im = image.load(datapath..'images/'..imname)
   if camera ~= 'gopro' then
      im = sfm2.undistortImage(im, K, distP)
   end
   local R, T, nf, ni = sfm2.getEgoMotion(previm, im, Kf, 1000)
   local prev_warped = sfm2.removeEgoMotion(previm, Kf, R)
   image.save(datapath..'undistorted_images/'..imname, im)
   image.save(datapath..'rectified_images/'..previmname, prev_warped)
   previm = im
   previmname = imname
   i = i + 1
   imname = string.format("%09d.%s", i, ext)
   collectgarbage()
end