require 'torch'

local cam = 'rectified_gopro'

if cam == 'ardrone' then
   wImg = 640
   hImg = 480
   
   sfm = {}
   sfm.max_points = 400
   sfm.points_quality = 0.001
   sfm.ransac_max_dist = 1
   bad_image_threshold = 0.2
   
   K = torch.FloatTensor(3,3):zero()
   K[1][1] = 293.824707 -- x focal length (in pixels)
   K[2][2] = 310.435730 -- y focal length (in pixels)
   K[1][3] = 300.631012 -- x principal point (in pixels)
   K[2][3] = 251.624924 -- y principal point (in pixels)
   K[1][2] = 0.0        -- skew (most of the time 0)
   K[3][3] = 1.0
   
   distortion = torch.FloatTensor(5)
   distortion[1] = -0.379940
   distortion[2] = 0.212737
   distortion[3] = 0.003098
   distortion[4] = 0.000870
   distortion[5] = -0.069770
   
   filename = 'ardrone.cal'
elseif cam == 'rectified_gopro' then
   wImg = 1280
   hImg = 720

   sfm = {}
   sfm.max_points = 1000
   sfm.points_quality = 0.0001
   sfm.points_min_dist = 50
   sfm.ransac_max_dist = 1
   sfm.ransac2_max_dist = 0.02
   bad_image_threshold = 0.2
   
   K = torch.FloatTensor(3,3):zero()
   K[1][1] = 602.663208 -- x focal length (in pixels)
   K[2][2] = 603.193289 -- y focal length (in pixels)
   K[1][3] = 641.455200 -- x principal point (in pixels)
   K[2][3] = 344.950836 -- y principal point (in pixels)
   K[1][2] = 0.0        -- skew (most of the time 0)
   K[3][3] = 1.0
   
   distortion = torch.FloatTensor(5):zero()
   
   filename = 'rectified_gopro.cal'
elseif cam == 'gopro' then
   wImg = 1280
   hImg = 720

   sfm = {}
   sfm.max_points = 400
   sfm.points_quality = 0.001
   sfm.ransac_max_dist = 1
   bad_image_threshold = 0.2
   
   K = torch.FloatTensor(3,3):zero()
   K[1][1] = 602.663208 -- x focal length (in pixels)
   K[2][2] = 603.193289 -- y focal length (in pixels)
   K[1][3] = 641.455200 -- x principal point (in pixels)
   K[2][3] = 344.950836 -- y principal point (in pixels)
   K[1][2] = 0.0        -- skew (most of the time 0)
   K[3][3] = 1.0
   
   distortion = torch.FloatTensor(5)
   distortion[1] = -0.355740
   distortion[2] = 0.142684
   distortion[3] = 0.000469
   distortion[4] = 0.000801
   distortion[5] = -0.027673
   
   filename = 'gopro.cal'
end   

local tosave = {}
tosave.wImg = wImg
tosave.hImg = hImg
tosave.sfm = sfm
tosave.bad_image_threshold = bad_image_threshold
tosave.K = K
tosave.distortion = distortion

torch.save(filename, tosave)