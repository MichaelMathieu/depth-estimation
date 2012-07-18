require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
package.path = "./?.lua;../?.lua;" .. package.path
package.cpath = "./?.so;../?.so;" .. package.cpath
require 'data'

local calibrationp = torch.load('rectified_gopro.cal')

local datap = {
   wImg = 320,
   hImg = 180,
   normalization_k = 17,
   hKernel = 17,
   wKernel = 17,
   wWin = 17,
   hWin = 17
}
datap.lWin = math.ceil(datap.wWin/2)-1
datap.tWin = math.ceil(datap.hWin/2)-1
datap.rWin = math.floor(datap.wWin/2)
datap.bWin = math.floor(datap.hWin/2)

local groundtruthp = {
   type = 'cross-correlation',
   params = {
      wWin = 17,
      hWin = 17,
      wKernel = 17,
      hKernel = 17
   }
}

local learningp = {
   nEpochs = 10,
   rate = 1e-4,
   weightDecay = 1e-8,
   rateDecay = 1e-4,
   trainingSetSize = 10,
   testSetSize = 20
}

local dataset = new_dataset('data/no-risk/part1', calibrationp, datap, groundtruthp)

local im1 = dataset:get_prev_image_by_idx(42)
local im2 = dataset:get_image_by_idx(42)
local gt = dataset:get_gt_by_idx(42)

image.display{im1, im2, im1-im2}
image.display(gt)