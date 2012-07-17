require 'data'

local calibrationp = torch.load('rectified_gopro.cal')

local datap = {
   wImg = 320,
   hImg = 180
}

local groundtruthp = {
   type = 'cross-correlation',
   params = {
      wWin = 17,
      hWin = 17,
      wKernel = 17,
      hKernel = 17
   }
}

local dataset = new_dataset('data/no-risk/part1', calibrationp, datap, groundtruthp)

local img = dataset:get_image_by_idx(42)
local prev_img = dataset:get_prev_image_by_idx(42)
local gt = dataset:get_gt_by_idx(47)
image.display{prev_img, img}
image.display{gt}