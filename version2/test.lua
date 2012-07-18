require 'data'
require 'network'

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

local dataset = new_dataset('data/no-risk/part1', calibrationp, datap, groundtruthp)

local img = dataset:get_image_by_idx(42)
local prev_img = dataset:get_prev_image_by_idx(42)
local gt = dataset:get_gt_by_idx(42)

local network = getNetwork(datap)
local output = network:forward({prev_img, img})
output:reshape(output:size(1), output:size(2), output:size(3)*output:size(4))
local _, idx = output:min(3)
idx = idx:add(-1):squeeze()
local yflow = (idx/datap.wWin):floor()
local xflow = (idx - yflow*datap.wWin)
yflow:add(-math.ceil(datap.hWin/2)+1)
xflow:add(-math.ceil(datap.wWin/2)+1)