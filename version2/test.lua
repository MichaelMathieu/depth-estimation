require 'data'
require 'network'

local calibrationp = torch.load('rectified_gopro.cal')

local datap = {
   wImg = 320,
   hImg = 180,
   normalization_k = 17,
   layers = {
      {3,17,17,32}
   }
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

local dataset
if paths.filep(dataset_filename) then
   dataset = torch.load(dataset_filename)
else
   dataset = new_dataset('data/', calibrationp, datap, groundtruthp)
   dataset:add_subdir('part1')
end

local network = getNetwork(datap)
local parameters, gradParameters = network:getParameters()
parameters:copy(torch.load('models/e106_no_bin'))

local input = {dataset:get_prev_image_by_idx(1), dataset:get_image_by_idx(1)}
local output = network:forward(input)
local _, idx = output:min(3)
idx = idx:add(-1):squeeze()
local yflow = (idx/datap.wWin):floor()
local xflow = idx-yflow*datap.wWin-datap.lWin
yflow = yflow-datap.tWin

image.display{xflow, yflow}