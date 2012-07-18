require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
require 'image'
require 'optim'
require 'xlua'
require 'network'
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
   nEpochs = 1000,
   rate = 1e-3,
   weightDecay = 1e-8,
   rateDecay = 1e-3,
   trainingSetSize = 1000,
   testSetSize = 5
}

local dataset_filename = 'dataset'
local dataset
if paths.filep(dataset_filename) then
   dataset = torch.load(dataset_filename)
else
   dataset = new_dataset('data/no-risk/', calibrationp, datap, groundtruthp)
   dataset:add_subdir('part1')
end

local network = getTrainerNetwork(datap)
local parameters, gradParameters = network:getParameters()

local criterion = nn.ClassNLLCriterion()
local config = {learningRate = learningp.rate,
		weightDecay = learningp.weightDecay,
		momentum = 0,
		learningRateDecay = learningp.rateDecay}

for iEpoch = 1,learningp.nEpochs do
   print("Epoch " .. iEpoch .. " over " .. learningp.nEpochs)

   win = image.display{image=network:getWeights().layer1, padding=2, zoom=4, win=win}

   local trainData = dataset:get_patches(learningp.trainingSetSize)
   --torch.save(dataset_filename, dataset)
   local meanErr = 0
   
   for iSample = 1,#trainData do
      xlua.progress(iSample, #trainData)
      local input = {trainData[iSample].patch1(), trainData[iSample].patch2()}
      local targetCrit = trainData[iSample].targetCrit
      local feval = function(x)
		       if x ~= parameters then
			  parameters:copy(x)
		       end
		       gradParameters:zero()
		       local output = network:forward(input)
		       local err = criterion:forward(output, targetCrit)
		       meanErr = meanErr + err
		       local df_do = criterion:backward(output, targetCrit)
		       network:backward(input, df_do)
		       return err, gradParameters
		    end
      optim.sgd(feval, parameters, config)
   end

   meanErr = meanErr / #trainData
   print('Training error : ' .. meanErr)
   collectgarbage()

   local testData = dataset:get_patches(learningp.testSetSize)
   local nGood = 0
   for iSample = 1,#testData do
      local input = {testData[iSample].patch1(), testData[iSample].patch2()}
      local targetCrit = testData[iSample].targetCrit

      local output = network:forward(input)
      local _, output_class = output:max(1)
      output_class = output_class:squeeze()
      if output_class == targetCrit then
	 nGood = nGood + 1
      end
   end

   print('Test precision rate : ' .. nGood/#testData)
   collectgarbage()
end