require 'torch'
require 'nnx'
require 'image'
require 'optim'
require 'load_data'
require 'groundtruth_opticalflow'
require 'sys'

op = xlua.OptionParser('%prog [options]')
op:option{'-n', '--n-train-set', action='store', dest='n_train_set', default=2000,
	  help='Number of patches in the training set'}
op:option{'-m', '--n-test-set', action='store', dest='n_test_set', default=1000,
	  help='Number of patches in the test set'}
op:option{'-d', '--delta', action='store', dest='delta', default=2,
	  help='Delta between two consecutive frames'}
op:option{'-rd', '--root-directory', action='store', dest='root_directory',
	  default='./data', help='Root dataset directory'}
op:option{'-nt', '--num-threads', action='store', dest='nThreads', default=2,
	  help='Number of threads used'}
op:option{'-ni', '--num-input-images', action='store', dest='num_input_images',
	  default=10, help='Number of annotated images used'}
op:option{'-e', '--num-epochs', action='store', dest='n_epochs', default=10,
	  help='Number of epochs'}

opt=op:parse()
opt.nThreads = tonumber(opt.nThreads)
opt.n_train_set = tonumber(opt.n_train_set)
opt.n_test_set = tonumber(opt.n_test_set)
opt.n_epochs = tonumber(opt.n_epochs)
opt.num_input_images = tonumber(opt.num_input_images)

torch.manualSeed(1)

if opt.nThreads > 1 then
   require 'openmp'
   openmp.setDefaultNumThreads(opt.nThreads)
end

local geometry = {}
geometry.wImg = 320
geometry.hImg = 180
geometry.wPatch2 = 32
geometry.hPatch2 = 32
geometry.wKernel = 16
geometry.hKernel = 16
geometry.maxw = geometry.wPatch2 - geometry.wKernel + 1
geometry.maxh = geometry.hPatch2 - geometry.hKernel + 1
geometry.wPatch1 = geometry.wPatch2 - geometry.maxw + 1
geometry.hPatch1 = geometry.hPatch2 - geometry.maxh + 1
geometry.nChannelsIn = 3
geometry.nFeatures = 10

function prepareInput(geometry, patch1, patch2)
   ret = {}
   ret[1] = patch1:narrow(2, math.ceil(geometry.maxh/2), geometry.hPatch1)
                  :narrow(3, math.ceil(geometry.maxw/2), geometry.wPatch1)
   ret[2] = patch2
   return ret
end

local model = nn.Sequential()
local parallel = nn.ParallelTable()
local parallelElem1 = nn.Sequential()
local parallelElem2 = nn.Sequential()
local conv = nn.SpatialConvolution(geometry.nChannelsIn, geometry.nFeatures,
				   geometry.wKernel, geometry.hKernel)
parallelElem1:add(nn.Reshape(geometry.nChannelsIn, geometry.hPatch1, geometry.wPatch1))
parallelElem1:add(conv)
parallelElem1:add(nn.Tanh())

parallelElem2:add(nn.Reshape(geometry.nChannelsIn, geometry.hPatch2, geometry.wPatch2))
parallelElem2:add(conv)
parallelElem2:add(nn.Tanh())

parallel:add(parallelElem1)
parallel:add(parallelElem2)
model:add(parallel)

model:add(nn.SpatialMatching(geometry.maxh, geometry.maxw, false))
model:add(nn.Reshape(geometry.hPatch2 - geometry.hKernel - geometry.maxh + 2,
		     geometry.wPatch2 - geometry.wKernel - geometry.maxw + 2,
		     geometry.maxw*geometry.maxh))
model:add(nn.Reshape(geometry.maxw*geometry.maxh)) --todo

model:add(nn.Minus())
model:add(nn.LogSoftMax())

parameters, gradParameters = model:getParameters()

criterion = nn.ClassNLLCriterion()

print('Loading images...')
raw_data = loadDataOpticalFlow(geometry, 'data/', opt.num_input_images, opt.delta)
--raw_data[2] = raw_data[1]
print('Generating training set...')
trainData = generateDataOpticalFlow(geometry, raw_data, opt.n_train_set);
print('Generating test set...')
testData = generateDataOpticalFlow(geometry, raw_data, opt.n_test_set);

for iEpoch = 1,opt.n_epochs do
   print('Epoch ' .. iEpoch)


   nGood = 0
   nBad = 0

   for t = 1,testData:size() do
      xlua.progress(t, testData:size())
      local sample = testData[t]
      local input = prepareInput(geometry, sample[1][1], sample[1][2])
      local target = yx2x(geometry, sample[2][2], sample[2][1])
      
      local output = model:forward(input)
      output = output:squeeze()
      
      _, ioutput = output:max(1)
      ioutput = ioutput:squeeze()
      a1, a2 = x2yx(geometry, ioutput)
      b1, b2 = x2yx(geometry, target)
      --print(a1 .. " " .. a2 .. " | " .. b1 .. " " .. b2)
      if ioutput == target then
	 nGood = nGood + 1
      else
	 nBad = nBad + 1
      end
   end
      
   print('nGood = ' .. nGood .. ' nBad = ' .. nBad)


   nGood = 0
   nBad = 0
   
   for t = 1,trainData:size() do
      xlua.progress(t, trainData:size())
      local sample = trainData[t]
      local input = prepareInput(geometry, sample[1][1], sample[1][2])
      local target = yx2x(geometry, sample[2][2], sample[2][1])
      
      local feval = function(x)
		       if x ~= parameters then
			  parameters:copy(x)
		       end
		       gradParameters:zero()
		       
		       local output = model:forward(input)
		       --output = torch.Tensor(output:size()):fill(1) - output:abs()
		       output = output:squeeze()
		       local err = criterion:forward(output, target)
		       local df_do = criterion:backward(output, target)
		       model:backward(input, df_do)
		       
		       _, ioutput = output:max(1)
		       ioutput = ioutput:squeeze()
		       --print(ioutput .. ' ' .. target)
		       if ioutput == target then
			  nGood = nGood + 1
		       else
			  nBad = nBad + 1
		       end

		       return err, gradParameters
		    end

      config = {learningRate = 1e-2,
		weightDecay = 0,
		momentum = 0,
		learningRateDecay = 5e-7}
      optim.sgd(feval, parameters, config)
   end
      
   print('nGood = ' .. nGood .. ' nBad = ' .. nBad)

end
