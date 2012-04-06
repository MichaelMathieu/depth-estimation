require 'groundtruth_opticalflow'
require 'image'


geometry = {}
geometry.wImg=320
geometry.hImg=180
geometry.hKernel=16
geometry.wKernel=16
geometry.hKernelGT=16
geometry.wKernelGT=16
geometry.maxh=16
geometry.maxw=16
geometry.maxhGT=16
geometry.maxwGT=16
geometry.maxhMS = geometry.maxh
geometry.maxwMS = geometry.maxw
geometry.wPatch2=geometry.maxw+geometry.wKernel-1
geometry.hPatch2=geometry.maxh+geometry.hKernel-1
geometry.nChannelsIn=3
geometry.motion_correction = true

nSamples = 1000

--T = -20
--while T < 21 do
torch.manualSeed(1)

raw_data = loadDataOpticalFlow(geometry, 'data/', 2, '000000012', 1)
trainData = generateDataOpticalFlow(geometry, raw_data, nSamples, 'uniform_position')

image.display(raw_data.flow[1])

nGood = 0
nBad = 0

--print(trainData.patches)


for iSample = 1,nSamples do
   sample = trainData[iSample]
   input = prepareInput(geometry, sample[1][1], sample[1][2])
   --image.display{image=input[1], zoom=4,min=0,max=1}
   --image.display{image=input[2], zoom=4,min=0,max=1}
   targetCrit, target = prepareTarget(geometry, sample[2])
   input1 = input[1]:reshape(geometry.hKernel*geometry.wKernel*3,1,1)
   input2 = input[2]:unfold(2, geometry.hKernel, 1):unfold(3, geometry.wKernel, 1)
   input2b = torch.Tensor(geometry.wKernel*geometry.hKernel*3, geometry.maxh, geometry.maxw)
   for i = 1,geometry.maxh do
      for j = 1,geometry.maxw do
	 input2b:select(2,i):select(2,j):copy(input2:select(2,i):select(2,j):reshape(3*geometry.wKernel*geometry.hKernel))
      end
   end

   net = nn.SpatialMatching(geometry.maxh, geometry.maxw, false)
   output = net:forward({input1, input2b})
   output = -output
   output2 = output:reshape(geometry.maxh*geometry.maxw)
   _, m = output2:max(1)
   m = m:squeeze()
   --p = processOutput(geometry, output2)
   --m = p.index
   --print(m .. ' ' .. target)
   --image.display{image=output:reshape(geometry.maxh, geometry.maxw), zoom=4}
   if m==target then
      nGood = nGood + 1
   else
      nBad = nBad + 1
      --[[
      print('--')
      print(x2yx(geometry, m))
      print(x2yx(geometry, target))
      --]]
      --image.display{image=sample[1], zoom=4,min=0,max=1}
   end
end
print("nGood = " .. nGood .. ' nBad = ' .. nBad)

--T = T+1
--end