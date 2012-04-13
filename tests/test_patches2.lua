require 'groundtruth_opticalflow'
require 'image'
require 'opticalflow_model'


geometry = {}
geometry.wImg=320
geometry.hImg=180
geometry.hKernel=16
geometry.wKernel=16
geometry.layers = {{3,geometry.hKernel, geometry.wKernel, geometry.hKernel*geometry.wKernel*3}}
geometry.hKernelGT=16
geometry.wKernelGT=16
geometry.maxhGT=16
geometry.maxwGT=16
--geometry.multiscale=true
geometry.multiscale=false
geometry.ratios={1,2}
if geometry.multiscale then
   geometry.maxh=8
   geometry.maxw=8
else
   geometry.maxh=geometry.maxhGT
   geometry.maxw=geometry.maxwGT
end
geometry.wPatch2=geometry.maxw+geometry.wKernel-1
geometry.hPatch2=geometry.maxh+geometry.hKernel-1

nSamples = 100
iBegin=8

torch.manualSeed(1)

raw_data = loadDataOpticalFlow(geometry, 'data/', 2, '000000000', 1, false)
trainData = generateDataOpticalFlow(geometry, raw_data, iBegin+nSamples,
				    'uniform_position', false)

nGood = 0
nBad = 0

if geometry.multiscale then
   model = getModelMultiscale(geometry, false)
else
   model = getModel(geometry, false)
   local weights = model.modules[1].modules[1].modules[1].weight
   for i = 1,geometry.hKernel do
      for j = 1,geometry.wKernel do
	 for k = 1,3 do
	    weights[(i-1)*geometry.wKernel*3+(j-1)*3+k]:zero()
	    weights[(i-1)*geometry.wKernel*3+(j-1)*3+k][k][i][j] = 1
	 end
      end
   end
end

for iSample = iBegin,iBegin+nSamples-1 do
   sample = trainData[iSample]
   --image.display{image=sample[1], zoom=4}
   --image.display{image={nn.SpatialDownSampling(2,2):forward(sample[1][1]),
--			nn.SpatialDownSampling(2,2):forward(sample[1][2])}, zoom=8}
   if not geometry.multiscale then
      sample = trainData[iSample]
      input = prepareInput(geometry, sample[1][1], sample[1][2])
      --image.display{image=input[1], zoom=8}
      --image.display{image=input[2], zoom=8}
   else
      sample = trainData:getElemFovea(iSample)
      input = sample[1][1]
      model:focus(sample[1][2][2], sample[1][2][1])
      y = sample[1][2][1]
      x = sample[1][2][2]
   end
   target = prepareTarget(geometry, sample[2])
   output = model:forward(input)
   m = processOutput(geometry, output, false).index
   local ay,ax = x2yxMulti(geometry, m)
   local by,bx = x2yxMulti(geometry, target)
   if m==target then
   --if (math.abs(ax-bx) <= 1) and (math.abs(ay-by) <= 1) then
      nGood = nGood + 1
      print('+| ' .. ay .. ',' .. ax .. ' ' .. by .. ',' .. bx)
   else
      nBad = nBad + 1
      print('-| ' .. ay .. ',' .. ax .. ' ' .. by .. ',' .. bx)
   end
end
print("nGood = " .. nGood .. ' nBad = ' .. nBad)