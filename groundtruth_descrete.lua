require 'torch'
require 'paths'
require 'sys'
require 'xlua'

require 'common'

depthDescretizer = {}
depthDescretizer.nClasses = -1
depthDescretizer.cutDepth = 0
depthDescretizer.binStep = 0

function depthDescretizer:getClass(depth)
   local step = 2*self.cutDepth/self.nClasses
   local class = math.ceil(depth/step)
   if class > self.nClasses then
      return self.nClasses
   end
   return class
end

function depthDescretizer:computeCutDepth(histogram, nUsedPatches)
   local nPerClass = torch.Tensor(self.nClasses):zero()
      
   local numberOfSamples = 0
   for i = 1,numberOfBins do
      numberOfSamples = numberOfSamples + #(histogram[i])
      if numberOfSamples > nUsedPatches/2 then
	 self.cutDepth = i
	 break
      end
   end
   print("cutDepth is " .. self.cutDepth)
   
   for i = 1,numberOfBins do
      --print("Bin " .. i .. " has " .. #(patchesMedianDepth[i]) .. " elements")
      nPerClass[self:getClass(i-0.1)] = nPerClass[self:getClass(i-0.1)] + #(histogram[i])
   end
   for i = 1,self.nClasses do
      print("Class " .. i .. " has " .. nPerClass[i] .. " elements")
   end

   self.binStep = math.floor(2*self.cutDepth/self.nClasses)
end

function depthDescretizer:randomBin(histogram)
   local randomClass = randInt(1, self.nClasses+1)
   local randomBinIndex
   local sizeOfBin = 0
   while sizeOfBin == 0 do
      randomBinIndex = randInt((randomClass-1)*self.binStep+1, randomClass*self.binStep+1)
      sizeOfBin = table.getn(histogram[randomBinIndex])
   end
   return randomBinIndex
end