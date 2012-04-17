local OutputExtractor = torch.class('nn.OutputExtractor', 'nn.Module')

function OutputExtractor:__init(training_mode, middleIndex)
   self.training_mode = training_mode or false
   if self.training_mode then
      self.module = nn.Sequential()
      self.module:add(nn.Minus())
      self.module:add(nn.SpatialClassifier(nn.LogSoftMax()))
   else
      self.middleIndex = middleIndex
   end
end

function OutputExtractor:updateOutput(input)
   if self.training_mode then
      return self.module:updateOutput(input)
   else
      --      t = torch.Timer()
      local m, idx = input:min(1)
      --print('min      : ' .. t:time()['real'])
      local flatPixels = torch.LongTensor(m:size(2), m:size(3)):copy(m:eq(input[self.middleIndex]))
      return flatPixels * self.middleIndex + (-flatPixels+1):cmul(idx:reshape(idx:size(2), idx:size(3)))
   end
end

function OutputExtractor:updateGradInput(input, gradOutput)
   if self.training_mode then
      return self.module:updateGradInput(input, gradOutput)
   else
      error("Can't use OutputExtractor backprop when not in training mode")
   end
end
