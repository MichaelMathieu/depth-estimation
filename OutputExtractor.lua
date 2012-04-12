local OutputExtractor = torch.class('nn.OutputExtractor', 'nn.Module')

function OutputExtractor:__init(training_mode)
   self.training_mode = training_mode or false
   if self.training_mode then
      self.module = nn.Sequential()
      self.module:add(nn.Minus())
      self.module:add(nn.SpatialClassifier(nn.LogSoftMax()))
   end
end

function OutputExtractor:updateOutput(input)
   if self.training_mode then
      return self.module:updateOutput(input)
   else
      return input:min(1)[1]
   end
end

function OutputExtractor:updateGradInput(input, gradOutput)
   if self.training_mode then
      return self.module:updateGradInput(input, gradOutput)
   else
      error("Can't use OutputExtractor backprop when not in training mode")
   end
end
