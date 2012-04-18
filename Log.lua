require 'nn'
local Log2, parent = torch.class('nn.Log2', 'nn.Module')

function Log2:__init()
   parent.__init(self)
   
   -- state
   self.output = torch.Tensor()
   self.gradInput = torch.Tensor()
end

function Log2:updateOutput(input)
   self.output:resizeAs(input)
   self.output:copy(input):log()
   return self.output 
end

function Log2:updateGradInput(input, gradOutput) 
   self.gradInput:resizeAs(gradOutput)
   self.gradInput:copy(gradOutput):cdiv(input)
   return self.gradInput
end
