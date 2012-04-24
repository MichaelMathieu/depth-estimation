require 'nn'
local Log2, parent = torch.class('nn.Log2', 'nn.Module')

function Log2:__init(null_epsilon)
   parent.__init(self)
   self.null_epsilon = null_epsilon or nil
   
   -- state
   self.output = torch.Tensor()
   self.gradInput = torch.Tensor()
end

function Log2:updateOutput(input)
   if self.null_epsilon then
      self.bad = self.bad or torch.Tensor():resizeAs(input)
      self.bad:copy(input:lt(null_epsilon))
      input:set(self.bad:mul(null_epsilon) + (-self.bad+1):cmul(input))
   end
   self.output:resizeAs(input)
   self.output:copy(input):log()
   return self.output 
end

function Log2:updateGradInput(input, gradOutput) 
   self.gradInput:resizeAs(gradOutput)
   self.gradInput:copy(gradOutput):cdiv(input)
   return self.gradInput
end
