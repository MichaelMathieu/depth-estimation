require 'nn'
local Log2, parent = torch.class('nn.Log2', 'nn.Module')

function Log2:__init(check_null)
   parent.__init(self)
   self.check_null = check_null or false
   
   -- state
   self.output = torch.Tensor()
   self.gradInput = torch.Tensor()
end

function Log2:updateOutput(input)
   if self.check_null then
      self.bad = self.bad or torch.Tensor():resizeAs(input)
      self.bad:copy(input:lt(1e-20))
      input = self.bad:mul(1e-20) + (-self.bad+1):cmul(input)
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
