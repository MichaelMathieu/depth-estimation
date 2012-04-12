require 'nn'
local Mul2, parent = torch.class('nn.Mul2', 'nn.Module')

function Mul2:__init()
   parent.__init(self)
  
   self.weight = torch.Tensor(1)
   self.gradWeight = torch.Tensor(1)
   
   -- state
   self.output = torch.Tensor()
   self.gradInput = torch.Tensor()

   self:reset()
end

 
function Mul2:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(1))
   end

   self.weight[1] = torch.uniform(-stdv, stdv);
end

function Mul2:updateOutput(input)
   self.output:resizeAs(input)
   self.output:copy(input);
   self.output:mul(self.weight[1]);
   return self.output 
end

function Mul2:updateGradInput(input, gradOutput) 
   self.gradInput:resizeAs(input)
   self.gradInput:zero()
   self.gradInput:add(self.weight[1], gradOutput)
   return self.gradInput
end

function Mul2:accGradParameters(input, gradOutput, scale) 
   scale = scale or 1
   self.gradWeight[1] = self.gradWeight[1] + scale*input:dot(gradOutput);
end
