require 'nnx'
local OutputExtractor = torch.class('nn.OutputExtractor', 'nn.Module')

function OutputExtractor:__init(maxh, maxw)
   self.xmulb = torch.Tensor(maxh*maxw)
   self.ymulb = torch.Tensor(maxh*maxw)
   for i = 1,maxh do
      for j = 1,maxw do
	 local k = (i-1) * maxw + j
	 self.ymulb[k] = i
	 self.xmulb[k] = j
      end
   end
   self.output = nil
   self.gradInput = torch.Tensor()
   self.tmp = torch.Tensor()
   self.xmul = torch.Tensor()
   self.ymul = torch.Tensor()
end

function OutputExtractor:updateOutput(input)
   if self.xmul:size() ~= input:size() then
      self.xmul = nn.Replicate(input:size(2)):forward(self.xmulb)
      self.xmul = nn.Replicate(input:size(1)):forward(self.xmul)
      self.ymul = nn.Replicate(input:size(2)):forward(self.ymulb)
      self.ymul = nn.Replicate(input:size(1)):forward(self.ymul)
   end
   self.tmp:resizeAs(input)
   self.tmp:zero():addcmul(1, input, self.xmul)
   local x = self.tmp:sum(3)[{{},{},1}]
   self.tmp:zero():addcmul(1, input, self.ymul)
   local y = self.tmp:sum(3)[{{},{},1}]
   self.output = {x, y}
   return self.output
end

function OutputExtractor:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   self.gradInput:add(gradOutput[1]:squeeze(), self.xmul)
   self.gradInput:add(gradOutput[2]:squeeze(), self.ymul)
   return self.gradInput
end
