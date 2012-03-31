local CascadingAddTable, parent = torch.class('nn.CascadingAddTable', 'nn.Module')

function CascadingAddTable:__init(ratios)
   parent.__init(self)
   self.ratios = ratios
   self.gradInput = {}
   self.output = {}
   for i = 1,#self.ratios do
      self.gradInput[i] = torch.Tensor()
      self.output[i] = torch.Tensor()
   end
   self.transformers = nn.Sequential()
   for i = 1,#self.ratios-1 do
      self.transformers[i] = nn.Sequential()
      self.transformers[i]:add(nn.Narrow(1, 1, 1))
      self.transformers[i]:add(nn.Narrow(2, 1, 1))
      self.transformers[i]:add(nn.SpatialUpSampling(self.ratios[i+1]/self.ratios[i],
						    self.ratios[i+1]/self.ratios[i],
						    1, 2))
   end
end

function CascadingAddTable:updateOutput(input)
   for i = 1,#input do
      if input[i]:nDimension() ~= 4 then
	 error('nn.CascadingAddTable: input must be a table of Kh x Kw x H x W')
      end
   end
   if #input ~= #self.ratios then
      error('nn.CascadingAddTable: input and ratios must have the same size')
   end
   for i = 1,#input do
      self.output[i]:resizeAs(input[i]):copy(input[i])
   end
   for i = #input-1,1,-1 do
      local r = self.ratios[i]
      local r2 = self.ratios[i+1]
      if ((math.mod(input[i]:size(1) * (r2-r), 2*r2) ~= 0) or
       (math.mod(input[i]:size(2) * (r2-r), 2*r2) ~= 0)) then
	 error('nn.CascadingAddTable: ratios and input sizes not compatible')
      end
      local dh = input[i]:size(1) * (r2-r) / (2*r2)
      local dw = input[i]:size(1) * (r2-r) / (2*r2)
      self.transformers[i].modules[1].index = dh+1
      self.transformers[i].modules[1].length = input[i]:size(1)-2*dh
      self.transformers[i].modules[2].index = dw+1
      self.transformers[i].modules[2].length = input[i]:size(2)-2*dw
      self.output[i]:add(self.transformers[i]:forward(self.output[i+1]))
   end
   return self.output
end

function CascadingAddTable:updateGradInput(input, gradOutput)
   for i = 1,#input do
      self.gradInput[i]:resizeAs(input[i])
      self.gradInput[i]:copy(gradOutput[i])
   end
   for i = 2,#input do
      self.gradInput[i]:add(self.transformers[i-1]:backward(input[i],self.gradInput[i-1]))
   end
   return self.gradInput
end