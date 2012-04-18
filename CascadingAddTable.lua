require 'nnx'
require 'Mul2'
require 'Log'
require 'SpatialZeroPadding2'
local CascadingAddTable, parent = torch.class('nn.CascadingAddTable', 'nn.Module')
local CascadingAddTableSplit, parent = torch.class('nn.CascadingAddTableSplit', 'nn.Module')

function CascadingAddTable:__init(ratios, trainable)
   parent.__init(self)
   self.ratios = ratios
   if trainable == nil then self.trainable = true else self.trainable = trainable end
   self.gradInput = {}
   self.outputBP = {}
   self.output = {}
   for i = 1,#self.ratios do
      self.gradInput[i] = torch.Tensor()
      self.outputBP[i] = torch.Tensor()
   end
   self.muls = {}
   self.padders = {}
   self.transformers = {}
   local beta = 1.
   for i = 1,#self.ratios-1 do
      local seq1 = nn.Sequential()
      --[[
      local mul1 = nn.Mul2()
      self.muls[#self.muls+1] = mul1
      seq1:add(mul1)
      seq1:add(nn.Tanh())
      --]]
      local mul1 = nn.Mul2()
      mul1.weight[1] = -beta
      seq1:add(mul1)
      seq1:add(nn.Exp())
      
      local seq2 = nn.Sequential()
      self.padders[i] = nn.SpatialZeroPadding2(0,0,0,0, 1, 2)
      seq2:add(self.padders[i])
      seq2:add(nn.SpatialUpSampling(self.ratios[i+1]/self.ratios[i],
				    self.ratios[i+1]/self.ratios[i], 1, 2))
      local mul2 = nn.Mul2()
      mul2.weight[1] = -beta
      seq2:add(mul2)
      seq2:add(nn.Exp())
      --[[
      local mul2 = nn.Mul2()
      self.muls[#self.muls+1] = mul2
      seq2:add(mul2)
      seq2:add(nn.Tanh())
      --]]

      local parallel = nn.ParallelTable()
      parallel:add(seq1)
      --parallel:add(nn.Identity())
      parallel:add(seq2)
      
      self.transformers[i] = nn.Sequential()
      self.transformers[i]:add(parallel)
      self.transformers[i]:add(nn.CAddTable())
      self.transformers[i]:add(nn.Log2())
      local mul3 = nn.Mul2()
      mul3.weight[1] = -1./beta
      self.transformers[i]:add(mul3)
   end
   self.postprocessors = nn.ParallelTable()
   for i = 1,#self.ratios do
      --[[
      local seq = nn.Sequential()
      local mul = nn.Mul2()
      self.muls[#self.muls+1] = mul
      seq:add(mul)
      seq:add(nn.Tanh())
      --]]
      local seq = nn.Identity()
      self.postprocessors:add(seq)
   end

   -- common weight and gradWeight vectors
   if self.trainable then
      self.weight = torch.Tensor(#self.muls)
      self.gradWeight = torch.Tensor(#self.muls)
      for i = 1,#self.muls do
	 self.muls[i].weight = self.weight:narrow(1,i,1)
	 self.muls[i].gradWeight = self.gradWeight:narrow(1,i,1)
      end
   end
   self:reset()
   --self.muls[#self.muls].weight[1] = 1000
end

function CascadingAddTable:reset(stdv)
   --[[
   for i = 1,#self.muls do
      self.muls[i].weight[1] = 0.1
   end
   for i = 1,#self.ratios do
      self.postprocessors.modules[i].modules[1].weight[1] = 0.1 / (#self.ratios-i+1)
   end
   --]]
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
   self.outputBP[#input]:resizeAs(input[#input]):copy(input[#input])
   for i = #input-1,1,-1 do
      local r = self.ratios[i]
      local r2 = self.ratios[i+1]
      if ((math.mod(input[i]:size(1) * (r2-r), 2*r2) ~= 0) or
       (math.mod(input[i]:size(2) * (r2-r), 2*r2) ~= 0)) then
	 error('nn.CascadingAddTable: ratios and input sizes not compatible')
      end
      local dh = input[i]:size(1) * (r2-r) / (2*r2)
      local dw = input[i]:size(2) * (r2-r) / (2*r2)

      self.padders[i].pad_t = -dh
      self.padders[i].pad_b = -dh
      self.padders[i].pad_l = -dw
      self.padders[i].pad_r = -dw
      self.outputBP[i]:resizeAs(input[i]):copy(self.transformers[i]:forward({input[i], self.outputBP[i+1]}))
   end
   self.output = self.postprocessors:updateOutput(self.outputBP)
   return self.output
end

function CascadingAddTable:updateGradInput(input, gradOutput)
   if self.trainable then
      for i = 1,#self.muls do --todo this is dirty, see how it's done when sharing weights
	 self.muls[i].weight = self.weight:narrow(1,i,1)
	 self.muls[i].gradWeight = self.gradWeight:narrow(1,i,1)
      end
   end
   self.postprocessors:updateGradInput(self.outputBP, gradOutput)
   local lastGrad = torch.Tensor(self.postprocessors.gradInput[1]:size()):zero()
   for i = 1,#input-1 do
      self.transformers[i]:updateGradInput({input[i], self.output[i+1]},
					   self.postprocessors.gradInput[i]+lastGrad)
      self.gradInput[i]:resizeAs(input[i]):copy(self.transformers[i].gradInput[1])
      lastGrad = self.transformers[i].gradInput[2]
   end
   self.gradInput[#input]:resizeAs(input[#input]):copy(self.postprocessors.gradInput[#input]+lastGrad)
   return self.gradInput
end

function CascadingAddTable:accGradParameters(input, gradOutput, scale_)
   local scale = scale_*0.25
   if self.trainable then
      self.postprocessors:accGradParameters(self.outputBP, gradOutput, scale)
      for i = 1,#input-1 do
	 self.transformers[i]:accGradParameters({input[i], self.outputBP[i+1]},
						self.postprocessors.gradInput[i]+self.transformers[i].gradInput[2],
						scale)
      end
   end
end