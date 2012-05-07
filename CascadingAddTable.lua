require 'nnx'
require 'Mul2'
require 'Log'
local CascadingAddTable, parent = torch.class('nn.CascadingAddTable', 'nn.Module')
local CascadingAddTableSplit, parent = torch.class('nn.CascadingAddTableSplit', 'nn.Module')

function CascadingAddTable:__init(ratios, trainable, single_beta)
   parent.__init(self)
   self.ratios = ratios
   single_beta = single_beta or false
   if trainable == nil then self.trainable = true else self.trainable = trainable end
   self.gradInput = {}
   self.output = {}
   self.lastGradZero = torch.Tensor()
   for i = 1,#self.ratios do
      self.gradInput[i] = torch.Tensor()
      self.output[i] = torch.Tensor()
   end
   self.muls = {}
   self.muls_normalizers = {}
   self.padders = {}
   self.transformers = {}
   local beta = 1.
   for i = 1,#self.ratios-1 do
      local seq1 = nn.Sequential()
      local mul1 = nn.Mul2()
      self.muls[#self.muls+1] = mul1
      mul1.weight[1] = -beta
      --seq1:add(mul1)
      seq1:add(nn.SmartReshape(-1,{-2,-3}))
      
      local seq2 = nn.Sequential()
      self.padders[i] = nn.SpatialZeroPadding(0,0,0,0, 2, 3)
      seq2:add(self.padders[i])
      seq2:add(nn.SpatialReSamplingEx{rwidth=self.ratios[i+1]/self.ratios[i],
				      rheight=self.ratios[i+1]/self.ratios[i],
				      yDim=2, xDim=3, mode = 'average'})
      local mul2
      if single_beta then
	 mul2 = mul1:clone('weight', 'gradWeight')
      else
	 mul2 = nn.Mul2()
	 self.muls[#self.muls+1] = mul2
	 mul2.weight[1] = -beta
      end
      --seq2:add(mul2)
      seq2:add(nn.SmartReshape(-1,{-2,-3}))

      local parallel = nn.ParallelTable()
      parallel:add(seq1)
      parallel:add(seq2)
      
      self.transformers[i] = nn.Sequential()
      --self.transformers[i]:add(nn.Tic('cascad'..i))
      self.transformers[i]:add(parallel)
      --self.transformers[i]:add(nn.Toc('cascad'..i, 'resample/reshape'))
      self.transformers[i]:add(nn.CAddTable())
      --self.transformers[i]:add(nn.Toc('cascad'..i, 'caddtable'))
      local pow = nn.Power(-1./beta)
      self.muls_normalizers[#self.muls_normalizers+1] = {pow, mul1, mul2}
      --self.transformers[i]:add(pow)
      --self.transformers[i]:add(nn.Toc('cascad'..i, 'pow'))
   end
end

function CascadingAddTable:reset(stdv)
   for i = 1,#self.muls do
      self.muls[i]:reset(stdv)
   end
end

function CascadingAddTable:parameters()
   local weight = {}
   local gradWeight = {}
   if self.trainable then
      for i = 1,#self.muls do
	 table.insert(weight, self.muls[i].weight)
	 table.insert(gradWeight, self.muls[i].gradWeight)
      end
   end
   return weight, gradWeight
end

function CascadingAddTable:getWeight()
   local p, gp = self:parameters()
   local ret = torch.Tensor(#p)
   for i = 1,#p do
      ret[i] = p[i][1]
   end
   return ret
end

function CascadingAddTable:updateNormalizers()
   for i = 1,#self.muls_normalizers do
      local m = self.muls_normalizers[i]
      m[1].pow = -1. / math.sqrt(m[2].weight[1] * m[3].weight[1])
   end
end

function CascadingAddTable:updateParameters(learningRate)
   local params, gradParams = self:parameters()
   for i=1,#params do
      params[i]:add(-learningRate, gradParams[i])
   end
   self:updateNormalizers()
end

function CascadingAddTable:updateOutput(input)
   for i = 1,#input do
      if input[i]:nDimension() ~= 3 then
	 error('nn.CascadingAddTable: input must be a table of 3D-tensors (HxW) x Kh x Kw')
      end
   end
   if #input ~= #self.ratios then
      error('nn.CascadingAddTable: input and ratios must have the same size')
   end
   self.output[#input]:resizeAs(input[#input]):copy(input[#input])
   for i = #input-1,1,-1 do
      local r = self.ratios[i]
      local r2 = self.ratios[i+1]
      if ((math.mod(input[i]:size(2) * (r2-r), 2*r2) ~= 0) or
       (math.mod(input[i]:size(3) * (r2-r), 2*r2) ~= 0)) then
	 error('nn.CascadingAddTable: ratios and input sizes not compatible')
      end
      local dh = input[i]:size(2) * (r2-r) / (2*r2)
      local dw = input[i]:size(3) * (r2-r) / (2*r2)

      self.padders[i].pad_t = -dh
      self.padders[i].pad_b = -dh
      self.padders[i].pad_l = -dw
      self.padders[i].pad_r = -dw
      self.output[i]:resizeAs(input[i]):copy(self.transformers[i]:forward({input[i], self.output[i+1]}))
   end
   return self.output
end

function CascadingAddTable:updateGradInput(input, gradOutput)
   self.lastGradZero:resizeAs(gradOutput[1]):zero()
   local lastGrad = self.lastGradZero
   for i = 1,#input-1 do
      self.transformers[i]:updateGradInput({input[i], self.output[i+1]},
					   gradOutput[i]+lastGrad)
      self.gradInput[i]:resizeAs(input[i]):copy(self.transformers[i].gradInput[1])
      lastGrad = self.transformers[i].gradInput[2]
   end
   self.gradInput[#input]:resizeAs(input[#input]):copy(gradOutput[#input]+lastGrad)
   if sys.isNaN(self.gradInput[1]:sum()) or sys.isNaN(self.gradInput[2]:sum()) then
      print(self.transformers[1].modules[2].output)
      print(self.transformers[1].modules[3].gradInput)
      print(self.transformers[1].modules[4].gradInput)
      error('stopped in CascabingAddTable')
   end
   return self.gradInput
end

function CascadingAddTable:accGradParameters(input, gradOutput, scale)
   local lastGrad = self.lastGradZero
   if self.trainable then
      for i = 1,#input-1 do
	 self.transformers[i]:accGradParameters({input[i], self.output[i+1]},
						gradOutput[i]+lastGrad, scale)
	 lastGrad = self.transformers[i].gradInput[2]
      end
   end
end