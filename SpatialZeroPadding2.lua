local SpatialZeroPadding2, parent = torch.class('nn.SpatialZeroPadding2', 'nn.Module')

function SpatialZeroPadding2:__init(pad_l, pad_r, pad_t, pad_b, dim1, dim2)
   parent.__init(self)
   self.dim1 = dim1 or 2
   self.dim2 = dim2 or 3
   self.pad_l = pad_l
   self.pad_r = pad_r or self.pad_l
   self.pad_t = pad_t or self.pad_l
   self.pad_b = pad_b or self.pad_l
end

function SpatialZeroPadding2:updateOutput(input)
   local h = input:size(self.dim1) + self.pad_t + self.pad_b
   local w = input:size(self.dim2) + self.pad_l + self.pad_r
   if w < 1 or h < 1 then error('input is too small') end
   local dims = input:size()
   dims[self.dim1] = h
   dims[self.dim2] = w
   self.output:resize(dims)
   self.output:zero()
   -- crop input if necessary
   local c_input = input
   if self.pad_t < 0 then c_input = c_input:narrow(self.dim1, 1 - self.pad_t, c_input:size(self.dim1) + self.pad_t) end
   if self.pad_b < 0 then c_input = c_input:narrow(self.dim1, 1, c_input:size(self.dim1) + self.pad_b) end
   if self.pad_l < 0 then c_input = c_input:narrow(self.dim2, 1 - self.pad_l, c_input:size(self.dim2) + self.pad_l) end
   if self.pad_r < 0 then c_input = c_input:narrow(self.dim2, 1, c_input:size(self.dim2) + self.pad_r) end
   -- crop outout if necessary
   local c_output = self.output
   if self.pad_t > 0 then c_output = c_output:narrow(self.dim1, 1 + self.pad_t, c_output:size(self.dim1) - self.pad_t) end
   if self.pad_b > 0 then c_output = c_output:narrow(self.dim1, 1, c_output:size(self.dim1) - self.pad_b) end
   if self.pad_l > 0 then c_output = c_output:narrow(self.dim2, 1 + self.pad_l, c_output:size(self.dim2) - self.pad_l) end
   if self.pad_r > 0 then c_output = c_output:narrow(self.dim2, 1, c_output:size(self.dim2) - self.pad_r) end
   -- copy input to output
   c_output:copy(c_input)
   return self.output
end

function SpatialZeroPadding2:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   -- crop gradInput if necessary
   local cg_input = self.gradInput
   if self.pad_t < 0 then cg_input = cg_input:narrow(self.dim1, 1 - self.pad_t, cg_input:size(self.dim1) + self.pad_t) end
   if self.pad_b < 0 then cg_input = cg_input:narrow(self.dim1, 1, cg_input:size(self.dim1) + self.pad_b) end
   if self.pad_l < 0 then cg_input = cg_input:narrow(self.dim2, 1 - self.pad_l, cg_input:size(self.dim2) + self.pad_l) end
   if self.pad_r < 0 then cg_input = cg_input:narrow(self.dim2, 1, cg_input:size(self.dim2) + self.pad_r) end
   -- crop gradOutout if necessary
   local cg_output = gradOutput
   if self.pad_t > 0 then cg_output = cg_output:narrow(self.dim1, 1 + self.pad_t, cg_output:size(self.dim1) - self.pad_t) end
   if self.pad_b > 0 then cg_output = cg_output:narrow(self.dim1, 1, cg_output:size(self.dim1) - self.pad_b) end
   if self.pad_l > 0 then cg_output = cg_output:narrow(self.dim2, 1 + self.pad_l, cg_output:size(self.dim2) - self.pad_l) end
   if self.pad_r > 0 then cg_output = cg_output:narrow(self.dim2, 1, cg_output:size(self.dim2) - self.pad_r) end
   -- copy gradOuput to gradInput
   cg_input:copy(cg_output)
   return self.gradInput
end
