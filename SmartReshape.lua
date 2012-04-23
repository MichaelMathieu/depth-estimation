local SmartReshape, parent = torch.class('nn.SmartReshape', 'nn.Module')

function SmartReshape:__init(...)
   parent.__init(self)
   local n = select('#', ...)
   self.sizes = {}
   for i = 1,n do
      self.sizes[i] = select(i, ...)
   end

   function getSize(code, input)
      if type(code) == 'number' then
	 if code >= 0 then
	    return code
	 else
	    return input:size(-code)
	 end
      else
	 local ret = 1
	 for j = 1,#code do
	    ret = ret * getSize(code[j], input)
	 end
	 return ret
      end
   end
   
   function self:size(input)
		  local ret = torch.LongStorage(n)
		  for i = 1,n do
		     ret[i] = getSize(self.sizes[i], input)
		  end
		  return ret
	       end
end

function SmartReshape:describe(sizes)
   if #sizes == 0 then
      return '{}'
   else
      ret = '{'
      for i = 1,#sizes do
	 if type(sizes[i]) == 'number' then
	    ret = ret .. sizes[i] .. ' '
	 else
	    ret = ret .. self:describe(sizes[i]) .. ' '
	 end
      end
      return ret:sub(1,-2) .. '}'
   end
end

function SmartReshape:updateOutput(input)
   input = input:contiguous()
   local size = self:size(input)
   if torch.LongTensor():set(size):prod(1)[1] ~= torch.LongTensor():set(input:size()):prod(1)[1] then
      error("SmartReshape: number of elements don't match.\n  input:size()=\n"
	    .. input:size():__tostring__() .. "\n  self.sizes=\n"
      .. self:describe(self.sizes))
   end
   self.output:set(input):resize(size)
   return self.output
end

function SmartReshape:updateGradInput(input, gradOutput)
   gradOutput = gradOutput:contiguous()
   self.gradInput:set(gradOutput):resizeAs(input)
   return self.gradInput
end
