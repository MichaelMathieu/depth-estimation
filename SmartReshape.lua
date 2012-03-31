local SmartReshape, parent = torch.class('nn.SmartReshape', 'nn.Module')

function SmartReshape:__init(...)
   parent.__init(self)
   local n = select('#', ...)
   local sizes = {}
   for i = 1,n do
      sizes[i] = select(i, ...)
   end
   self.isize = {}

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
   
   self.size = function(input)
		  local ret = torch.LongStorage(n)
		  for i = 1,n do
		     ret[i] = getSize(sizes[i], input)
		  end
		  return ret
	       end
end

function SmartReshape:updateOutput(input)
   input = input:contiguous()
   self.output:set(input):resize(self.size(input))
   return self.output
end

function SmartReshape:updateGradInput(input, gradOutput)
   gradOutput = gradOutput:contiguous()
   self.gradInput:set(gradOutput):resizeAs(input)
   return self.gradInput
end
