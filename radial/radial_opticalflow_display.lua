require 'torch'
require 'image'
require 'radial_opticalflow_polar'

function flow2depth(networkp, flow, center, kinfty)
   if center == nil then
      center = torch.Tensor(2)
      center[1] = flow:size(2)/2
      center[2] = flow:size(1)/2
   end
   local infty = getRMax(networkp.hImg, networkp.wImg, center)*kinfty
   local ret = torch.Tensor(flow:size()):zero()
   local confs = torch.Tensor(flow:size()):fill(1)
   for i = 1,flow:size(1) do
      for j = 1,flow:size(2) do
	 local d = math.sqrt((j-center[1])*(j-center[1])+(i-center[2])*(i-center[2]))
	 if d > 10 then
	    if flow[i][j] < 1.0 then
	       ret[i][j] = infty
	    else
	       ret[i][j] = d/flow[i][j]
	    end
	 else
	    confs[i][j] = 0
	 end
      end
   end
   return ret/infty, confs
end

function saturate(t, min, max)
   local mask = torch.FloatTensor(t:size()):copy(t:gt(min))
   t = t:cmul(mask) + (-mask+1)*min
   mask = torch.Tensor(t:size()):copy(t:lt(max))
   t = t:cmul(mask) + (-mask+1)*max
   return t
end

function depth2color(depth, confs)
   local k = 1.5
   depth = saturate(depth, 0, 1)
   local tohsv = torch.Tensor(3, depth:size(1), depth:size(2))
   tohsv[1]:copy(depth/k)
   tohsv[2]:fill(1.0)
   tohsv[3]:copy(confs*0.5)
   return image.hsl2rgb(tohsv)
end

function padOutput(networkp, im)
   local output = torch.Tensor(im:size(1), networkp.hImg, networkp.wImg):zero()
   local dh = output:size(2) - im:size(2)
   local dw = output:size(3) - im:size(3)
   output:sub(1, im:size(1),
	      math.ceil(dh/2), im:size(2)+math.ceil(dh/2)-1,
	      math.ceil(dw/2), im:size(3)+math.ceil(dw/2)-1):copy(im)
   return output
end