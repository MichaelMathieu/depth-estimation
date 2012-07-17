require 'torch'
require 'image'
require 'radial_opticalflow_polar'
require 'inline'

function flow2depth(networkp, flow, center, kinfty)
   if center == nil then
      center = torch.Tensor(2)
      center[1] = flow:size(2)/2
      center[2] = flow:size(1)/2
   end
   local infty = getRMax(networkp.hImg, networkp.wImg, center)*kinfty
   local ret = torch.Tensor(flow:size()):zero()
   local confs = torch.Tensor(flow:size()):fill(1)

   local do_depths = inline.load [[
	 #define square(a) ((a)*(a))
	 
	 const void* idtensor = luaT_checktypename2id(L, "torch.FloatTensor");
	 THFloatTensor*  input = (THFloatTensor*)luaT_checkudata(L, 1, idtensor);
	 THFloatTensor* output = (THFloatTensor*)luaT_checkudata(L, 2, idtensor);
	 THFloatTensor*  confs = (THFloatTensor*)luaT_checkudata(L, 3, idtensor);
	 float xcenter = lua_tonumber(L, 4);
	 float ycenter = lua_tonumber(L, 5);
	 float infty   = lua_tonumber(L, 6);
	 
	 int h = input->size[0];
	 int w = input->size[1];
	 long* is = input->stride;
	 long* os = output->stride;
	 long* cs = confs->stride;
	 float*  input_p = THFloatTensor_data(input);
	 float* output_p = THFloatTensor_data(output);
	 float* confs_p = THFloatTensor_data(confs);

	 int i, j;
	 float d, flow;
	 for (i = 0; i < h; ++i) {
	    for (j = 0; j < w; ++j) {
	       d = sqrt(square((float)j-xcenter) + square((float)i-ycenter));
	       if (d > 10.0f) {
		  flow = input_p[i*is[0] + j*is[1] ];
		  if (flow < 0.1f) {
		     output_p[i*os[0] + j*os[1] ] = infty;
		  } else {
		     output_p[i*os[0] + j*os[1] ] = d/flow;
		  }
	       } else {
		  confs_p[i*cs[0] + j*cs[1] ] = 0.0f;
	       }
	    }
	 }
   ]]

   do_depths(flow, ret, confs, center[1], center[2], infty)
   
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