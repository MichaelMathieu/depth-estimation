
function flow2pol(geometry, y, x)
   y, x = onebased2centered(geometry, y, x)
   local ang = math.atan2(y, x)
   local norm = math.sqrt(x*x+y*y)
   return ang, norm
end

function evalOpticalflow(output, gt)
   local diff = (output - gt):abs()
   diff = diff[1]+diff[2]
   local nGood = 0
   local nNear = 0
   local nBad = 0
   for i = 1,diff:size(1) do
      for j = 1,diff:size(2) do
	 if diff[i][j] == 0 then
	    nGood = nGood + 1
	 elseif diff[i][j] == 1 then
	    nNear = nNear + 1
	 else
	    nBad = nBad + 1
	 end
      end
   end

   local meanDst = 0.0
   local meanDst2 = 0.0
   local d = 0.0
   local n = 0

   for i = 18,output:size(2)-17 do
      for j = 18,output:size(3)-17 do
	 local y, x = onebased2centered(geometry, output[1][i][j], output[2][i][j])
	 local ygt, xgt = onebased2centered(geometry, gt[1][i][j], gt[2][i][j])
	 y = y-ygt
	 x = x-xgt
	 local n2 = x*x+y*y
	 d = d + n2

	 meanDst = meanDst + math.sqrt(n2)
	 meanDst2 = meanDst2 + n2
	 n = n + 1
      end
   end

   d = math.sqrt(d/n)
   meanDst = meanDst / n
   meanDst2 = meanDst2 / n
   local stdDst = math.sqrt(meanDst2 - meanDst*meanDst)

   return nGood, nNear, nBad, d, meanDst, stdDst
end

function flow2hsv(geometry, flow)
   local todisplay = torch.Tensor(3, flow:size(2), flow:size(3))
   for i = 1,flow:size(2) do
      for j = 1,flow:size(3) do
	 local ang, norm = flow2pol(geometry, flow[1][i][j], flow[2][i][j])
	 todisplay[1][i][j] = ang/(math.pi*2.0)
	 todisplay[2][i][j] = 1.0
	 todisplay[3][i][j] = norm/math.max(geometry.maxh/2, geometry.maxw/2)
      end
   end
   return image.hsl2rgb(todisplay)
end
