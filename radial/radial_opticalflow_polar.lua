require 'torch'
require 'cartesian2polar'

function getRMax(networkp, e2)
   local h = networkp.hImg-1
   local w = networkp.wImg-1
   return math.floor(math.sqrt(math.max(math.max(e2[1]*e2[1]+e2[2]*e2[2],
						 (w-e2[1])*(w-e2[1])+e2[2]*e2[2]),
					math.max(e2[1]*e2[1]+(h-e2[2])*(h-e2[2]),
						 (w-e2[1])*(w-e2[1])+(h-e2[2])*(h-e2[2])))))

end

function getKOutput(networkp)
   local hPolar = networkp.hInput-networkp.hKernel-networkp.hWin+2
   local kOutput = hPolar/networkp.hInput
   return kOutput
end

function getP2CMaskOF(networkp, e2)
   local wPolar = networkp.wInput
   local hPolar = networkp.hInput-networkp.hKernel-networkp.hWin+2
   local kOutput = hPolar/networkp.hInput
   local wOutput = networkp.wImg*kOutput
   local hOutput = networkp.hImg*kOutput
   local new_e2 = e2*kOutput
   local newRMax = getRMax(networkp, e2)*kOutput
   
   local mask = getP2CMask(wPolar, hPolar, wOutput, hOutput, new_e2[1], new_e2[2], newRMax)
   return mask
end