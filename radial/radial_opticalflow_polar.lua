require 'torch'
require 'cartesian2polar'

function getRMax(networkp)
   return math.max(math.floor(networkp.hImg/2),math.floor(networkp.wImg/2))
end

function getP2CMaskOF(networkp, e2)
   local wPolar = networkp.wInput
   local hPolar = networkp.hInput-networkp.hKernel-networkp.hWin+2
   local kOutput = hPolar/networkp.hInput
   local wOutput = networkp.wImg*kOutput
   local hOutput = networkp.hImg*kOutput
   local new_e2 = e2*kOutput
   local newRMax = getRMax(networkp)*kOutput
   
   local mask = getP2CMask(wPolar, hPolar, wOutput, hOutput, new_e2[1], new_e2[2], newRMax)
   return mask
end