require 'torch'

function filterOutputTrainer(output, threshold)
   local m, idx = output:max(1)
   idx:add(-1)
   idx = idx:squeeze()
   m = m:squeeze()
   return idx, math.exp(m) >= threshold
end