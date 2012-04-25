package.path = "./?.lua;/home/myrhev/local/share/torch/lua/?.lua;/home/myrhev/local/share/torch/lua/?/init.lua;/home/myrhev/local/lib/torch/?.lua;/home/myrhev/local/lib/torch/?/init.lua"
package.cpath = "./?.so;/home/myrhev/local/lib/torch/?.so;/home/myrhev/local/lib/torch/loadall.so"

require 'torch'
--require 'opticalflow_model'

function nextFrameDepth()
   collectgarbage()
   return torch.FloatTensor(180, 320):fill(0)
end