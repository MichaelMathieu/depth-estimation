torch.setdefaulttensortype('torch.FloatTensor')

require 'depth_estimation_api'
require 'opencv'
require 'sys'


local masked = torch.Tensor()

debug_display = true

while true do
   sys.tic()
   local im, last_im, w_im, flowx, flowy, mask = nextFrameDepth()
   print("FPS: " .. 1./sys.toc())

   masked:resize(3, flowx:size(1), flowx:size(2))
   masked[1]:copy(flowx:cmul(mask))
   masked[2]:copy(flowx:cmul(mask))
   masked[3]:copy(flowx:cmul(mask)+(-mask+1):mul(8))
   
   win  = opencv.display{win=win, image={masked}, min=-8, max=8}
   win2 = opencv.display{win=win2, image={last_im, im, w_im}, min=0, max=1}
   win3 = opencv.display{image={last_im - im, w_im - im}, win=win3, min=-1, max=1}
end
