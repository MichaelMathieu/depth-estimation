require 'depth_estimation_api'

local masked = torch.Tensor()

while true do
   local im, last_im, w_im, flowx, flowy, mask = nextFrameDepth()

   masked:resize(3, flowx:size(1), flowx:size(2))
   masked[1]:copy(flowx:cmul(mask))
   masked[2]:copy(flowx:cmul(mask))
   masked[3]:copy(flowx:cmul(mask)+(-mask+1):mul(8))
   
   win=image.display{win=win, image={masked}, min=-8, max=8}
   win2 = image.display{win=win2, image={last_im, im, w_im}, min=0, max=1}
   win3 = image.display{image={last_im - im, w_im - im}, win=win3, min=-1, max=1}
end