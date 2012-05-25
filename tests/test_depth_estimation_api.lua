torch.setdefaulttensortype('torch.FloatTensor')

require 'depth_estimation_api'
require 'opencv'
require 'sys'
require 'xlua'
require 'download_model'

op = xlua.OptionParser('%prog [options]')
op:option{'-i', '--input-model', action='store', dest='input_model', default=nil,
	  help='Trained convnet, this option isn\'t used if -dldir is used'}
op:option{'-dldir', '--download-dir', action='store', dest='download_dir', default=nil,
	  help='scp command to the models folder (eg. mfm352@access.cims.nyu.edu:depth-estimation/models)'}
--op:option{'-rd', '--root-directory', action='store', dest='root_directory',
--	  default='./data/', help='Root dataset directory'}
opt=op:parse()

if opt.input_model then input_model = opt.input_model end
if opt.download_dir then input_model = downloadModel(opt.download_dir) end


local xmasked = torch.Tensor()
local ymasked = torch.Tensor()

debug_display = true

while true do
   sys.tic()
   local im, last_im, w_im, flowx, flowy, mask = nextFrameDepth()
   print("FPS: " .. 1./sys.toc())

   xmasked:resize(3, flowx:size(1), flowx:size(2))
   xmasked[1]:copy(flowx:cmul(mask))
   xmasked[2]:copy(flowx:cmul(mask))
   xmasked[3]:copy(flowx:cmul(mask)+(-mask+1):mul(8))

   ymasked:resize(3, flowy:size(1), flowy:size(2))
   ymasked[1]:copy(flowy:cmul(mask))
   ymasked[2]:copy(flowy:cmul(mask))
   ymasked[3]:copy(flowy:cmul(mask)+(-mask+1):mul(8))

   
   win=image.display{win=win, image={xmasked, ymasked}, min=-8, max=8}
   win2 = image.display{win=win2, image={last_im, im, w_im}, min=0, max=1}
   win3 = image.display{image={last_im - im, w_im - im}, win=win3, min=-1, max=1}
end