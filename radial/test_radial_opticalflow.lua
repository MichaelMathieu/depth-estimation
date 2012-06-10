require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
package.path = "./?.lua;../?.lua;" .. package.path
package.cpath = "./?.so;../?.so;" .. package.cpath
require 'xlua'
require 'sys'
require 'openmp'
require 'image'
require 'radial_opticalflow_data'
require 'radial_opticalflow_network'
require 'radial_opticalflow_filtering'

torch.manualSeed(1)

op = xlua.OptionParser('%prog [options]')
-- general
op:option{'-nt', '--num-threads', action='store', dest='nThreads', default=2,
	  help='Number of threads used'}

-- input
op:option{'-rd', '--root-directory', action='store', dest='root_directory',
	  default='data/no-risk/part1/', help='Root dataset directory'}
op:option{'-fi', '--first-image', action='store', dest='first_image', default=0,
	  help='Index of first image used'}
op:option{'-d', '--delta', action='store', dest='delta', default=1,
	  help='Delta between two consecutive frames'}
op:option{'-ni', '--num-input-images', action='store', dest='num_input_images',
	  default=10, help='Number of annotated images used'}
op:option{'-cal', '--caligration', dest='calibration_file', default='rectified_gopro.cal',
	  action='store', help='Calibration parameters file'}
op:option{'-i', '--network', action='store', dest='network_file',
	  default=nil, help='Path to the saved network'}

opt = op:parse()
opt.nTherads = tonumber(opt.nThreads)
opt.first_image = tonumber(opt.first_image)
opt.delta = tonumber(opt.delta)
opt.num_input_images = tonumber(opt.num_input_images)
if opt.root_directory:sub(-1) ~= '/' then opt.root_directory = opt.root_directory .. '/' end

openmp.setDefaultNumThreads(opt.nThreads)

local network, networkp = loadTesterNetwork(opt.network_file)

local calibrationp = torch.load(opt.calibration_file)

local datap = {}
datap.first_image = opt.first_image
datap.delta = opt.delta
datap.n_images = opt.num_input_images

local raw_data = load_training_raw_data(opt.root_directory, networkp, nil,
					datap, calibrationp)

for i = 1,datap.n_images do
   --local iImg = (i-1)*datap.delta+datap.first_image
   local output = network:forward({raw_data.polar_prev_images[i],
				   raw_data.polar_images[i]})
   local _, idx = output:min(3)
   idx = torch.Tensor(idx:squeeze():size()):copy(idx)
   idx:add(-1)

   local rmax = math.max(math.floor(networkp.hImg/2),math.floor(networkp.wImg/2))
   local p2cmask = getP2CMask(idx:size(2), idx:size(1),
			      (1-(networkp.wKernel-1)/networkp.wInput)*networkp.wImg,
			      (1-(networkp.hKernel-1)/networkp.hInput)*networkp.hImg,
			      raw_data.e2[i][1], raw_data.e2[i][2], rmax)
   local cartidx = cartesian2polar(idx, p2cmask)

   win = image.display{image=cartidx, win=win}
end