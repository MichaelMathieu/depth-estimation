require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
require 'image'
require 'groundtruth_opticalflow'
require 'opticalflow_model_io'
require 'score_opticalflow'
require 'xlua'
require 'download_model'
require 'depth_estimation_api'
require 'image_loader'
require 'inline'

op = xlua.OptionParser('%prog [options]')
op:option{'-i', '--input-model', action='store', dest='input_model', default=nil,
	  help='Trained convnet, this option isn\'t used if -dldir is used'}
op:option{'-dldir', '--download-dir', action='store', dest='download_dir', default=nil,
	  help='scp command to the models folder (eg. mfm352@access.cims.nyu.edu:depth-estimation/models)'}
op:option{'-fi', '--first-image', action='store', dest='first_image', default=0,
	  help='Index of first image used'}
op:option{'-d', '--delta', action='store', dest='delta', default=1,
	  help='Delta between two consecutive frames'}
op:option{'-ni', '--num-input-images', action='store', dest='num_input_images',
	  default=10, help='Number of annotated images used'}
op:option{'-rd', '--root-directory', action='store', dest='root_directory',
	  default='./data/', help='Root dataset directory'}
op:option{'-o', '--output-dir', action='store', dest='output_dir',
	  default='video.mp4', help='Output directory'}
op:option{'-c', '--camera', action='store', dest='camera',
	  default='ardrone', help='Camera (gopro | ardrone)'}
op:option{'-gt', '--groundtruth', action='store', dest='groundtruth',
	  default=nil, help='Groundtruth full path'}
opt=op:parse()

if opt.input_model then input_model = opt.input_model end
if opt.download_dir then input_model = downloadModel(opt.download_dir) end

local output_dir = opt.output_dir

local correction = {}
correction.motion_correction = 'sfm'
if opt.camera == 'gopro' then
   correction.wImg = 1280
   correction.hImg = 720
   correction.bad_image_threshold = 0.2
   correction.K = torch.FloatTensor(3,3):zero()
   correction.K[1][1] = 602.663208
   correction.K[2][2] = 603.193289
   correction.K[1][3] = 641.455200
   correction.K[2][3] = 344.950836
   correction.K[3][3] = 1.0
   correction.Khalf = correction.K:clone()*0.25
   correction.Khalf[3][3] = 1.0
   correction.distP = torch.FloatTensor(5)
   correction.distP[1] = -0.355740
   correction.distP[2] = 0.142684
   correction.distP[3] = 0.000469
   correction.distP[4] = 0.000801
   correction.distP[5] = -0.027673
elseif opt.camera == 'ardrone' then
   correction.wImg = 640
   correction.hImg = 480
   correction.bad_image_threshold = 0.2
   correction.K = torch.FloatTensor(3,3):zero()
   correction.K[1][1] = 293.824707
   correction.K[2][2] = 310.435730
   correction.K[1][3] = 300.631012
   correction.K[2][3] = 251.624924
   correction.K[3][3] = 1.0
   correction.Khalf = correction.K:clone()*0.5
   correction.Khalf[3][3] = 1.0
   correction.distP = torch.FloatTensor(5)
   correction.distP[1] = -0.379940
   correction.distP[2] = 0.212737
   correction.distP[3] = 0.003098
   correction.distP[4] = 0.000870
   correction.distP[5] = -0.069770
end

input_model = input_model or 'model'
local loaded
if opt.camera == 'gopro' then
   loaded = loadModel(input_model, true, false, 320, 180)
else
   loaded = loadModel(input_model, true, false)
end
local filter = loaded.filter
local model = loaded.model
model.modules[5] = nn.SoftMax()
--model.modules[2] = nn.SpatialRadialMatching(12)
--model.modules[3] = nn.Identity()
--model.modules[4] = nn.Identity()
--model.modules[5] = nn.Identity()
--model.modules[6] = nn.Identity()
print(model)
local geometry = loaded.geometry
geometry.prefilter = false
geometry.training_mode = false
geometry.motion_correction = correction.motion_correction
geometry.output_extraction_method = 'max'

local learning = {}
learning.delta = 1
learning.groundtruth = 'liu'

function scoreOpticalFlow(flow, confs, gt)
   local y = (flow[1] - gt[1]):cmul(confs)
   local x = (flow[2] - gt[2]):cmul(confs)
   local nConfs = confs:sum()
   local nErrs = ((x:abs() + y:abs()):gt(1)):sum()
   local colorflowx = torch.Tensor(3, flow:size(2), flow:size(3)):zero()
   local colorflowy = torch.Tensor(3, flow:size(2), flow:size(3)):zero()
   colorflowx[1]:copy(flow[1])
   colorflowy[1]:copy(flow[2])
   colorflowx[2] = -confs+1
   colorflowy[2] = -confs+1
   return nConfs, nErrs
end

local imagenames = {}
for i = 1,opt.num_input_images do
   imagenames[i] = string.format('%09d', opt.first_image+(i-1)*opt.delta)
end
images = {}
gt = {}
local last_ims = {}
local warped_ims = {}
local warped_masks = {}
local ims = {}
local gts = {}
loader = ImageLoader
loader:init(geometry, opt.root_directory, opt.first_image, 1)

local last_im = loader:getNextFrame()
local last_im = image.scale(last_im, correction.wImg, correction.hImg)
if opt.camera ~= 'gopro' then
   last_im = sfm2.undistortImage(last_im, correction.K, correction.distP)
end
local last_im_scaled = image.scale(last_im, geometry.wImg, geometry.hImg)

local h = geometry.hImg
local w = geometry.wImg

function radial(geometry, flow, mh, mw)
   mh = mh or flow:size(2)/2
   mw = mw or flow:size(3)/2
   local ret = torch.Tensor(flow:size(2), flow:size(3)):zero()
   local conf = torch.Tensor(flow:size(2), flow:size(3)):zero()
   local infty = geometry.wImg/2
   
   local f = inline.load [[
	 #define min(a,b) (((a)<(b))?(a):(b))
	 const void* idfloat = luaT_checktypename2id(L, "torch.FloatTensor");
	 THFloatTensor* flow = (THFloatTensor*)luaT_checkudata(L, 1, idfloat);
	 float mh = lua_tonumber(L, 2);
	 float mw = lua_tonumber(L, 3);
	 THFloatTensor* ret = (THFloatTensor*)luaT_checkudata(L, 4, idfloat);
	 THFloatTensor* conf = (THFloatTensor*)luaT_checkudata(L, 5, idfloat);
	 float infty = lua_tonumber(L, 6);

	 int h = flow->size[1];
	 int w = flow->size[2];
	 long* fs = flow->stride;
	 long* rs = ret->stride;
	 long* cs = conf->stride;
	 float* flow_p = THFloatTensor_data(flow);
	 float* ret_p  = THFloatTensor_data(ret);
	 float* conf_p = THFloatTensor_data(conf);

	 float px, py, dx, dy, pn, dn;
	 int i, j;
	 for (i = 0; i < h; ++i) {
	    for (j = 0; j < w; ++j) {
	       py = (float)i-mh;
	       px = (float)j-mw;
	       pn = sqrt(px*px+py*py);
	       dy = flow_p[i*fs[1] + j*fs[2] ];
	       dx = flow_p[fs[0] + i*fs[1] + j*fs[2] ];
	       dn = sqrt(dx*dx+dy*dy);
	       if (dn >= 0.2f) {
		  ret_p[i*rs[0] + j*rs[1] ] = min(pn/dn, infty);
		  if (px*dx+dy*dy > 0.125f)
		     conf_p[i*cs[0] + j*cs[1] ] = 1.0f;
	       } else {
		  conf_p[i*cs[0] + j*cs[1] ] = 1.0f;
		  ret_p[i*rs[0] + j*rs[1] ] = infty;
	       }
	    }
	 }
   ]]

   f(flow, mh, mw, ret, conf, infty)
	       
   --[[
   local p = torch.Tensor(2)
   local d
   for i = 1,flow:size(2) do
      for j = 1,flow:size(3) do
	 p[1] = i-mh
	 p[2] = j-mw
	 d = flow[{{},i,j}]
	 local dn = d:norm()
	 local pn = p:norm()
	 if dn >= 1.0 then
	    if d:dot(p)/(dn*pn) > 0.125 then
	       conf[i][j] = 1
	       ret[i][j] = pn/dn
	    end
	 else
	    conf[i][j] = 1
	    ret[i][j] = geometry.wImg/2
	 end
      end
   end
   --]]
   return ret, conf
end

function saturate(t, min, max)
   local mask = torch.FloatTensor(t:size()):copy(t:gt(min))
   t = t:cmul(mask) + (-mask+1)*min
   mask = torch.FloatTensor(t:size()):copy(t:lt(max))
   t = t:cmul(mask) + (-mask+1)*max
   return t
end

function depth2color(depthmap, confs)
   saturate(depthmap, 0, 1)
   local tohsv = torch.Tensor(3, depthmap:size(1), depthmap:size(2))
   tohsv[1]:copy(depthmap/1.3)
   --tohsv[2]:copy(confs*0.99)
   tohsv[2]:fill(0.99)
   tohsv[3]:copy(confs*0.5)
   return image.hsl2rgb(tohsv)
end

function displayResult(disp, flow, flowproc, mask, mask_entropy, ih, h, w, use_rad_conf)
   local dispxflow = disp[{{}, {  ih*h+1,(ih+1)*h}, {    1,  w}}]
   local dispyflow = disp[{{}, {  ih*h+1,(ih+1)*h}, {  w+1,2*w}}]
   local disprflow = disp[{{}, {  ih*h+1,(ih+1)*h}, {2*w+1,3*w}}]
   dispxflow[1]:copy((flow[2]/16+0.5):cmul(mask))
   dispxflow[2]:copy((flow[2]/16+0.5):cmul(mask))
   dispxflow[3]:copy((flow[2]/16+0.5):cmul(mask) + (-mask+1)*0.5)
   dispyflow[1]:copy((flow[1]/16+0.5):cmul(mask))
   dispyflow[2]:copy((flow[1]/16+0.5):cmul(mask))
   dispyflow[3]:copy((flow[1]/16+0.5):cmul(mask) + (-mask+1)*0.5)
   local rad, radconf = radial(geometry, flowproc, correction.Khalf[2][3], correction.Khalf[1][3])
   if not use_rad_conf then
      radconf:fill(1)
   end
   radconf = radconf:cmul(mask):cmul(mask_entropy)
   rad = (rad/200):cmul(radconf)
   --disprflow[1]:copy(rad)
   --disprflow[2]:copy(rad)
   --disprflow[3]:copy(rad + (-radconf+1)*0.5)
   disprflow:copy(depth2color(rad, radconf))
   --win42 = image.display{image=depth2color(rad, radconf), win=win42}
end

local maskmiddle = torch.Tensor(h, w):fill(1)
local rmaskmiddle = 25
local rmaskmiddlelarge = 0
for i = 1,h do
   for j = 1,w do
      local ii = i-h/2
      local jj = j-w/2
      local d = math.sqrt(ii*ii+jj*jj)
      if d < rmaskmiddle then
	 maskmiddle[i][j] = 0
      elseif d < (rmaskmiddle+rmaskmiddlelarge) then
	 maskmiddle[i][j] = (d-rmaskmiddle)/rmaskmiddlelarge
      end
   end
end

os.execute('mkdir -p '..output_dir..' && rm -f '..output_dir..'/*.png')
for i = 1,opt.num_input_images do
   local im = loader:getNextFrame()
   local im = image.scale(im, correction.wImg, correction.hImg)
   if opt.camera ~= 'gopro' then
      im = sfm2.undistortImage(im, correction.K, correction.distP)
   end
   local R, T, nFound, nInliers = sfm2.getEgoMotion(last_im, im, correction.K, 400)
   im_scaled = image.scale(im, geometry.wImg, geometry.hImg)
   local warped_im, mask = sfm2.removeEgoMotion(last_im_scaled, correction.Khalf, R)

   local groundtruth = nil
   if opt.groundtruth then
      groundtruth = image.load(string.format("%s%09d.png", opt.groundtruth, loader.nextFrame))
      groundtruth = groundtruth[{{1,2},{},{}}]*256 - 128
   end
   
   local poutput, poutput2, mask_entropy
   if (nInliers/nFound < 0.2) then
      mask = torch.Tensor(h, w):zero()
      mask_entropy = torch.Tensor(h, w):zero()
      poutput = {}
      poutput2 = {}
      poutput.full = torch.Tensor(2, h, w):zero()
      poutput2.full = torch.Tensor(2, h, w):zero()
   else
      k = 2
      
      --[[
      local input = {warped_im, im_scaled}
      for i = -2,2 do
	 for j = -2,2 do
	    model.modules[2].xm = (warped_im:size(2)/2-12) + i
	    model.modules[2].ym = (warped_im:size(2)/2-12) + j
	    local moutput = model:forward(input)
	    local b = moutput:size(3)
	    moutput = moutput:reshape(moutput:size(1), moutput:size(2), b*moutput:size(4))
	    local m
	    m, poutput = moutput:max(3)
	    poutput = poutput:squeeze()
	    poutput = torch.Tensor(poutput:size()):copy(poutput)
	    poutput = poutput - ((poutput-1)/b):floor()*b
	    win33 = image.display{image = poutput, win = win33}
	    print(i,j)
	 end
      end
      
      if false then
      local nfeats = 256*3
      local input1 = warped_im:unfold(2, 16, 1):unfold(3, 16, 1)
      local h1 = input1:size(2)
      local w1 = input1:size(3)
      local input1b = torch.Tensor(nfeats, h1, w1)
      for i = 1,h1 do
	 for j = 1,w1 do
	    input1b:select(2,i):select(2,j):copy(input1:select(2,i):select(2,j):reshape(nfeats))
	 end
      end
      local input2 = im_scaled:unfold(2, 16, 1):unfold(3, 16, 1)
      local h2 = input2:size(2)
      local w2 = input2:size(3)
      local input2b = torch.Tensor(nfeats, h2, w2)
      for i = 1,h2 do
	 for j = 1,w2 do
	    input2b:select(2,i):select(2,j):copy(input2:select(2,i):select(2,j):reshape(nfeats))
	 end
      end
      local input = {input1, input2}
      local moutput = nn.SpatialRadialMatching(12):forward(input)
      end
      --]]
      
      local input = prepareInput(geometry, warped_im, im_scaled)
      geometry.output_extraction_method = 'max'
      local moutput = model:forward(input)
      poutput = processOutput(geometry, moutput, true, nil)
      win33 = image.display{image = poutput.full, win = win33}
      geometry.output_extraction_method = 'mean'
      poutput2 = processOutput(geometry, moutput, true, nil)
      
      local poutput_mask = processOutput(geometry, moutput, true, k)
      
      enlargeMask(mask,
		  math.ceil((geometry.wImg-poutput.y:size(2))/2),
		  math.ceil((geometry.hImg-poutput.y:size(1))/2))
      mask_entropy = poutput_mask.full_confidences
      --mask_entropy = torch.Tensor(h, w):fill(1)
      mask_entropy:cmul(maskmiddle)
      poutput.full[1]:cmul(mask)
      poutput.full[2]:cmul(mask)
      poutput2.full[1]:cmul(mask)
      poutput2.full[2]:cmul(mask)
   end

   --local disp = torch.Tensor(3, 3*h, 3*w):zero()
local disp = torch.Tensor(3, 2*h, 3*w):zero()
   disp[{{}, {  1,  h}, {  1  ,  w  }}]:copy(last_im_scaled)
   disp[{{}, {  1,  h}, {  w+1,2*w}}]:copy(warped_im)
   disp[{{}, {  1,  h}, {2*w+1,3*w}}]:copy(im_scaled)

   --print(poutput:size())
   local mask_total = mask:clone():cmul(mask_entropy)
   poutput.postprocessed = postProcessImage(poutput.full, mask_total, 3, 'med')
   --poutput2.postprocessed = postProcessImage(poutput2.full, mask_total, 3, 'med')
   
   print(mask:size())
   print(poutput.full:size())
   print(mask_entropy:size())
   displayResult(disp, poutput.full, poutput.postprocessed, mask, mask_entropy, 1, h, w, true)
   --displayResult(disp, poutput2.full, poutput2.postprocessed, mask, mask_entropy,
   --1, h, w, true)
   --displayResult(disp, groundtruth, mask, torch.Tensor(mask:size()):fill(1), 2, h, w, false)
   
   win_im = image.display{image=disp, win=win_im}
   image.save(string.format(output_dir..'/%09d.png', i-1), disp)
   
   last_im:copy(im)
   last_im_scaled:copy(im_scaled)
end
os.execute('cd '..output_dir..' && ffmpeg -sameq -r 10 -i %09d.png video.mp4')
 
--[[

local scores = {}

os.execute('mkdir -p dump_tmp && rm dump_tmp/*.png')
for i = 2,#images do
   local input = prepareInput(geometry, images[i-1], images[i])
   local output = model:forward(input)
   output = processOutput(geometry, output)
   if opt.post_process_winsize ~= 1 then
      output = postProcessImage(output.full, opt.post_process_winsize)
   else
      output = output.full
   end
   local im = flow2hsv(geometry, output)
   local gthsv = flow2hsv(geometry, gt[i-1])

   nGood, nNear, nBad, d, meanDst, stdDst = evalOpticalflow(output, gt[i-1])
   table.insert(scores, {nGood, nNear, nBad, d, meanDst, stdDst})
   print{nGood, nNear, nBad, d, meanDst, stdDst}
   
   local im2 = torch.Tensor(im:size(1), 2*im:size(2), 2*im:size(3))
   im2:sub(1,im2:size(1),            1, im:size(2),            1, im:size(3)):copy(images[i-1])
   im2:sub(1,im2:size(1),            1, im:size(2), im:size(3)+1,im2:size(3)):copy(images[i])
   im2:sub(1,im2:size(1), im:size(2)+1,im2:size(2),            1, im:size(3)):copy(im)  
   im2:sub(1,im2:size(1), im:size(2)+1,im2:size(2), im:size(3)+1,im2:size(3)):copy(gthsv)
   image.save(string.format('dump_tmp/%09d.png', i-1), im2)
   --a = image.display{image=flow2hsv(geometry, output.full), win=a, min=0, max=1}
   --a = image.display{image=output.full, win=a, min=1, max=17}
end
os.execute('cd dump_tmp && ffmpeg -sameq -r 10 -i %09d.png ' .. opt.output_video)

torch.save('last_scores', scores)
--]]