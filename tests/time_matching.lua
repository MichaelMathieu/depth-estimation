require 'nnx'
require 'opticalflow_model'
require 'OutputExtractor'

wsize = 16
w = 320
h = 180
nfeats = 10

geometry = {}
geometry.maxh = wsize
geometry.maxw = wsize
geometry.layers = {{3,5,5,4},{4,5,5,4},{4,5,5,10}}

filter = getFilter(geometry)

matcher = nn.Sequential()
matcher:add(nn.SpatialMatching(wsize, wsize, false))
matcher:add(nn.Reshape(wsize*wsize, w-wsize+1-12, h-wsize+1-12))
--matcher:add(nn.OutputExtractor(false, 14))

time_filter = 0.
time_matcher = 0.
time_min = 0.

timer = torch.Timer()

for i = 1,10 do
   print(i)
   im1 = torch.randn(3, h, w)
   timer:reset()
   im1 = filter:forward(im1)
   time_filter = time_filter + timer:time()['real']
   im2 = torch.randn(3, h, w)
   im2 = filter:forward(im2)
   input = prepareInput(geometry, im1, im2)
   
   timer:reset()
   output = matcher:forward(input)
   time_matcher = time_matcher+ timer:time()['real']
   timer:reset()
   output2 = output:min(1)
   time_min = time_min + timer:time()['real']

   print(time_filter/i)
   print(time_matcher/i)
   print(time_min/i)
end

