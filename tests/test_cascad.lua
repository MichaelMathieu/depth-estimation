require 'CascadingAddTable'

local function test_CascadingAddTable()
   local hfeats = 8*math.random(1,3)
   local wfeats = 8*math.random(1,3)
   local nratios = math.random(1,4)
   local ratios = {}
   for i = 1,nratios do
      ratios[i] = math.pow(2, i-1)
   end
   local iwidth = math.random(1,5)
   local iheight = math.random(1,5)

   local casc = nn.CascadingAddTable(ratios, true)
   --print(ratios)
   local module = nn.Sequential()
   module:add(nn.SplitTable(1))
   module:add(casc)
   module:add(nn.JoinTable(1))

   input = torch.randn(nratios, hfeats, wfeats, iheight, iwidth)

   local err = nn.Jacobian.testJacobian(module, input, 0.1, 2)
   local precision = 1e-5
   assert(math.abs(err) < precision)

   if nratios > 1 then
      local err = nn.Jacobian.testJacobianParameters(module, input,
						     module.modules[2].weight,
						     module.modules[2].gradWeight,
						     0.1, 2)
   end
   assert(math.abs(err) < precision)


   local ferr, berr = nn.Jacobian.testIO(module, input)
   assert(ferr == 0)
   assert(berr == 0)
end

local ntests = 10
xlua.progress(0, ntests)
for i = 1,10 do
   xlua.progress(i, ntests)
   test_CascadingAddTable()
end