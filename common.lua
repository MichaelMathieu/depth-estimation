require 'torch'
require 'xlua'

function round(x)
   return math.floor(x+0.5)
end

function modProgress(i, max, mod)
   if math.mod(i, mod) == 0 then
      xlua.progress(i, max)
   end
end

function randInt(a, b) --a included, b excluded
   return math.floor(torch.uniform(a, b))
end

function randomPermutation(n)
   local ret = torch.Tensor(n)
   for i = 1,n do
      local rnd = randInt(1, i+1)
      ret[i] = ret[rnd]
      ret[rnd] = i
   end
   return ret
end

-- Get the median of a table.
function median(t)
  local temp={}

  -- deep copy table so that when we sort it, the original is unchanged
  -- also weed out any non numbers
  for k,v in pairs(t) do
    if type(v) == 'number' then
      table.insert( temp, v )
    end
 end

  if (#temp == 0) then
     print("error: median : empty table")
  end

  table.sort( temp )

  -- If we have an even number of table elements or odd.
  if math.fmod(#temp,2) == 0 then
    -- return mean value of middle two elements
    return ( temp[#temp/2] + temp[(#temp/2)+1] ) / 2
  else
    -- return middle elements
    return temp[math.ceil(#temp/2)]
  end
end