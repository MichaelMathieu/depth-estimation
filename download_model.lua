require 'torch'
require 'xlua'

function split(str, char)
   local nb, ne
   local e = 0
   ret = {}
   while true do
      nb, ne = str:find(char, e+1)
      if nb == nil then
	 table.insert(ret, str:sub(e+1))
	 return ret
      end
      table.insert(ret, str:sub(e+1, nb-1))
      e = ne
   end
end


function listdir(sshpath, dir)
   local r = io.popen('ssh ' .. sshpath .. " 'ls " .. dir .. "'")
   local ret = {}
   while true do
      local t = r:read()
      if t == nil then
	 break
      end
      if t:sub(-1) ~= '~' and (t:sub(1) ~= '#' or t:sub(-1) ~= '#') then
	 table.insert(ret, t)
      end
   end
   return ret
end

function parseFilter(str)
   local s1 = split(str, '-')
   local kernels = s1[1]
   local multiscale = nil
   if #s1 > 1 then
      multiscale = {}
      for i = 2,#s1 do
	 table.insert(multiscale, s1[i])
      end
   end
   local s2 = split(s1[1], '_')
   local layers = {}
   for i = 1,#s2 do
      local layer = {}
      local s3 = split(s2[i], 'x')
      if #s3 ~= 4 then
	 return nil
      end
      for j = 1,#s3 do
	 table.insert(layer, s3[j])
      end
      table.insert(layers, layer)
   end
   return {layers, multiscale}
end

function filterFilters(strs)
   local ret = {}
   for i = 1,#strs do
      local parsed = parseFilter(strs[i])
      if parsed ~= nil then
	 table.insert(ret, {strs[i], parsed[1], parsed[2]})
      end
   end
   return ret
end

function filterLearnings(strs)
   local ret = {}
   for i = 1,#strs do
      table.insert(ret, {strs[i]})
   end
   return ret
end

function filterImages(strs)
   local ret = {}
   for i = 1,#strs do
      table.insert(ret, {strs[i]})
   end
   return ret
end

function filterEpochs(strs)
   local ret = {}
   for i = 1,#strs do
      local str = strs[i]
      if str:sub(1,11) == 'model_of__e' then
	 local n = tonumber(str:sub(12))
	 if n ~= nil then
	    table.insert(ret, {str, n})
	 end
      end
   end
   return ret
end

function selectFile(files)
   for i = 1,#files do
      print('(' .. i .. ') ' .. files[i][1])
   end
   local i = nil
   while i == nil do
      i = tonumber(io.read())
   end
   return files[i]
end

function selectEpoch(epochs)
   local mini = #epochs+1000
   local maxi = -1
   for i = 1,#epochs do
      if epochs[i][2] < mini then
	 mini = epochs[i][2]
      end
      if epochs[i][2] > maxi then
	 maxi = epochs[i][2]
      end
   end
   if maxi == -1 or mini ~= 0 or maxi ~= #epochs-1 then
      print("Missing epochs, can't perform model selection")
      return nil
   end
   print("Select epoch: " .. mini .. ".." .. maxi)
   local i = nil
   while i == nil do
      i = tonumber(io.read())
   end
   assert(mini <= i and i <= maxi)
   return {'model_of__e' .. i, i}
end

function prompt(sshpath, basepath)
   local path = basepath
   local filters = filterFilters(listdir(sshpath, path))
   local filter = selectFile(filters)
   path = path .. '/' .. filter[1]
   local learnings = filterLearnings(listdir(sshpath, path))
   local learning = selectFile(learnings)
   path = path .. '/' .. learning[1]
   local images = filterImages(listdir(sshpath, path))
   local image = selectFile(images)
   path = path .. '/' .. image[1]
   local epochs = filterEpochs(listdir(sshpath, path))
   local epoch = selectEpoch(epochs)
   path = path .. '/' .. epoch[1]
   return path, epoch[1]
end

function downloadModel(sshfullpath)
   local splited = split(sshfullpath, ':')
   if #splited ~= 2 then
      print('Wrong ssh path')
      return nil
   end
   local sshpath = splited[1]
   local basepath = splited[2]
   local modeldir = 'models_downloaded'
   os.execute('mkdir -p ' .. modeldir)
   local path, filename = prompt(sshpath, basepath)
   os.execute('scp ' .. sshpath .. ':' .. path .. ' ' .. modeldir .. '/')
   return modeldir .. '/' .. filename
end
