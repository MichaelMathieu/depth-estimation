require 'sys'

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

function strip(str, chars)
   local function nochar(a)
      for i = 1,#chars do
	 if a == chars[i] then
	    return false
	 end
      end
      return true
   end
   local i = 1
   while i <= str:len() do
      if nochar(str:sub(i,i)) then
	 break
      end
      i = i+1
   end
   local str2 = str:sub(i)
   i = str2:len()
   while i >= 1 do
      if nochar(str2:sub(i,i)) then
	 break
      end
      i = i-1
   end
   return str2:sub(1,i)
end

function listdir(sshpath, dir)
   if dir:sub(-1) ~= '/' then
      dir = dir .. '/'
   end
   local r = sys.execute(string.format("ssh %s '[[ `uname` == \"Linux\" ]] && ls -l --time-style +%%F | awk '\\''{print $6 \" \" $7}'\\'' || [[ `uname` == \"SunOS\" ]] && ls -cE %s | awk '\\''{print $6 \" \" $9}'\\'' || echo ERROR `uname`'", sshpath, dir))
   local ret = {}
   local lines = split(r:strip({'\n'}), '\n')
   if #lines == 1 and split(lines[1], ' ')[1] == 'ERROR' then
      print('Error : No support for ' .. split(lines[1], ' ')[2])
      return nil
   end
   for i = 1,#lines do
      local t = lines[i]:strip({' '})
      if t ~= '' then
	 local splited = split(t, ' ')
	 local date = splited[1]
	 local name = splited[2]
	 if #splited == 2 and name:sub(-1) ~= '~' and (name:sub(1) ~= '#' or name:sub(-1) ~= '#') then
	    table.insert(ret, {name, date})
	 end
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
      local parsed = parseFilter(strs[i][1])
      if parsed ~= nil then
	 table.insert(ret, {strs[i][1], strs[i][2], parsed})
      end
   end
   return ret
end

function filterLearnings(strs)
   return strs
end

function filterImages(strs)
   return strs
end

function filterEpochs(strs)
   local ret = {}
   for i = 1,#strs do
      local str = strs[i][1]
      if str:sub(1,11) == 'model_of__e' then
	 local n = tonumber(str:sub(12))
	 if n ~= nil then
	    table.insert(ret, {str, strs[i][2], n})
	 end
      end
   end
   return ret
end

function isRecent(date1, today)
   --alright, that doesn't work on bissextile year :P
   local date1day = tonumber(strip(sys.execute('date -d ' .. date1 .. ' +%j'), {' ', '\n'}))
   local todayday = tonumber(strip(sys.execute('date -d ' .. today .. ' +%j'), {' ', '\n'}))
   return math.mod(todayday-date1day, 365) < 2
end

function selectFile(files, today)
   for i = 1,#files do
      if isRecent(files[i][2], today) then
	 print('(' .. i .. ') ' .. files[i][1] .. ' *')
      else
	 print('(' .. i .. ') ' .. files[i][1])
      end
   end
   local i = nil
   while i == nil do
      i = tonumber(io.read())
   end
   return files[i]
end

function selectEpoch(epochs, recent)
   local mini = #epochs+1000
   local maxi = -1
   for i = 1,#epochs do
      if epochs[i][3] < mini then
	 mini = epochs[i][3]
      end
      if epochs[i][3] > maxi then
	 maxi = epochs[i][3]
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
   local recent = sys.execute('date +%F'):strip({' ', '\n'})
   local filters = filterFilters(listdir(sshpath, path))
   local filter = selectFile(filters, recent)
   path = path .. '/' .. filter[1]
   local learnings = filterLearnings(listdir(sshpath, path))
   local learning = selectFile(learnings, recent)
   path = path .. '/' .. learning[1]
   local images = filterImages(listdir(sshpath, path))
   local image = selectFile(images, recent)
   path = path .. '/' .. image[1]
   local epochs = filterEpochs(listdir(sshpath, path))
   local epoch = selectEpoch(epochs, recent)
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
