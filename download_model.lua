require 'sys'
require 'common'

function listdir(sshpath, dir)
   options = options or ''
   if dir:sub(-1) ~= '/' then
      dir = dir .. '/'
   end
   local r = sys.execute(string.format("ssh %s '( [[ `uname` == \"Linux\" ]] && ls -lt --time-style +%%F %s | awk '\\''{print $6 \" \" $7}'\\'' ) || ( ( [[ `uname` == \"SunOS\" ]] && ls -cEt %s | awk '\\''{print $6 \" \" $9}'\\'' ) || echo ERROR `uname` ) '", sshpath, dir, dir))
   local ret = {}
   local lines = split(strip(r, {'\n'}), '\n')
   if #lines == 1 and split(lines[1], ' ')[1] == 'ERROR' then
      print('Error : No support for ' .. split(lines[1], ' ')[2])
      return nil
   end
   for i = 1,#lines do
      local t = strip(lines[i], {' '})
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

function getRecents(path)
   local r = sys.execute(string.format("ls -lt --time-style +%%F %s | awk '{print $7 \" \" $6}'", path))
   local lines = split(strip(r, {"\n"}), '\n')
   local ret = {}
   for i = 1,#lines do
      local splited = split(lines[i]:strip({' '}), ' ')
      if #splited == 2 then
	 table.insert(ret, splited)
      end
   end
   return ret
end

function parseFilter(str)
   local s1 = split(str, '-')
   local kernels = s1[1]
   local multiscale = nil
   local sf = false
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
      if #s3 == 1 then
	 if s3[1] == 'sf' then
	    sf = true
	 end
      elseif #s3 == 4 then
	 for j = 1,#s3 do
	    table.insert(layer, s3[j])
	 end
	 table.insert(layers, layer)
      else
	 return nil
      end
   end
   return {layers, multiscale, sf}
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
   if date1 == today then
      return ' **'
   end
   local date1day = tonumber(strip(sys.execute('date -d ' .. date1 .. ' +%j'), {' ', '\n'}))
   local todayday = tonumber(strip(sys.execute('date -d ' .. today .. ' +%j'), {' ', '\n'}))
   if math.mod(todayday-date1day, 365) < 2 then
      return ' *'
   else
      return ''
   end
end

function selectFile(files, today, specials)
   specials = specials or {}
   if #files == 0 then
      print('No files in specified directory')
      return nil
   end
   for i = 1,#files do
      print('(' .. i .. ') ' .. files[i][1] .. isRecent(files[i][2], today))
   end
   local i = nil
   while i == nil do
      i = io.read()
      for j = 1,#specials do
	 if i == specials[j] then
	    return i
	 end
      end
      if i == '' and #files == 1 then
	 i = 1
      end
      i = tonumber(i)
   end
   return files[i]
end

function selectEpoch(epochs, today)
   table.sort(epochs, function(a, b) return a[3] < b[3] end)
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
   print("Select epoch: " .. mini .. ".." .. maxi .. ' (default = last)')
   local i = nil
   while i == nil do
      i = io.read()
      if i == '' then
	 return {eopchs[maxi][1], maxi}
      else
	 i = tonumber(i)
      end
   end
   assert(mini <= i and i <= maxi)
   return {epochs[i][1], i}
end

function prompt(sshpath, basepath, savepath)
   local path = basepath
   local today = sys.execute('date +%F'):strip({' ', '\n'})
   local filters = filterFilters(listdir(sshpath, path))
   local filter = selectFile(filters, today, {''})
   if filter == nil then return nil end
   if filter == '' then
      local recents = getRecents(savepath)
      local recent = selectFile(recents, today)
      return savepath .. '/' .. recent[1]
   else
      path = path .. '/' .. filter[1]
      local learnings = filterLearnings(listdir(sshpath, path))
      local learning = selectFile(learnings, today)
      if learning == nil then return nil end
      path = path .. '/' .. learning[1]
      local images = filterImages(listdir(sshpath, path))
      local image = selectFile(images, today)
      if image == nil then return nil end
      path = path .. '/' .. image[1]
      local epochs = filterEpochs(listdir(sshpath, path))
      local epoch = selectEpoch(epochs, today)
      if epoch == nil then return nil end
      path = path .. '/' .. epoch[1]
      os.execute('scp ' .. sshpath .. ':' .. path .. ' ' .. savepath .. '/')
      return savepath .. '/' .. epoch[1]
   end
end

function downloadModel(sshfullpath)
   local splited = split(sshfullpath, ':')
   if #splited ~= 2 then
      print('Wrong ssh path')
      return nil
   end
   local sshpath = splited[1]
   local basepath = splited[2]
   local savedir = 'models_downloaded'
   local savepath = savedir
   os.execute('mkdir -p ' .. savepath)
   local path = prompt(sshpath, basepath, savepath)
   if path == nil then
      return nil
   else
      return path
   end
end
