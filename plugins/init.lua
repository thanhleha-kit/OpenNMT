require 'paths'

local Plugins = {
  path = '',
  plugins = {}
}

function Plugins.list()
  Plugins['path'] = paths.concat(paths.cwd(), 'plugins')
  local folderNames = {}
  for folderName in paths.iterdirs(Plugins['path']) do
      if folderName:find('$') then
          table.insert(folderNames, folderName)
      end
  end
  return folderNames
end

function Plugins.load(listPlugins, cmd)
  for plugin in string.gmatch(listPlugins, "[^, ]+") do
    local p = require('plugins.'..plugin..'.init')
    p.init(cmd)
    Plugins['plugins'][plugin] = p
    print('Initializing plugin \''..plugin..'\'')
  end
end

function Plugins.triggerHooks(name, params)
  for plugin,v in pairs(Plugins['plugins']) do
    if v['hooks'] and v['hooks'][name] then
      v['hooks'][name](params)
    end
  end
end

return Plugins
