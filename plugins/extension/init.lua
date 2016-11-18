local Extension = {
  hooks = {}
}

local function doSomething(model)
  print('doing something for module "Extension"...')
end

function Extension.init(cmd)
  Extension.hooks['training:after_batch'] = doSomething
end

function Extension.registerOptions(cmd)
  cmd:option('-extension:opt1', -2, [[option 1 for extension]])
end

return Extension
