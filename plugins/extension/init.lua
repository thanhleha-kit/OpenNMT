local Extension = {
  hooks = {}
}

local function doSomething(model)
  print('doing something for module "Extension"...')
end

function Extension.init(cmd)
  Extension.hooks['training:after_batch'] = doSomething
end

return Extension
