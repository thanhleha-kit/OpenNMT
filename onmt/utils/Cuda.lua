local ExtendedCmdLine = require('onmt.utils.ExtendedCmdLine')

local Cuda = {
  gpuIds = {},
  activated = false
}

local options = {
  {'-gpuid',     '0',   [[List of comma-separated GPU identifiers (1-indexed). CPU is used when set to 0.]],
                                 {valid=ExtendedCmdLine.listUInt}},
  {'-fallback_to_cpu', false, [[If GPU can't be use, rollback on the CPU.]]},
  {'-no_nccl', false, [[Disable usage of nccl in parallel mode.]]}
}

function Cuda.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
end

function Cuda.init(opt, masterGPU)
  for _, val in ipairs(onmt.utils.String.split(opt.gpuid, ',')) do
    local id = tonumber(val)
    assert(id ~= nil and id >= 0, 'invalid GPU identifier: ' .. val)
    if id > 0 then
      table.insert(Cuda.gpuIds, id)
    end
  end

  Cuda.activated = #Cuda.gpuIds > 0

  if Cuda.activated then
    local _, err = pcall(function()
      require('cutorch')
      require('cunn')
      require('cudnn')

      if masterGPU == nil then
        masterGPU = 1

        -- Validate GPU identifiers.
        for i = 1, #Cuda.gpuIds do
          assert(Cuda.gpuIds[i] <= cutorch.getDeviceCount(),
                 'GPU ' .. Cuda.gpuIds[i] .. ' is requested but only '
                   .. cutorch.getDeviceCount() .. ' GPUs are available')
        end

        _G.logger:info('Using GPU(s): ' .. table.concat(Cuda.gpuIds, ', '))
      end

      cutorch.setDevice(Cuda.gpuIds[masterGPU])

      if opt.seed then
        cutorch.manualSeed(opt.seed)
      end
    end)

    if err then
      if opt.fallback_to_cpu then
        _G.logger:warning('Falling back to CPU')
        Cuda.activated = false
      else
        error(err)
      end
    end
  end
end

--[[
  Recursively move all supported objects in `obj` on the GPU.
  When using CPU only, converts to float instead of the default double.
]]
function Cuda.convert(obj)
  local objtype = torch.typename(obj)
  if objtype then
    if Cuda.activated and obj.cuda ~= nil then
      if objtype:find('torch%..*LongTensor') then
        return obj:cudaLong()
      else
        return obj:cuda()
      end
    elseif not Cuda.activated and obj.float ~= nil then
      -- Defaults to float instead of double.
      if objtype:find('torch%..*LongTensor') then
        return obj:long()
      else
        return obj:float()
      end
    end
  end

  if objtype or type(obj) == 'table' then
    for k, v in pairs(obj) do
      obj[k] = Cuda.convert(v)
    end
  end

  return obj
end

--[[
  Synchronize operations on current device if working on GPU.
  Do nothing otherwise.
]]
function Cuda.synchronize()
  if Cuda.activated then cutorch.synchronize() end
end

--[[
  Number of available GPU.
]]
function Cuda.gpuCount()
  return #Cuda.gpuIds
end

--[[
  Free memory on the current GPU device.
]]
function Cuda.freeMemory()
  if Cuda.activated then
    local freeMemory = cutorch.getMemoryUsage(cutorch.getDevice())
    return freeMemory
  end
  return 0
end

return Cuda
