--[[
  This file provides generic parallel class - allowing to run functions
  in different threads and on different GPU
]]--

local Parallel = {
  gpus = {0},
  _pool = nil,
  count = 1,
  gradBuffer = torch.Tensor()
}

-- Synchronizes the current stream on dst device with src device. This is only
-- necessary if we are not on the default stream
local function waitForDevice(dst, src)
   local stream = cutorch.getStream()
   if stream ~= 0 then
      cutorch.streamWaitForMultiDevice(dst, stream, { [src] = {stream} })
   end
end

function Parallel.init(args)
  if utils.Cuda.activated then
    Parallel.count = args.nparallel
    Parallel.gpus = utils.Cuda.getGPUs(args.nparallel)
    Parallel.gradBuffer = utils.Cuda.convert(Parallel.gradBuffer)
    if Parallel.count > 1 then
      print('Using ' .. Parallel.count .. ' threads on ' .. #Parallel.gpus .. ' GPUs')
      local threads = require 'threads'
      threads.Threads.serialization('threads.sharedserialize')
      local thegpus = Parallel.gpus
      Parallel._pool = threads.Threads(
        Parallel.count,
        function(threadid)
          require 'cunn'
          require 'nngraph'
          require('lib.utils.init')
          require('lib.train.init')
          require('lib.onmt.init')
          require('lib.data')
          utils.Cuda.init(args, thegpus[threadid])
        end
      ) -- dedicate threads to GPUs
      Parallel._pool:specific(true)
    end
    if Parallel.count > 1 and not(args.no_nccl) then
      -- check if we have nccl installed
      local ret
      ret, Parallel.usenccl = pcall(require, 'nccl')
      if not ret  then
        print("WARNING: for improved efficiency in nparallel mode - do install nccl")
        Parallel.usenccl = nil
      elseif os.getenv('CUDA_LAUNCH_BLOCKING') == '1' then
        print("WARNING: CUDA_LAUNCH_BLOCKING set - cannot use nccl")
        Parallel.usenccl = nil
      end
    end
  end
end

function Parallel.getGPU(i)
  if utils.Cuda.activated and Parallel.gpus[i] ~= 0 then
    return Parallel.gpus[i]
  end
  return 0
end

--[[ Launch function in parallel on different threads. ]]
function Parallel.launch(label, closure, endcallback)
  endcallback = endcallback or function() end
  if label ~= nil then
    print("START",label)
  end
  for j = 1, Parallel.count do
    if Parallel._pool == nil then
      endcallback(closure(j))
    else
      Parallel._pool:addjob(j, function() return closure(j) end, endcallback)
    end
  end
  if Parallel._pool then
    Parallel._pool:synchronize()
  end
  if label ~= nil then
    print("DONE",label)
  end
end

--[[ Accumulate the gradient parameters from the different replica. ]]
function Parallel.accGradParams(params, grad_params, batches)
  if Parallel.count > 1 then
    for h = 1, #grad_params[1] do
      local param_replica = { params[1][h] }
      local grad_replica = { grad_params[1][h] }
      for j = 2, #batches do
        -- TODO - this is memory costly since we need to clone full parameters from one GPU to another
        -- to avoid out-of-memory, we can copy/add by batch
        table.insert(param_replica, params[j][h])
        table.insert(grad_replica, grad_params[j][h])
        if not Parallel.usenccl then
          -- Synchronize before and after copy to ensure that it doesn't overlap
          -- with this add or previous adds
          waitForDevice(Parallel.gpus[j], Parallel.gpus[1])
          local remoteGrads = utils.Tensor.reuseTensor(Parallel.gradBuffer, grad_params[j][h]:size())
          remoteGrads:copy(grad_params[j][h])
          waitForDevice(Parallel.gpus[1], Parallel.gpus[j])
          grad_params[1][h]:add(remoteGrads)
        end
      end
      if not Parallel.usenccl then
        Parallel.syncParams(param_replica)
      else
        Parallel.usenccl.reduce(grad_replica, param_replica, true)
      end
    end
  end
end

--[[ Sync parameters from main model to different parallel threads. ]]
function Parallel.syncParams(params)
  for j = 2, Parallel.count do
    params[j]:copy(params[1])
    waitForDevice(Parallel.gpus[j], Parallel.gpus[1])
  end
end

return Parallel
