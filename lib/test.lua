require('lib.onmt.init')
require('lib.data')
local mytester = torch.Tester()

local nmttest = torch.TestSuite()

function nmttest.Data()

end

mytester:add(nmttest)

function onmt.test(tests, seed)
  -- Limit number of threads since everything is small
  local nThreads = torch.getnumthreads()
   torch.setnumthreads(1)

   -- Randomize stuff
   local seed = seed or (1e5 * torch.tic())
   print('Seed: ', seed)
   math.randomseed(seed)
   torch.manualSeed(seed)
   mytester:run(tests)
   torch.setnumthreads(nThreads)
   return mytester
end
