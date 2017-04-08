--[[ Class for managing the training process by logging and storing
  the state of the current epoch.
]]
local EpochState = torch.class('EpochState')

--[[ Initialize for epoch `epoch`]]
function EpochState:__init(epoch, startIterations, numIterations, learningRate)
  self.epoch = epoch
  self.iterations = startIterations - 1
  self.numIterations = numIterations
  self.learningRate = learningRate

  self.globalTimer = torch.Timer()

  self:reset()
end

function EpochState:reset()
  self.trainLoss = 0
  self.sourceWords = 0
  self.targetWords = 0
  self.numSamplesXENT = 0
  self.numSamplesRF   = 0
  self.timer = torch.Timer()
  self.totCumRewardPredError = 0
  self.totalReward = 0
end

--[[ Update training status. Takes `batch` (described in data.lua) and last loss.]]
function EpochState:update(model, batch, lossXENT, lossRF, numSamplesXENT, numSamplesRF, totCumRewardPredError)
  self.iterations = self.iterations + 1
  self.trainLoss = self.trainLoss + lossXENT
  self.sourceWords = self.sourceWords + model:getInputLabelsCount(batch)
  self.targetWords = self.targetWords + model:getOutputLabelsCount(batch)
  self.numSamplesXENT = self.numSamplesXENT + numSamplesXENT
  self.numSamplesRF = self.numSamplesRF + numSamplesRF
  
  if lossRF ~= 0 then
	self.totalReward = self.totalReward - lossRF * numSamplesRF
	self.totCumRewardPredError = self.totCumRewardPredError + totCumRewardPredError
  end
end

--[[ Log to status stdout. ]]
function EpochState:log(iteration)  
  _G.logger:info('Epoch %d ; Iteration %d/%d ; Learning rate %.4f ; Source tokens/s %d ; Perplexity %.2f; Avg. Reward %.2f; Tot. CRPE %.4f',
                 self.epoch,
                 iteration or self.iterations, self.numIterations,
                 self.learningRate,
                 self.sourceWords / self.timer:time().real,
                 math.exp(self.trainLoss / self.numSamplesXENT),
                 self.totalReward / self.numSamplesRF,
                 (self.totCumRewardPredError / self.numSamplesRF))

  self:reset()
end

function EpochState:getTime()
  return self.globalTimer:time().real
end

return EpochState
