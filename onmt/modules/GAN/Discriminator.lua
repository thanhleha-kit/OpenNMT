local function reverseInput(batch)
  batch.sourceInput, batch.sourceInputRev = batch.sourceInputRev, batch.sourceInput
  batch.sourceInputFeatures, batch.sourceInputRevFeatures = batch.sourceInputRevFeatures, batch.sourceInputFeatures
  batch.sourceInputPadLeft, batch.sourceInputRevPadLeft = batch.sourceInputRevPadLeft, batch.sourceInputPadLeft
end

--[[ Discriminator is a bidirectional Sequencer used for the source language.


 `netFwd`

    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

 `netBwd`

    h_1 <= h_2 <= h_3 <= ... <= h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 <= h_2 <= h_3 <= ... <= h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

Inherits from [onmt.Sequencer](onmt+modules+Sequencer).

--]]
local Discriminator, parent = torch.class('onmt.Discriminator', 'nn.Container')

--[[ Create a bi-encoder.

Parameters:

  * `input` - input neural network.
  * `rnn` - recurrent template module.
  * `merge` - fwd/bwd merge operation {"concat", "sum"}
]]
function Discriminator:__init(input, rnn, merge)
  parent.__init(self)

  self.fwd = onmt.EncoderNoContext.new(input, rnn)
  self.bwd = onmt.EncoderNoContext.new(input:clone('weight', 'bias', 'gradWeight', 'gradBias'), rnn:clone())

  self.args = {}
  self.args.merge = merge

  self.args.rnnSize = rnn.outputSize
  self.args.numEffectiveLayers = rnn.numEffectiveLayers

  if self.args.merge == 'concat' then
    self.args.hiddenSize = self.args.rnnSize * 2
  else
    self.args.hiddenSize = self.args.rnnSize
  end

  self:add(self.fwd)
  self:add(self.bwd)
  
  -- a predictor takes the final state of the rnn(s) 
  -- and predict the final value of 0 or 1
  self.predictor = nn.Sequential()
					:add(nn.Linear(self.args.hiddenSize, self.args.hiddenSize))
					:add(nn.Tanh())
					:add(nn.Linear(self.args.hiddenSize, 2)) -- prob real or fake
					:add(nn.LogSoftMax())
  
  self:add(self.predictor)
  
  _G.logger:info(' * Discriminator created with Bidirectional RNN')

  self:resetPreallocation()
end

--[[ Return a new Discriminator using the serialized data `pretrained`. ]]
function Discriminator.load(pretrained)
  local self = torch.factory('onmt.Discriminator')()

  parent.__init(self)

  self.fwd = onmt.EncoderNoContext.load(pretrained.modules[1])
  self.bwd = onmt.EncoderNoContext.load(pretrained.modules[2])
  self.args = pretrained.args

  self:add(self.fwd)
  self:add(self.bwd)

  self:resetPreallocation()

  return self
end

--[[ Return data to serialize. ]]
function Discriminator:serialize()
  local modulesData = {}
  for i = 1, #self.modules do
    table.insert(modulesData, self.modules[i]:serialize())
  end

  return {
    name = 'Discriminator',
    modules = modulesData,
    args = self.args
  }
end

function Discriminator:resetPreallocation()
  -- Prototype for preallocated full context vector.
  self.contextProto = torch.Tensor()

  -- Prototype for preallocated full hidden states tensors.
  self.stateProto = torch.Tensor()
end

function Discriminator:maskPadding()
  self.fwd:maskPadding()
  self.bwd:maskPadding()
end

function Discriminator:forward(batch)
  if self.statesProto == nil then
    self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                         self.stateProto,
                                                         { batch.size, self.args.hiddenSize })
  end

  --~ local states = onmt.utils.Tensor.reuseTensorTable(self.statesProto, { batch.size, self.args.hiddenSize })
  
  local state = onmt.utils.Tensor.reuseTensor(self.statesProto, {batch.size, self.args.hiddenSize})

  local fwdState = self.fwd:forward(batch)
  reverseInput(batch)
  local bwdState = self.bwd:forward(batch)
  reverseInput(batch)

  if self.args.merge == 'concat' then
      state:narrow(2, 1, self.args.rnnSize):copy(fwdState)
      state:narrow(2, self.args.rnnSize + 1, self.args.rnnSize):copy(bwdState)

  elseif self.args.merge == 'sum' then
      state:copy(fwdState)
      state:add(bwdState)
  end
  
  local finalState = states[#states]
  
  self.output = finalState
 
  return self.output
end

function Discriminator:backward(batch, criterion)
	
	local prediction = self.predictor:forward(self.output)
	local loss = criterion:forward(prediction)
	
	local dLoss = criterion:backward(prediction)
	
	local gradState = self.predictor:backward(self.output, dLoss)
	
	-- now we backprop to the RNN(s)
	local gradStateOutputFwd, gradStateOutputBwd
	
	if self.args.merge == 'concat' then
		local statesSplit = gradState:chunk(2, 2)
		gradStateOutputFwd = statesSplit[1]
		gradStateOutputBwd = statesSplit[2]
	elseif self.args.merge == 'sum' then
		gradStateOutputFwd = gradState
		gradStateOutputBwd = gradState
	end
	
	local gradInputFwd = self.fwd:backward(batch, gradStateOutputFwd)
	
	local gradInputBwd = self.bwd:backward(batch, gradStatesOutputBwd)
	
	-- this is for the embedding layer
	for t = 1, batch.sourceLength do
		onmt.utils.Tensor.recursiveAdd(gradInputFwd[t], gradInputBwd[batch.sourceLength - t + 1])
	end
	
	-- gradInputFwd is the gradInput of the inputNetwork
	-- in the GAN problem, this is the gradOutput of the translation model
	-- note: we never use it in normal translation models
	-- note 2: currently the lookuptable doesn't have gradInput yet !!! 
	return gradInputFwd
end
--~ 
--~ function Discriminator:backward(batch, gradStatesOutput)
  --~ gradStatesOutput = gradStatesOutput
    --~ or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         --~ onmt.utils.Cuda.convert(torch.Tensor()),
                                         --~ { batch.size, self.args.hiddenSize })
--~ 
  --~ local gradStatesOutputFwd = {}
  --~ local gradStatesOutputBwd = {}
--~ 
  --~ if self.args.merge == 'concat' then
    --~ for i = 1, #gradStatesOutput do
      --~ local statesSplit = gradStatesOutput[i]:chunk(2, 2)
      --~ table.insert(gradStatesOutputFwd, statesSplit[1])
      --~ table.insert(gradStatesOutputBwd, statesSplit[2])
    --~ end
  --~ elseif self.args.merge == 'sum' then
    --~ gradStatesOutputFwd = gradStatesOutput
    --~ gradStatesOutputBwd = gradStatesOutput
  --~ end
--~ 
  --~ local gradInputFwd = self.fwd:backward(batch, gradStatesOutputFwd)
--~ 
  --~ local gradInputBwd = self.bwd:backward(batch, gradStatesOutputBwd)
--~ 
  --~ for t = 1, batch.sourceLength do
    --~ onmt.utils.Tensor.recursiveAdd(gradInputFwd[t], gradInputBwd[batch.sourceLength - t + 1])
  --~ end
--~ 
  --~ return gradInputFwd
--~ end
