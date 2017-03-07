local function reverseInput(batch)
  batch.sourceInput, batch.sourceInputRev = batch.sourceInputRev, batch.sourceInput
  batch.sourceInputFeatures, batch.sourceInputRevFeatures = batch.sourceInputRevFeatures, batch.sourceInputFeatures
  batch.sourceInputPadLeft, batch.sourceInputRevPadLeft = batch.sourceInputRevPadLeft, batch.sourceInputPadLeft
end

--[[ BiEncoder is a bidirectional Sequencer used for the source language.


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
local BiEncoderWithLoss, parent = torch.class('onmt.BiEncoderWithLoss', 'nn.Container')

--[[ Create a bi-encoder.

Parameters:

  * `input` - input neural network.
  * `rnn` - recurrent template module.
  * `merge` - fwd/bwd merge operation {"concat", "sum"}
  * 
]]
function BiEncoderWithLoss:__init(input, rnn, generator, merge)
  parent.__init(self)

  self.fwd = onmt.Encoder.new(input, rnn)
  self.bwd = onmt.Encoder.new(input:clone('weight', 'bias', 'gradWeight', 'gradBias'), rnn:clone())

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
  
  self.generator = generator
  
  --~ print(self.generator)
  self:add(self.generator)
  
  self.args.alpha = 0.2

  self:resetPreallocation()
end

--[[ Return a new BiEncoder using the serialized data `pretrained`. ]]
function BiEncoderWithLoss.load(pretrained)
  local self = torch.factory('onmt.BiEncoder')()

  parent.__init(self)

  self.fwd = onmt.Encoder.load(pretrained.modules[1])
  self.bwd = onmt.Encoder.load(pretrained.modules[2])
  self.generator = pretrained.modules[3]
  self.args = pretrained.args

  self:add(self.fwd)
  self:add(self.bwd)
  self:add(self.generator)

  self:resetPreallocation()

  return self
end

--[[ Return data to serialize. ]]
function BiEncoderWithLoss:serialize()
  local modulesData = {}
  for i = 1, #self.modules - 1  do
    table.insert(modulesData, self.modules[i]:serialize())
  end
  
  table.insert(modulesData, self.generator)

  return {
    name = 'BiEncoderWithLoss',
    modules = modulesData,
    args = self.args
  }
end

function BiEncoderWithLoss:resetPreallocation()
  -- Prototype for preallocated full context vector.
  self.contextProto = torch.Tensor()

  -- Prototype for preallocated full hidden states tensors.
  self.stateProto = torch.Tensor()

  -- Prototype for preallocated gradient of the backward context
  self.gradContextBwdProto = torch.Tensor()
  
  local inputLUT = self.fwd.inputNet
  
  if torch.isTypeOf(inputLUT, 'nn.ParallelTable') then
  
  else
	inputLUT = inputLUT.modules[1]
  end
  
  -- share weight between generator and input embedding to save weight
  print(self.generator.net.modules[3])
  print(inputLUT)
  self.generator.net.modules[3]:share(inputLUT, 'weight','gradWeight')
  
  self.criterion = nn.ClassNLLCriterion()
end

function BiEncoderWithLoss:maskPadding()
  self.fwd:maskPadding()
  self.bwd:maskPadding()
end

function BiEncoderWithLoss:forward(batch)
  if self.statesProto == nil then
    self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                         self.stateProto,
                                                         { batch.size, self.args.hiddenSize })
  end
  --~ print(batch.sourceLength)
  local states = onmt.utils.Tensor.reuseTensorTable(self.statesProto, { batch.size, self.args.hiddenSize })
  local context = onmt.utils.Tensor.reuseTensor(self.contextProto,
                                                { batch.size, batch.sourceLength, self.args.hiddenSize })

  local fwdStates, fwdContext = self.fwd:forward(batch)
  reverseInput(batch)
  local bwdStates, bwdContext = self.bwd:forward(batch)
  reverseInput(batch)

  if self.args.merge == 'concat' then
    for i = 1, #fwdStates do
      states[i]:narrow(2, 1, self.args.rnnSize):copy(fwdStates[i])
      states[i]:narrow(2, self.args.rnnSize + 1, self.args.rnnSize):copy(bwdStates[i])
    end
    for t = 1, batch.sourceLength do
      context[{{}, t}]:narrow(2, 1, self.args.rnnSize)
        :copy(fwdContext[{{}, t}])
      context[{{}, t}]:narrow(2, self.args.rnnSize + 1, self.args.rnnSize)
        :copy(bwdContext[{{}, batch.sourceLength - t + 1}])
    end
  elseif self.args.merge == 'sum' then
    for i = 1, #states do
      states[i]:copy(fwdStates[i])
      states[i]:add(bwdStates[i])
    end
    for t = 1, batch.sourceLength do
      context[{{}, t}]:copy(fwdContext[{{}, t}])
      context[{{}, t}]:add(bwdContext[{{}, batch.sourceLength - t + 1}])
    end
  end
  
  -- If training: we also forward the softmax for encoder language models
  if self.train == true then
	  --~ print(batch.size)
	  --~ print(batch.sourceLength)
	  --~ print(context:size())
	  --~ local viewedContext = context:view(-1, self.args.hiddenSize)
	  --~ self.generatorInput = context:view(-1, self.args.hiddenSize)
	  self.generatorInput = context
	  self.generatorOutput = self.generator:forward(self.generatorInput)
	  
	  --~ print(self.generatorOutput)
	  --~ print(batch.sourceInput)
	  --~ print(generatorOutput)
	  
	  --~ print(batch.sourceInput:size())
	  self.encoderLMLoss =  self.criterion:forward(self.generatorOutput, batch.sourceInput:view(-1))
  end

  return states, context
end

function BiEncoderWithLoss:backward(batch, gradStatesOutput, gradContextOutput)
  gradStatesOutput = gradStatesOutput
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.hiddenSize })
  assert(self.encoderLMLoss)
  
  local gradLMLoss = self.criterion:backward(self.generatorOutput, batch.sourceInput:view(-1))
  
  --~ print(gradLMLoss:size())
  --~ print(self.generatorInput:size())
  local gradLMContextOutput = self.generator:backward(self.generatorInput, gradLMLoss)
  --~ gradLMContextOutput = gradLMContextOutput:view(batch.size, batch.sourceLength, -1)
  
  gradContextOutput:add(gradLMContextOutput * self.args.alpha)
  
  local gradContextOutputFwd
  local gradContextOutputBwd

  local gradStatesOutputFwd = {}
  local gradStatesOutputBwd = {}

  if self.args.merge == 'concat' then
    local gradContextOutputSplit = gradContextOutput:chunk(2, 3)
    gradContextOutputFwd = gradContextOutputSplit[1]
    gradContextOutputBwd = gradContextOutputSplit[2]

    for i = 1, #gradStatesOutput do
      local statesSplit = gradStatesOutput[i]:chunk(2, 2)
      table.insert(gradStatesOutputFwd, statesSplit[1])
      table.insert(gradStatesOutputBwd, statesSplit[2])
    end
  elseif self.args.merge == 'sum' then
    gradContextOutputFwd = gradContextOutput
    gradContextOutputBwd = gradContextOutput

    gradStatesOutputFwd = gradStatesOutput
    gradStatesOutputBwd = gradStatesOutput
  end

  local gradInputFwd = self.fwd:backward(batch, gradStatesOutputFwd, gradContextOutputFwd)

  -- reverse gradients of the backward context
  local gradContextBwd = onmt.utils.Tensor.reuseTensor(self.gradContextBwdProto,
                                                       { batch.size, batch.sourceLength, self.args.rnnSize })

  for t = 1, batch.sourceLength do
    gradContextBwd[{{}, t}]:copy(gradContextOutputBwd[{{}, batch.sourceLength - t + 1}])
  end

  local gradInputBwd = self.bwd:backward(batch, gradStatesOutputBwd, gradContextBwd)

  for t = 1, batch.sourceLength do
    onmt.utils.Tensor.recursiveAdd(gradInputFwd[t], gradInputBwd[batch.sourceLength - t + 1])
  end

  return gradInputFwd
end
