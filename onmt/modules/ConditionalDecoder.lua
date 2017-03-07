--[[ Unit to decode a sequence of output tokens.

     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

Inherits from [onmt.Sequencer](onmt+modules+Sequencer).

--]]
local ConditionalDecoder, parent = torch.class('onmt.ConditionalDecoder', 'onmt.Sequencer')


--[[ Construct a conditional decoder layer.

Parameters:

  * `inputNetwork` - input nn module.
  * `inputRnn` - recurrent module, such as [onmt.LSTM](onmt+modules+LSTM). This RNN combines the input x_t with the prev h_t 
  * `generator` - optional, an output [onmt.Generator](onmt+modules+Generator).
  * `inputFeed` - bool, enable input feeding.
  * `tie_weight_dec` - bool, enable weight sharing between output and input embedding to save memory (if necessary). 
--]]
function ConditionalDecoder:__init(inputNetwork, inputRnn, generator, attention, inputFeed, tie_weight_dec)
  self.inputRnn = inputRnn
  self.inputNet = inputNetwork

  self.args = {}
  self.args.rnnSize = self.inputRnn.outputSize
  self.args.numEffectiveLayers = self.inputRnn.numEffectiveLayers
  
  
  self.args.totalLayers = self.args.numEffectiveLayers 
  

  self.args.inputIndex = {}
  self.args.outputIndex = {}

  -- Input feeding means the decoder takes an extra
  -- vector each time representing the attention at the
  -- previous step.
  self.args.inputFeed = inputFeed
  
  
  -- Attention type
  self.args.attention = attention
  self.args.tying = tie_weight_dec
  
  -- The main network is added in this step
  parent.__init(self, self:_buildModel())

  -- The generator use the output of the decoder sequencer to generate the
  -- likelihoods over the target vocabulary.
  self.generator = generator
  self:add(self.generator)

  self:resetPreallocation()
end

--[[ Return a new Decoder using the serialized data `pretrained`. ]]
function ConditionalDecoder.load(pretrained)
  local self = torch.factory('onmt.Decoder')()

  self.args = pretrained.args

  parent.__init(self, pretrained.modules[1])
  self.generator = pretrained.modules[2]
  self:add(self.generator)

  self:resetPreallocation()

  return self
end

--[[ Return data to serialize. ]]
function ConditionalDecoder:serialize()
  return {
    modules = self.modules,
    args = self.args,
    name = "ConditionalDecoder"
  }
end

function ConditionalDecoder:resetPreallocation()
  if self.args.inputFeed then
    self.inputFeedProto = torch.Tensor()
  end

  -- Prototype for preallocated hidden and cell states.
  self.stateProto = torch.Tensor()

  -- Prototype for preallocated output gradients.
  self.gradOutputProto = torch.Tensor()

  -- Prototype for preallocated context gradient.
  self.gradContextProto = torch.Tensor()
  
  if self.args.tying == true then
	  _G.logger:info('* Tying input and output weight of decoder to save more memory')
	  -- Retrieving the lookup table
	  local inputLUT = self.inputNet
	  if torch.isTypeOf(inputLUT, 'nn.ParallelTable') then
	  else
		inputLUT = inputLUT.modules[1]
	  end
	  
	  -- share weight between generator and input embedding to save more memory
	  self.generator.net.modules[1]:noBias() -- because the lookup table doesn't have weight
	  self.generator.net.modules[1]:share(inputLUT, 'weight','gradWeight')
  end
end

--[[ Build a default one time-step of the decoder

Returns: An nn-graph mapping

  $${(c^1_{t-1}, h^1_{t-1}, .., c^L_{t-1}, h^L_{t-1}, x_t, con/H, if) =>
  (c^1_{t}, h^1_{t}, .., c^L_{t}, h^L_{t}, a)}$$

  Where ${c^l}$ and ${h^l}$ are the hidden and cell states at each layer,
  ${x_t}$ is a sparse word to lookup,
  ${con/H}$ is the context/source hidden states for attention,
  ${if}$ is the input feeding, and
  ${a}$ is the context vector computed at this timestep.
--]]
function ConditionalDecoder:_buildModel()
  local inputs = {} -- input for the whole network
  local states = {} -- input for the input RNN
  local condStates = {} -- input for the conditional RNN
  

  -- Inputs are previous layers for input RNN first.
  for _ = 1, self.args.numEffectiveLayers do
    local h0 = nn.Identity()() -- batchSize x rnnSize
    table.insert(inputs, h0)
    table.insert(states, h0)
  end
  
  --~ for _ = 1, self.args.numCondLayers do
    --~ local h0 = nn.Identity()() -- batchSize x rnnSize
    --~ table.insert(inputs, h0)
    --~ table.insert(condStates, h0)
  --~ end 
  
  -- Word index (or plus features) 
  local x = nn.Identity()() -- batchSize
  table.insert(inputs, x)
  self.args.inputIndex.x = #inputs
	
  local context = nn.Identity()() -- batchSize x sourceLength x rnnSize
  table.insert(inputs, context)
  self.args.inputIndex.context = #inputs

  local inputFeed
  if self.args.inputFeed then
    inputFeed = nn.Identity()() -- batchSize x rnnSize
    table.insert(inputs, inputFeed)
    self.args.inputIndex.inputFeed = #inputs
  end

  -- Compute the input network.
  local embedding = self.inputNet(x)
  
  
  local input = embedding
  -- If set, concatenate previous decoder output.
  if self.args.inputFeed then
    input = nn.JoinTable(2)({input, inputFeed})
  end
  table.insert(states, input)

  -- Forward states and input into the RNN.
  local outputs = self.inputRnn(states)

  -- The output of a subgraph is a node: split it to access the last RNN output.
  outputs = { outputs:split(self.args.numEffectiveLayers) }

  -- Compute the attention here using h^L as query.
  
  local attnLayer = onmt.AttentionLayer(self.args.rnnSize)
  
  attnLayer.name = 'decoderAttn'
  
  -- s' in the formula
  --~ local intermediateHidden = nn.Dropout(self.inputRnn.dropout)(outputs[#outputs])
  
  local intermediateHidden = outputs[#outputs]
  
  local attnInput = {intermediateHidden, context}
  
  -- a GRU to combine the contextVector and the intermediateHidden
  --~ local contextVector = nn.Dropout(self.inputRnn.dropout)(attnLayer(attnInput))
  --~ intermediateHidden = nn.Dropout(self.inputRnn.dropout)(intermediateHidden)
  
  local contextVector = attnLayer(attnInput)	
  
  local resetGate = nn.Sigmoid()(nn.CAddTable()({nn.Linear(self.args.rnnSize, self.args.rnnSize)(contextVector), 
												 nn.Linear(self.args.rnnSize, self.args.rnnSize)(intermediateHidden) }))
  local updateGate = nn.Sigmoid()(nn.CAddTable()({nn.Linear(self.args.rnnSize, self.args.rnnSize)(contextVector), 
											      nn.Linear(self.args.rnnSize, self.args.rnnSize)(intermediateHidden) }))
  
  local candidate = nn.Tanh()(nn.CAddTable()({nn.Linear(self.args.rnnSize, self.args.rnnSize)(contextVector), 
										 nn.CMulTable()({resetGate, nn.Linear(self.args.rnnSize, self.args.rnnSize)(intermediateHidden)})}))
  
  -- s_bar in the formula
  local candidateGate = nn.AddConstant(1,false)(nn.MulConstant(-1,false)(updateGate))
  
  local finalHidden = nn.CAddTable()({ 
		nn.CMulTable()({updateGate, intermediateHidden}),
		nn.CMulTable()({candidateGate, candidate})
		})
  
  local generatorInput
  if self.inputRnn.dropout > 0 then
    generatorInput = nn.Dropout(self.inputRnn.dropout)(finalHidden)
  else
	generatorInput = nn.Identity()(finalHidden)
  end
  
  --~ table.insert(condStates, attnOutput) -- input the attention output to the cond RNN
  --~ 
  --~ local finalOutput = self.condRnn(condStates)
  --~ 
  --~ 
  --~ finalOutput = { finalOutput:split(self.args.numCondLayers) }
  --~ 
  --~ for k = 1, #finalOutput do
	--~ table.insert(outputs, finalOutput[k])
  --~ end
  --~ 
  --~ local generatorInput
  --~ if self.inputRnn.dropout > 0 then
	--~ generatorInput = nn.Dropout(self.inputRnn.dropout)(finalOutput[#finalOutput])
  --~ else
	--~ generatorInput = nn.Identity()(finalOutput[#finalOutput])
  --~ end
  --~ 
  table.insert(outputs, generatorInput)
  
  -- The final `outputs` table includes:
  --  - hiddens from inputRNN 
  --  - hiddens from condRNN
  --  - generator input (last state from condRNN)
   
  return nn.gModule(inputs, outputs)
end

--[[ Mask padding means that the attention-layer is constrained to
  give zero-weight to padding. This is done by storing a reference
  to the softmax attention-layer.

  Parameters:

  * See  [onmt.MaskedSoftmax](onmt+modules+MaskedSoftmax).
--]]
function ConditionalDecoder:maskPadding(sourceSizes, sourceLength)
  if not self.decoderAttn then
    self.network:apply(function (layer)
      if layer.name == 'decoderAttn' then
        self.decoderAttn = layer
      end
    end)
  end

  self.decoderAttn:replace(function(module)
    if module.name == 'softmaxAttn' then
      local mod
      if sourceSizes ~= nil then
        mod = onmt.MaskedSoftmax(sourceSizes, sourceLength)
      else
        mod = nn.SoftMax()
      end

      mod.name = 'softmaxAttn'
      mod:type(module._type)
      self.softmaxAttn = mod
      return mod
    else
      return module
    end
  end)
end

--[[ Run one step of the decoder.

Parameters:

  * `input` - input to be passed to inputNetwork.
  * `prevStates` - stack of hidden states (batch x layers*model x rnnSize)
  * `context` - encoder output (batch x n x rnnSize)
  * `prevOut` - previous distribution (batch x #words)
  * `t` - current timestep

Returns:

 1. `out` - Top-layer hidden state.
 2. `states` - All states.
--]]
function ConditionalDecoder:forwardOne(input, prevStates, context, prevOut, t)
  local inputs = {}

  -- Create RNN input (see sequencer.lua `buildNetwork('dec')`).
  onmt.utils.Table.append(inputs, prevStates)
  table.insert(inputs, input)
  table.insert(inputs, context)
  local inputSize
  if torch.type(input) == 'table' then
    inputSize = input[1]:size(1)
  else
    inputSize = input:size(1)
  end

  if self.args.inputFeed then
    if prevOut == nil then
      table.insert(inputs, onmt.utils.Tensor.reuseTensor(self.inputFeedProto,
                                                         { inputSize, self.args.rnnSize }))
    else
      table.insert(inputs, prevOut)
    end
  end

  -- Remember inputs for the backward pass.
  if self.train then
    self.inputs[t] = inputs
  end
  

  local outputs = self:net(t):forward(inputs)

  -- Make sure decoder always returns table.
  if type(outputs) ~= "table" then outputs = { outputs } end

  local out = outputs[#outputs]
  local states = {}
  for i = 1, #outputs - 1 do
    table.insert(states, outputs[i])
  end
  

  return out, states
end

--[[Compute all forward steps.

  Parameters:

  * `batch` - `Batch` object
  * `encoderStates` -
  * `context` -
  * `func` - Calls `func(out, t)` each timestep.
--]]

function ConditionalDecoder:forwardAndApply(batch, encoderStates, context, func)
  -- TODO: Make this a private method.
  if self.statesProto == nil then
    self.statesProto = onmt.utils.Tensor.initTensorTable(#encoderStates,
                                                         self.stateProto,
                                                         { batch.size, self.args.rnnSize })
  end

  local states = onmt.utils.Tensor.copyTensorTable(self.statesProto, encoderStates)
  

  local prevOut

  for t = 1, batch.targetLength do
	--~ print(t)
    prevOut, states = self:forwardOne(batch:getTargetInput(t), states, context, prevOut, t)
    func(prevOut, t)
  end
end

--[[Compute all forward steps.

  Parameters:

  * `batch` - a `Batch` object.
  * `encoderStates` - a batch of initial decoder states (optional) [0]
  * `context` - the context to apply attention to.

  Returns: Table of top hidden state for each timestep.
--]]
function ConditionalDecoder:forward(batch, encoderStates, context)
  --~ print(encoderStates)
  encoderStates = encoderStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })
  --~ print(#encoderStates)
  if self.train then
    self.inputs = {}
  end

  local outputs = {}

  self:forwardAndApply(batch, encoderStates, context, function (out)
    table.insert(outputs, out)
  end)
  
  --~ print(#outputs)

  return outputs
end

--[[ Compute the backward update.

Parameters:

  * `batch` - a `Batch` object
  * `outputs` - expected outputs
  * `criterion` - a single target criterion object

  Note: This code runs both the standard backward and criterion forward/backward.
  It returns both the gradInputs and the loss.
  -- ]]
function ConditionalDecoder:backward(batch, outputs, criterion)
  if self.gradOutputsProto == nil then
    self.gradOutputsProto = onmt.utils.Tensor.initTensorTable(self.args.totalLayers + 1,
                                                              self.gradOutputProto,
                                                              { batch.size, self.args.rnnSize })
  end

  local gradStatesInput = onmt.utils.Tensor.reuseTensorTable(self.gradOutputsProto,
                                                             { batch.size, self.args.rnnSize })
  local gradContextInput = onmt.utils.Tensor.reuseTensor(self.gradContextProto,
                                                         { batch.size, batch.sourceLength, self.args.rnnSize })

  local loss = 0

  for t = batch.targetLength, 1, -1 do
    -- Compute decoder output gradients.
    -- Note: This would typically be in the forward pass.
    local pred = self.generator:forward(outputs[t])
    local output = batch:getTargetOutput(t)

    loss = loss + criterion:forward(pred, output)

    -- Compute the criterion gradient.
    local genGradOut = criterion:backward(pred, output)
    for j = 1, #genGradOut do
      genGradOut[j]:div(batch.totalSize)
    end

    -- Compute the final layer gradient.
    local decGradOut = self.generator:backward(outputs[t], genGradOut)
    gradStatesInput[#gradStatesInput]:add(decGradOut)

    -- Compute the standard backward.
    local gradInput = self:net(t):backward(self.inputs[t], gradStatesInput)

    -- Accumulate encoder output gradients.
    gradContextInput:add(gradInput[self.args.inputIndex.context])
    gradStatesInput[#gradStatesInput]:zero()

    -- Accumulate previous output gradients with input feeding gradients.
    if self.args.inputFeed and t > 1 then
      gradStatesInput[#gradStatesInput]:add(gradInput[self.args.inputIndex.inputFeed])
    end

    -- Prepare next decoder output gradients.
    for i = 1, #self.statesProto do
      gradStatesInput[i]:copy(gradInput[i])
    end
  end

  return gradStatesInput, gradContextInput, loss
end

--[[ Compute the loss on a batch.

Parameters:

  * `batch` - a `Batch` to score.
  * `encoderStates` - initialization of decoder.
  * `context` - the attention context.
  * `criterion` - a pointwise criterion.

--]]
function ConditionalDecoder:computeLoss(batch, encoderStates, context, criterion)
  encoderStates = encoderStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })

  local loss = 0
  self:forwardAndApply(batch, encoderStates, context, function (out, t)
    local pred = self.generator:forward(out)
    local output = batch:getTargetOutput(t)
    loss = loss + criterion:forward(pred, output)
  end)

  return loss
end


--[[ Compute the score of a batch.

Parameters:

  * `batch` - a `Batch` to score.
  * `encoderStates` - initialization of decoder.
  * `context` - the attention context.

--]]
function ConditionalDecoder:computeScore(batch, encoderStates, context)
  encoderStates = encoderStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })

  local score = {}

  self:forwardAndApply(batch, encoderStates, context, function (out, t)
    local pred = self.generator:forward(out)
    for b = 1, batch.size do
      if t <= batch.targetSize[b] then
        score[b] = (score[b] or 0) + pred[1][b][batch.targetOutput[t][b]]
      end
    end
  end)

  return score
end
