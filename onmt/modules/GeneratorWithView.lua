--[[ Default decoder generator. Given RNN state, produce categorical distribution.

Simply implements $$softmax(W h + b)$$.
--]]
local GeneratorWithView, parent = torch.class('onmt.GeneratorWithView', 'onmt.Network')


function GeneratorWithView:__init(rnnSize, outputSize, dropout)
  parent.__init(self, self:_buildGenerator(rnnSize, outputSize, dropout))
end

function GeneratorWithView:_buildGenerator(rnnSize, outputSize, dropout)
  return nn.Sequential()
	:add(nn.View(-1, rnnSize):setNumInputDims(3))
	:add(nn.Dropout(dropout))
    :add(nn.Linear(rnnSize, outputSize, false))
    :add(nn:LogSoftMax())
end

function GeneratorWithView:updateOutput(input)
  self.output = self.net:updateOutput(input)
  return self.output
end

function GeneratorWithView:updateGradInput(input, gradOutput)
  self.gradInput = self.net:updateGradInput(input, gradOutput)
  return self.gradInput
end

function GeneratorWithView:accGradParameters(input, gradOutput, scale)
  self.net:accGradParameters(input, gradOutput[1], scale)
end
