require('nngraph')

--[[ A baseline rewarder that computes the potential reward
     based on the hidden state of the Decoder RNN
--]]
local BaselineRewarder, parent = torch.class('onmt.BaselineRewarder', 'onmt.Network')

--[[A nn-style module computing attention.

  Parameters:

  * `dim` - dimension of the context vectors.
--]]
function BaselineRewarder:__init(dim, cutoffGrad)

	cutoffGrad = cutoffGrad or true
  parent.__init(self, self:_buildModel(dim, cutoffGrad))
end

function BaselineRewarder:_buildModel(dim, cutoffGrad)
  
  self.inputViewer = nn.View(1,1,-1):setNumInputDims(3)
  self.outputViewer = nn.View(1,-1):setNumInputDims(2)
	
	local network = nn.Sequential()
	network:add(self.inputViewer)
	
	if cutoffGrad == true then
		network:add(nn.LinearNoBackpropInput(dim, 1))
	else
		network:add(nn.Linear(dim, 1))
	end
	network:add(self.outputViewer)

	return network
end


function BaselineRewarder:postParametersInitialization()
  self.net.modules[2].weight:zero()
  self.net.modules[2].bias:fill(0.01) 
end

-- input is a sequence of hidden layer with size nsteps x batchsize x dim
function BaselineRewarder:updateOutput(input)

	local batchSize = input:size(2)
	local seqLength = input:size(1)
	
	self.inputViewer:resetSize(batchSize * seqLength, -1)
	
	self.outputViewer:resetSize(seqLength, batchSize) 
	
	self.output = self.net:updateOutput(input)
	
	return self.output
end
