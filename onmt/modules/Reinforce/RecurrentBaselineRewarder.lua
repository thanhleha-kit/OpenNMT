require('nngraph')

--[[ A baseline rewarder that computes the potential reward
     based on the hidden state of the Decoder RNN
--]]
local RecurrentBaselineRewarder, parent = torch.class('onmt.RecurrentBaselineRewarder', 'onmt.Network')

--[[A nn-style module computing attention.

  Parameters:

  * `dim` - dimension of the context vectors.
--]]
function RecurrentBaselineRewarder:__init(dim)
  parent.__init(self, self:_buildModel(dim))
end

function RecurrentBaselineRewarder:_buildModel(dim)

	self.inputViewer = nn.View(1,1,-1):setNumInputDims(3)
  self.outputViewer = nn.View(1,-1):setNumInputDims(2)
	
	local network = nn.Sequential()
	network:add(nn.IdentityNoBackprop()) -- to cut off backprop 
	
	local rnn = cudnn.LSTM(dim, dim, 1, false, 0, false)
	network:add(rnn)
	
	local linearRewarder = onmt.BaselineRewarder(dim, false) -- we DONT cut off grad from here
	network:add(linearRewarder)
	
	return network
end


function RecurrentBaselineRewarder:updateOutput(input)
	
	self.output = self.net:updateOutput(input)
	
	return self.output
end
