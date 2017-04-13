require('nngraph')

--[[ A baseline rewarder that computes the potential reward
     based on the hidden state of the Decoder RNN
--]]
local RecurrentBaselineRewarder, parent = torch.class('onmt.BaselineRewarder', 'onmt.Network')

--[[A nn-style module computing attention.

  Parameters:

  * `dim` - dimension of the context vectors.
--]]
function RecurrentBaselineRewarder:__init(dim)
  parent.__init(self, self:_buildModel(dim))
end

function BaselineRewarder:_buildModel(dim)

	local network = nn.LinearNoBackpropInput(dim, 1)
	--~ local network = nn.Sequential():add(nn.LinearNoBackpropInput(dim, dim))
	--~ network:add(nn.Tanh())
	--~ network:add(nn.Linear(dim, 1))
	
	return network
end


function BaselineRewarder:postParametersInitialization()
  self.net.weight:zero()
  self.net.bias:fill(0.01) 
end
