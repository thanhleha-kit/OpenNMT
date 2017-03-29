local Highway, parent = torch.class('onmt.Highway','onmt.Network')


function Highway:__init(dim, activation, useBias)
	
	self.dim = dim
	self.useBias = useBias or false
	self.activation = activation or nn.ReLU()
	self.highwayBias = -2
	parent.__init(self, self:_buildModel(self.dim, self.activation, self.useBias))
	
end

--build the nn Graph. 
-- Simply concatenate two vectors, 
-- Then use a linear transformation, with an activation followed
function Highway:_buildModel(dim, activation, useBias)
	
	local inputs = {}
	
	table.insert(inputs, nn.Identity()())
	
	local output = activation(nn.Linear(dim, dim, useBias)(inputs[1]))
	
	local transformGate = nn.Sigmoid()(nn.AddConstant(-2)(nn.Linear(dim, dim, useBias)(inputs[1])))
	
	local carryGate     = nn.AddConstant(1)(nn.MulConstant(-1)(transformGate))
	
	output = nn.CAddTable()({
		nn.CMulTable()({transformGate, output}),
		nn.CMulTable()({carryGate, inputs[1]}) })
		
	return nn.gModule(inputs, {output})

end

