require('nngraph')

local MemoryReader, parent = torch.class('onmt.MemoryReader', 'onmt.Network')


function MemoryReader:__init(hiddenSize, memorySize, nMemSlot, dropout)
	
	self.hiddenSize = hiddenSize
	self.memorySize = memorySize
	self.nMemSlot = nMemSlot
	dropout = dropout or 0
	
	parent.__init(self, self:_buildModel(hiddenSize, memorySize, nMemSlot, dropout))

end

function MemoryReader:_buildModel(hiddenSize, memorySize, nMemSlot, dropout)
	
	
	local inputs = {}
	local outputs = {}
	
	table.insert(inputs, nn.Identity()()) -- query: the hidden state from RNN
	table.insert(inputs, nn.Identity()()) -- the Memory matrix | batchSize x nSlot x memSize
	
	local query = inputs[1]
	local memory = inputs[2]
	
	--~ query = nn.Dropout(dropout)(query)
	
	local transformedQuery = nn.Linear(hiddenSize, memorySize, false)(query)
	
	local memWeight = nn.MM()({memory, nn.Replicate(1,3)(transformedQuery)})
	memWeight = nn.Sum(3)(memWeight)
	memWeight = nn.SoftMax()(memWeight)
	local weight = memWeight
	memWeight = nn.Replicate(1, 2)(memWeight) 
	
	local memCombined = nn.Sum(2)(nn.MM()({memWeight, memory}))
	
	--~ local queryOutput = onmt.GRUModule(memorySize, hiddenSize, dropout)({query, memCombined})
	--~ 
	memCombined = nn.Tanh()(nn.Linear(memorySize, hiddenSize, false)(memCombined))
	--~ 
	
	-- a tanh combination between the query and the memory
	local queryOutput = onmt.JoinLinear(hiddenSize, nn.Tanh)({query, memCombined})
	
	--~ return nn.gModule(inputs, {queryOutput, weight})
	return nn.gModule(inputs, {queryOutput})

end


