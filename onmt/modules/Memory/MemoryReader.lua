require('nngraph')

local MemoryReader, parent = torch.class('onmt.MemoryReader', 'onmt.Network')


function MemoryReader:__init(hiddenSize, memorySize, nMemSlot)
	
	self.hiddenSize = hiddenSize
	self.memorySize = memorySize
	self.nMemSlot = nMemSlot
	
	parent.__init(self, self:_buildModel(hiddenSize, memorySize, nMemSlot))

end

function MemoryReader:_buildModel(hiddenSize, memorySize, nMemSlot)
	
	
	local inputs = {}
	local outputs = {}
	
	table.insert(inputs, nn.Identity()()) -- query: the hidden state from RNN
	table.insert(inputs, nn.Identity()()) -- the Memory matrix | batchSize x nSlot x memSize
	
	local query = inputs[1]
	local memory = inputs[2]
	
	local transformedQuery = nn.Linear(hiddenSize, memorySize, false)(query)
	
	local memWeight = onmt.CosineAddressing()({transformedQuery, memory})
	memWeight = nn.SoftMax()(memWeight)
	memWeight = nn.Replicate(1, 2)(memWeight) 
	
	local memCombined = nn.Sum(2)(nn.MM()({memWeight, memory}))
	
	memCombined = nn.Linear(memorySize, hiddenSize, false)(memCombined)
	
	-- a tanh combination between the query and the memory
	local queryOutput = onmt.JoinLinear(hiddenSize, nn.Tanh)({query, memCombined})
	
	return nn.gModule(inputs, {queryOutput})

end


