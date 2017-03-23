require('nngraph')

local MemoryWriter, parent = torch.class('onmt.MemoryWriter', 'onmt.Network')


function MemoryWriter:__init(hiddenSize, memorySize, nMemSlot)
	
	self.hiddenSize = hiddenSize
	self.memorySize = memorySize
	self.nMemSlot = nMemSlot
	
	parent.__init(self, self:_buildModel(hiddenSize, memorySize, nMemSlot))

end

-- input: current memory and the 
function MemoryWriter:_buildModel(hiddenSize, memorySize, nMemSlot)
	
	local inputs = {}
	
	table.insert(inputs, nn.Identity()())
	table.insert(inputs, nn.Identity()())
	
	local input = inputs[1]
	local currentMemory = inputs[2] 
	
	local transformedInput = nn.Linear(hiddenSize, memorySize, false)(input)
	
	-- we can think about forcing these weights to be the same with reading and writing 
	-- but later
	local writingWeights = onmt.CosineAddressing()({transformedInput, currentMemory})
	writingWeights = nn.SoftMax()(writingWeights)
	writingWeights = nn.Replicate(memorySize, 3)(writingWeights) -- batchSize x nMemSlot x mSize
	-- batchSize x nMemSlot 
	
	
	local eraseValue = nn.Sigmoid()(nn.Linear(hiddenSize, memorySize, false)(input))
	eraseValue = nn.MulConstant(-1, false)(eraseValue)
	eraseValue = nn.AddConstant(1, false)(eraseValue)
	
	eraseValue = nn.Replicate(nMemSlot, 2)(eraseValue) -- batchSize x nMemSlot x mSize
	
	local memoryToBeErased = nn.CMulTable()({writingWeights, eraseValue})
	
	local addValue = nn.Sigmoid()(nn.Linear(hiddenSize, memorySize, false)(input))
	addValue = nn.Replicate(nMemSlot, 2)(addValue)
	
	local memoryAdded = nn.CMulTable()({addValue, writingWeights})
	
	
	local newMemory = nn.CAddTable()({currentMemory, memoryToBeErased, memoryAdded})
	
	return nn.gModule(inputs, {newMemory})
	
	
	
end

