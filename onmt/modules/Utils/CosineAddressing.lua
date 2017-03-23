--~ -- Content-based addressing based on cosine similarity between the query and the memory entries --

local CosineAddressing, parent = torch.class('onmt.CosineAddressing','onmt.Network')

function CosineAddressing:__init()
	
	parent.__init(self, self:_buildModel())
end

--build the nn Graph. 
-- Simply concatenate two vectors, 
-- Then use a linear transformation, with an activation followed
function CosineAddressing:_buildModel()
	
	local inputs = {}
	
	table.insert(inputs, nn.Identity()()) -- query | batchSize x dim
	table.insert(inputs, nn.Identity()()) -- memory | batchSize x memSize x dim
	
	local targetT = inputs[1]
	local context = inputs[2]
	
	local attn = nn.MM()({context, nn.Replicate(1,3)(targetT)})
	attn = nn.Sum(3)(attn)
	
	return nn.gModule(inputs, {attn})
end
