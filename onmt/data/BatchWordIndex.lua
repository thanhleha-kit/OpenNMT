local BatchWordIndex = torch.class('BatchWordIndex')

--[[
  Take Batch x TimeStep  tensors
  This should be the output of sampling
]]
function BatchWordIndex:__init(T, batchFirst)
  
  batchFirst = batchFirst or false
  
  self.t = T
  
  if batchFirst then
	self.t = self.t:transpose(1, 2)
  end
  
  assert(self.t:dim() == 2, 'Expecting two dimensional input')
  
  self.sourceLength = T:size()[2]
  self.sourceSize = torch.LongTensor(T:size()[1]):fill(self.sourceLength)
  self.size = T:size()[1]
  
  self.sourceInputPadLeft = true
  self.sourceInputRevPadLeft = false
  
  -- find the true length of the sequence
  for b = 1, self.size do
	
	for tt = 1, self.sourceLength do
		if self.t[tt][b] == onmt.Constants.EOS then
			self.sourceSize[b] = tt
			break
		end
	end
  end
  
  
end

function BatchWordIndex:getSourceInput(t)
  return self.t:select(2,t)
end

return BatchWordIndex
