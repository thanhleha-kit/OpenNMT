local LUTG, parent = torch.class('nn.LookupTableWithGrad', 'nn.LookupTable')

-- just initialize like normal
function LUTG:__init(nIndex, nOutput, paddingValue, maxNorm, normType)
	parent.__init(self, nIndex, nOutput, paddingValue, maxNorm, normType)
	self.nIndex = nIndex
end

function LUTG:updateGradInput(input, gradOutput)
	
	-- the input can be of any type (as in the forward it's
	-- converted anyway to LongTensor) thus, need to allocate
	-- new memory each time the user changes the input type
	if torch.type(self.gradInput) ~= torch.type(input) then
		self.gradInput = input.new()
	end
	
	
	self.gradInput:resizeAs(input:size(1), self.nIndex)
	self.weight = self.weight:transpose(1, 2) -- d x V
	
	self.gradInput:addmm(gradOutput, self.weight) -- [ b x d ] x [ d x V ]
	
	self.weight = self.weight:transpose(1, 2) -- V x d
	
	return self.gradInput
end


