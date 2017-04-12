
--

local MaskFill, parent = torch.class('nn.MaskFill','nn.Module')
-- This is like Linear, except that it does not backpropagate gradients w.r.t.
-- input.
function MaskFill:__init(value)
	parent.__init(self)
	self.value = value
	self.gradInput = {}
end

function MaskFill:updateOutput(input)
	
	local inputTensor, mask = unpack(input)
	
	mask = mask:cudaByte()
	
	self.output = self.output or inputTensor.new()
	
	self.output:resizeAs(inputTensor):copy(inputTensor)
	
	self.output:maskedFill(mask, self.value)
	
	--~ print(self.output)
	
	return self.output
end


function MaskFill:updateGradInput(input, gradOutput)

	local inputTensor, mask = unpack(input)
	
	mask = mask:cudaByte()
	
	self.gradInput[1] = self.gradInput[1] or gradOutput.new()
	
	self.gradInput[1]:resizeAs(gradOutput):copy(gradOutput)
	
	--~ self.gradInput[1]:maskedFill(mask, 0.) -- the masked position has 0 gradient
	
	--~ self.gradInput[2] = self.gradInput[2] or mask.new()
	
	--~ self.gradInput[2]:resizeAs(mask):fill(0)
	
	return self.gradInput
end
--~ 
--~ function LinearNoBackpropInput:updateGradInput(input, gradOutput)
   --~ if self.gradInput then
      --~ self.gradInput:resizeAs(input)
      --~ self.gradInput:zero()
      --~ return self.gradInput
   --~ end
--~ end
