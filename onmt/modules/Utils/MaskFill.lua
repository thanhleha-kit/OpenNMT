
--

local MaskFill, parent = torch.class('nn.MaskFill','nn.Module')
-- This is like Linear, except that it does not backpropagate gradients w.r.t.
-- input.
function MaskFill:__init(value)
	parent.__init(self)
	self.value = value
end

function MaskFill:updateOutput(input)
	
	local inputTensor, mask = unpack(input)
	
	self.output = self.output or inputTensor.new()
	
	self.output:resizeAs(inputTensor):copy(inputTensor)
	
	self.output:maskedFill(mask, self.value)
	
	return self.output
end


function MaskFill:updateGradInput(input, gradOutput)

	local inputTensor, mask = unpack(input)
	
	self.gradInput = self.gradInput or gradOutput.new()
	
	self.gradInput:resizeAs(gradOutput):copy(gradOutput)
	
	self.gradInput:maskedFill(mask, 0.) -- the masked position has 0 gradient
	
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
