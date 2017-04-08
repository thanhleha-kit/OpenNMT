--[[
  Define parallel ClassNLLCriterionWeighted.
  To maintain backward compatibility with main branch
--]]
local ParallelClassNLLCriterionWeighted, parent = torch.class('onmt.ParallelClassNLLCriterionWeighted', 'nn.ParallelCriterion')

function ParallelClassNLLCriterionWeighted:__init(globalWeight, outputSizes)
  parent.__init(self, false)

  for i = 1, #outputSizes do
    self:_addCriterion(outputSizes[i])
  end
end

function ParallelClassNLLCriterionWeighted:_addCriterion(size)
  -- Ignores padding value.
  local w = torch.ones(size)
  w[onmt.Constants.PAD] = 0
  
  local nll = nn.ClassNLLCriterionWeighted(globalWeight, w)

  -- Let the training code manage loss normalization.
  nll.sizeAverage = false
  self:add(nll)
end

-- reweighting the criterions 
function ParallelClassNLLCriterionWeighted:setWeight(w)
	
	for i,criterion in ipairs(self.criterions) do
		criterion.globalWeight = w
	end
end
