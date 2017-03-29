-- 
-- Copyright (c) 2015, Facebook, Inc.
-- All rights reserved.
-- 
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.
-- 
-- Author: Marc'Aurelio Ranzato <ranzato@fb.com>
-- Sumit Chopra <spchopra@fb.com>
-- Michael Auli <michaelauli@fb.com>
-- Wojciech Zaremba <zaremba@cs.nyu.edu>
--

local GanSampler, parent = torch.class('onmt.GanSampler', 'nn.Module')

-- Module that takes a tensor storing log-probabilities (output of a LogSoftmax)
-- and samples from the corresponding multinomial distribtion.
-- Assumption: this receives input from a LogSoftMax and receives gradients from
-- a ReinforceCriterion.
function GanSampler:__init(distribution)
	parent.__init(self)
	self.distribution = distribution or 'multinomial'
	self.prob = torch.Tensor()
	
	self.outputProto = torch.Tensor()
	self.gradInput = {}
	self.enable = true
	self.argmax = false
	
	assert(self.distribution == 'multinomial', 'Only multinomial sampling is supported')
end

function GanSampler:Enable()
	--~ _G.logger:info("Debugging: Enabling set to true")
	self.enable = true
end

function GanSampler:Disable()
	self.enable = false
end


-- Input is the output of the logsoftmax function (at each decoding step)
function GanSampler:updateOutput(input)

	if torch.type(input) == 'table' then
		input = input[1]
	end
	

	local batchSize = input:size(1)
	
	
	if torch.type(input) == 'table' then
		input = input[1]
	end
	-- Multinomial Sampling
	self.prob:resizeAs(input)
	self.prob:copy(input)
	self.prob:exp() -- to get probability distribution
	
	self.output:resize(batchSize, 1):zero() -- batch_size * 1
	
	if self.enable == true then
	
		-- switch to correct type
		if torch.typename(self.output):find('torch%.Cuda.*Tensor') then
			self.output = self.output:cudaLong()
		else
			self.output = self.output:long()
		end
		
		self.prob.multinomial(self.output, self.prob, 1)
		
		if torch.typename(self.output):find('torch%.Cuda.*Tensor') then
			self.output = self.output:cuda()
		else
			self.output = self.output:float()
		end
	end
	
	self.output:resize(batchSize)
	
	self.prob:set()
		
	return self.output -- batch x 1
end


-- the sampler receives the gradient w.r.t the sampled samples
-- Reinforce criterion from FAIR is an example
-- gradOutput has size: batchSize 
function GanSampler:updateGradInput(input, gradOutput)

	-- to  be compatible with factored neural machine translation
	if torch.type(input) == 'table' then
		assert(#input == 1, 'factored NMT currently not supported for GAN')
		
		for k = 1, #input do
			self.gradInput[k] = self.gradInput[k] or input[k].new()
			self.gradInput[k]:resizeAs(input[k])
			self.gradInput[k]:zero()
		end
	end
	
	-- only compute gradInput if we enabled sampling
	if self.enable == true then
	
		-- we only update gradInput for the first 
		for ss = 1, self.gradInput[1]:size(1) do
			
			--adding round because sometimes multinomial returns a float 1e-6 far from an integer.
			self.gradInput[1][ss][torch.round(self.output[ss])] = 
					gradOutput[ss]
		end
	
	end
	
	--~ print(self.gradInput[1]:norm())
	
	return self.gradInput

end


