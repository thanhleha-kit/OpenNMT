--
--  Copyright (c) 2015, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Author: Marc'Aurelio Ranzato <ranzato@fb.com>
--          Sumit Chopra <spchopra@fb.com>
--          Michael Auli <michaelauli@fb.com>
--          Wojciech Zaremba <zaremba@cs.nyu.edu>
--

local Sample, parent = torch.class('nn.MultinomialSample', 'nn.Module')
-- Module that takes a tensor storing log-probabilities (output of a LogSoftmax)
-- and samples from the corresponding multinomial distribtion.
-- Assumption: this receives input from a LogSoftMax and receives gradients from
-- a ReinforceCriterion.
function Sample:__init()
   parent.__init(self)
   self.prob = torch.Tensor()
   self.gradInput = {}
   self.running = true
   self.weight = nil
   self.bias = nil
   self.gradWeight = nil
   self.gradBias = nil
end

function Sample:enable()
	self.running = true
end

function Sample:disable()
	self.running = false
end

function Sample:evaluate()
	parent.evaluate(self)
	self.running = false
end

function Sample:updateOutput(input) -- input here is a table of outputs for generator

	assert(#input == 1, 'Features with Reinforce are currently not supported yet')
    local wordDist = input[1]
    
	self.prob:resizeAs(wordDist)
	self.prob:copy(wordDist)
	self.prob:exp()
	self.output:resize(wordDist:size(1), 1):fill(0)
	
	if self.running == true then
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
    
    
	self.output:resize(wordDist:size(1))
	self.prob:set() -- to clear memory
	return self.output -- batchSize
end

-- NOTE: in order for this to work, it has to be connected
-- to a ReinforceCriterion.
function Sample:updateGradInput(input, gradOutput)

	for n = 1, #input do
		self.gradInput[n] = self.gradInput[n] or input[n].new()
		self.gradInput[n]:resizeAs(input[n])
		self.gradInput[n]:zero()
	end
	
	if self.running == true then
		-- loop over mini-batches and build sparse vector of gradients
		-- such that each sample has a vector of gradients that is all 0s
		-- except for the component corresponding to the chosen word.
		-- We assume that the gradients are provided by a ReinforceCriterion.
		for b = 1, self.gradInput[1]:size(1) do
		  -- adding round because sometimes multinomial returns a float 1e-6 far
		  -- from an integer.
		  self.gradInput[1][b][torch.round(self.output[b])] =
			  gradOutput[b]
		end
	end
	
	return self.gradInput
   
end
