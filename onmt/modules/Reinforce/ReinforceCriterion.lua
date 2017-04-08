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
local ReinforceCriterion, parent = torch.class('onmt.ReinforceCriterion', 'nn.Criterion')

-- This criterion implements the REINFORCE algorithm under the assumption that
-- the reward does not depend on the model parameters.
-- The constructor takes as input a function which is used to compute the reward
-- given the ground truth input sequence, the generated sequence and the current
-- time step.
-- The input to the criterion is a table whose entries are the output of the
-- RNN at a certain time step, namely:
-- (chosen_word, predicted_cumulative_reward)_t
-- It computes the total reward and bprop the derivative
-- w.r.t. the above provided inputs.
-- rewarder: user provided function to compute the reward
-- given ground truth, current sequence and current time step.
-- max_length is the length of the sequence we use
-- skips is the number of time steps we skip from the input and target (init)
-- weight is the weight on the loss produced by this criterion
-- weight_predictive_reward is the weight on the gradient of the cumulative
-- reward predictor (only)

function ReinforceCriterion:__init(rewarder, max_length,
								skips, weight,
								weight_predictive_reward)
	parent.__init(self)
	
	self.gradInput = {}
	self.max_length = max_length
	
	self.gradInput[1] = torch.Tensor()
	self.gradInput[2] = torch.Tensor()
	
	self.sizeAverage = true
	self.reward = torch.Tensor()
	self.cumreward = torch.Tensor()
	self.skips = (skips == nil) and 1 or skips
	assert(self.skips <= max_length)
	
	self.weight_predictive_reward =
		(weight_predictive_reward == nil) and 1 or weight_predictive_reward
	self.weight = (weight == nil) and 1 or weight
	
	self.num_samples = 0
	self.normalizing_coeff = 1
	self.reset = torch.Tensor()
	
	self.rewarder = rewarder
	
	
end


function ReinforceCriterion:type(tp)
	parent.type(self, tp)
	
	self.gradInput[1] = self.gradInput[1]:type(tp)
	self.gradInput[2] = self.gradInput[2]:type(tp)
	self.reward = self.reward:type(tp)
	self.cumreward = self.cumreward:type(tp)
	return self
end

function ReinforceCriterion:setWeight(ww)
	self.weight = ww
end

function ReinforceCriterion:setSkips(skips)
	self.skips = skips
end

-- input is a table storing the tuple
-- (sampled, pred_rewards)
-- sampled: tensor (max_length x batch_size) : label sequence sampled from the model
-- pred_rewards: tensor (max_length x batch_size): cumulative reward at each step
-- target is a tensor (nsteps (real) x batch_size): ground truth 
function ReinforceCriterion:updateOutput(input, target)
	
	local sampled = input[1]
	local pred_rewards = input[2]
	-- Sanity checks
	assert(sampled:size(2) == target:size(2)) -- batch_size must be the same
	assert(sampled:size(1) == pred_rewards:size(1)) 
	--~ assert(sampled:size(2) == seq_rewards:size(1))
	
	-- Getting sample-dependent parameters
	local batchSize = sampled:size(2)
	local seqLength = sampled:size(1)
	
	local skip = math.min(self.skips, seqLength)
	--~ pri
	local nSteps = seqLength - skip -- from 1 -> self.skips we don't sample
	self.reward:resize(seqLength, batchSize):zero()
	self.cumreward:resize(seqLength, batchSize):zero()
	
	-- Compute reward for each sequence in the batch
	local seqReward = onmt.utils.Cuda.convert(self.rewarder:computeScore(sampled, target))
	
	--~ print(seqReward)
		
	assert(sampled:size(2) == seqReward:size(1))
	-- Getting the real lengths of sampled sequences
	-- Default is seqLength (end of sequence) 
	local realLength = seqReward:clone():fill(seqLength)
	for b = 1, batchSize do
		for t = 1, seqLength do
			if sampled[t][b] == onmt.Constants.EOS then
				realLength[b] = t  
				break
			end
		end
		
		-- Rewarding only at the end of sentence
		-- We also want to reward the EOS token as well
		self.reward[realLength[b]][b] = seqReward[b]
	end
	
	--~ print(realLength)
	
	
	-- Rewarding only at the end of sentence
	-- We also want to reward the EOS token as well
	--~ for b = 1, batchSize do
		--~ self.reward[realLength[b]][b] = seqReward[b]
	--~ end
	
	--~ print(self.reward)
	
		
	for tt = seqLength, skip+1, -1 do
		
		-- getting cumulative rewards
		if tt == seqLength then
			self.cumreward[tt]:copy(self.reward[tt])
		else
			self.cumreward[tt]:add(self.cumreward[tt+1], self.reward[tt])
		end
	end
	--~ 
	--~ print(self.cumreward)
	
	
	--~ 
	
	-- we are considering sentence level rewards, so number of samples = number of sentences
	self.num_samples = batchSize -- simply number of sentences
	--~ self.reset:ne(sampled[1], onmt.Constants.PAD)
	--~ self.num_samples = self.reset:sum()
	
	
	self.normalizing_coeff = self.weight / (self.sizeAverage and self.num_samples or 1)
	
	if skip == seqLength then
		self.output = 0
	else
		self.output = - self.cumreward[skip+1]:sum() * self.normalizing_coeff 
	end
	
	--~ print (self.cumreward[skip+1])
	
	--~ print(self.output)
	return self.output, self.num_samples
end

-- input is a table storing the tuple
-- (chosen_word, predicted_cumulative_reward)_t, t = 1..T
-- sampled: tensor (max_length x batch_size) : label sequence sampled from the model
-- pred_rewards: tensor (max_length x batch_size): cumulative reward at each step
-- target is a tensor (nsteps (real) x batch_size): ground truth 
function ReinforceCriterion:updateGradInput(input, target)

	local sampled = input[1]
	local pred_rewards = input[2]
	local batchSize = target:size(2)
	local seqLength = sampled:size(1)
	
	self.cumreward = self.cumreward:cuda()
	
	local skip = math.min(self.skips, seqLength)
	
	self.gradInput[1]:resize(seqLength, batchSize):fill(0)
	self.gradInput[2]:resize(seqLength, batchSize):fill(0)
	
	-- going backwards
	for tt = seqLength, skip+1, -1 do
		
		--~ self.gradInput[1][tt]:resizeAs(sampled[tt]) -- -- derivative w.r.t. chosen action
		
		-- derivative through chosen action is:
		-- (predicted_cumulative_reward - actual_cumulative_reward)_t.
		self.gradInput[1][tt]:add(
			pred_rewards[tt], -1, self.cumreward[tt])
		self.gradInput[1][tt]:mul(self.normalizing_coeff)
		
		-- reset gradient to 0 if input has PAD
		self.reset:resize(batchSize)
		self.reset:ne(sampled[tt], onmt.Constants.PAD)
		self.gradInput[1][tt]:cmul(self.reset)
		-- copy over the other input gradient as well
		self.gradInput[2][tt]:resizeAs(pred_rewards[tt]) -- derivative w.r.t. the predicted reward
		self.gradInput[2][tt]:copy(self.gradInput[1][tt])
		self.gradInput[2][tt]:mul(self.weight_predictive_reward)
	end
	
	-- The gradients for non-sampling parts are zero
	
	--~ for tt = skip, 1, -1 do
		--~ self.gradInput[1][tt]:resizeAs(sampled[tt])
		--~ self.gradInput[1][tt]:fill(0)
		--~ self.gradInput[2][tt]:resizeAs(pred_rewards[tt])
		--~ self.gradInput[2][tt]:fill(0)
	--~ end
	
	return self.gradInput
	
end
