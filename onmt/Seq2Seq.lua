--[[ Sequence to sequence model with attention. ]]
local Seq2Seq, parent = torch.class('Seq2Seq', 'Model')

local options = {
  {'-layers', 2,           [[Number of layers in the RNN encoder/decoder]],
                     {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-rnn_size', 500, [[Size of RNN hidden states]],
                     {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-rnn_type', 'LSTM', [[Type of RNN cell]],
                     {enum={'LSTM','GRU'}}},
  {'-word_vec_size', 0, [[Common word embedding size. If set, this overrides -src_word_vec_size and -tgt_word_vec_size.]],
                     {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-src_word_vec_size', '500', [[Comma-separated list of source embedding sizes: word[,feat1,feat2,...].]]},
  {'-tgt_word_vec_size', '500', [[Comma-separated list of target embedding sizes: word[,feat1,feat2,...].]]},
  {'-feat_merge', 'concat', [[Merge action for the features embeddings]],
                     {enum={'concat','sum'}}},
  {'-feat_vec_exponent', 0.7, [[When features embedding sizes are not set and using -feat_merge concat, their dimension
                                will be set to N^exponent where N is the number of values the feature takes.]]},
  {'-feat_vec_size', 20, [[When features embedding sizes are not set and using -feat_merge sum, this is the common embedding size of the features]],
                     {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-coverage', 0, [[Coverage vector size. 0 to disable this feature.]],
                     {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-input_feed', 1, [[Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.]],
                     {enum={0,1}}},
  {'-residual', false, [[Add residual connections between RNN layers.]]},
  {'-attention', 'global', [[Attention type: global|cgate. Global is the typical one, cgate is global with context gate]]},
  {'-brnn', false, [[Use a bidirectional encoder]]},
  {'-brnn_merge', 'sum', [[Merge action for the bidirectional hidden states]],
                     {enum={'concat','sum'}}},
  {'-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings on the encoder side.
                                     See README for specific formatting instructions.]],
                         {valid=onmt.utils.ExtendedCmdLine.fileNullOrExists}},
  {'-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load
                                       pretrained word embeddings on the decoder side.
                                       See README for specific formatting instructions.]],
                         {valid=onmt.utils.ExtendedCmdLine.fileNullOrExists}},
  {'-fix_word_vecs_enc', false, [[Fix word embeddings on the encoder side]]},
  {'-fix_word_vecs_dec', false, [[Fix word embeddings on the decoder side]]},
  {'-dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]]},
  {'-dropout_input', 0, [[Dropout probability on embedding (input of LSTM)]]},
  {'-tie_embedding', false, [[Tie the embedding layer and the linear layer of the output]]}
}

function Seq2Seq.declareOpts(cmd)
  cmd:setCmdLineOptions(options, Seq2Seq.modelName())
end

function Seq2Seq:buildCriterion(dicts)
	
	_G.scorer = onmt.utils.BLEU.new(dicts.tgt.words, 4, 1)
	self.weightXENT = 1.
	self.criterion = onmt.ParallelClassNLLCriterionWeighted(self.weightXENT, onmt.Factory.getOutputSizes(dicts.tgt))
	self.weightRF = 1. - self.weightXENT 
	self.rfcriterion = onmt.ReinforceCriterion(_G.scorer, onmt.Constants.MAX_TARGET_LENGTH, self.nStepInits, self.weightRF)
end

function Seq2Seq:__init(args, dicts, verbose)
  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, options))

  self.models.encoder = onmt.Factory.buildWordEncoder(args, dicts.src, verbose)
  self.models.decoder = onmt.Factory.buildWordDecoder(args, dicts.tgt, verbose)
  --~ self.criterion = onmt.ParallelClassNLLCriterion(onmt.Factory.getOutputSizes(dicts.tgt))
  
  self:buildCriterion(dicts)
end

function Seq2Seq.load(args, models, dicts, isReplica)
  local self = torch.factory('Seq2Seq')()

  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, options))

  self.models.encoder = onmt.Factory.loadEncoder(models.encoder, isReplica)
  self.models.decoder = onmt.Factory.loadDecoder(models.decoder, isReplica)
  --~ self.criterion = onmt.ParallelClassNLLCriterion(onmt.Factory.getOutputSizes(dicts.tgt))
  
  self:buildCriterion(dicts)

  return self
end

-- Returns model name.
function Seq2Seq.modelName()
  return 'Sequence to Sequence with Attention'
end

-- Returns expected dataMode.
function Seq2Seq.dataType()
  return 'bitext'
end

function Seq2Seq:enableProfiling()
  _G.profiler.addHook(self.models.encoder, 'encoder')
  _G.profiler.addHook(self.models.decoder, 'decoder')
  _G.profiler.addHook(self.models.decoder.modules[2], 'generator')
  _G.profiler.addHook(self.criterion, 'criterion')
end

function Seq2Seq:getOutput(batch)
  return batch.targetOutput
end

function Seq2Seq:forwardComputeLoss(batch)
	self:maskPadding(batch)
  self.criterion:setWeight(1.)
  local encoderStates, context = self.models.encoder:forward(batch)
  local loss = self.models.decoder:computeLoss(batch, encoderStates, context, self.criterion)
  self.criterion:setWeight(self.weightXENT)
  return loss
end

function Seq2Seq:maskPadding(batch)

	--~ if batch.uneven == true then self.models.encoder:maskPadding() end
	if self.sortTarget == true then
		
		
		--~ self.models.decoder:maskPadding(batch.sourceSize, batch.sourceLength)
		--~ 
			--~ 
		if batch.uneven == true then 
			self.models.encoder:maskPadding()
			--~ self.models.decoder:maskPadding(batch.sourceSize, batch.sourceLength)
		else
			--~ self.models.decoder:maskPadding()
		end
	end
end

function Seq2Seq:trainNetwork(batch, dryRun)

	self:maskPadding(batch)
  local encStates, context = self.models.encoder:forward(batch)

  local decOutputs = self.models.decoder:forward(batch, encStates, context)

  if dryRun then
    decOutputs = onmt.utils.Tensor.recursiveClone(decOutputs)
  end

  local encGradStatesOut, gradContext, lossXENT, lossRF, numSamplesXENT, numSamplesRF, totCumRewardPredError = self.models.decoder:backward(batch, decOutputs, self.criterion, self.rfcriterion)
  self.models.encoder:backward(batch, encGradStatesOut, gradContext)
  
  --~ if lossXENT ~= lossXENT then
	--~ print('loss is NaN.  This usually indicates a bug.')
	--~ os.exit(0) 
  --~ end
  
  -- so that the perplexity report will be correct
  return lossXENT / self.weightXENT, lossRF / (self.weightRF + 1e-6), numSamplesXENT, numSamplesRF, totCumRewardPredError
end

function Seq2Seq:sampleBatch(batch, maxLength, argmax)
	self:maskPadding(batch)
	
	local encStates, context = self.models.encoder:forward(batch)
	
	local sampledBatch = self.models.decoder:sampleBatch(batch, encStates, context, maxLength, argmax)
	
	return sampledBatch
end

function Seq2Seq:setNSamplingSteps(nstep)
	
	self.models.decoder:setNSamplingSteps(nstep)
end

-- reweighting the criterions
function Seq2Seq:setWeight(w)

	self.weightRF = w
	self.weightXENT = 1. - w
	
	self.rfcriterion:setWeight(w)
	self.criterion:setWeight(1. - w)
	

end

return Seq2Seq
