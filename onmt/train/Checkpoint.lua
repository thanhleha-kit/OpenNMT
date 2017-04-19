-- Class for saving and loading models during training.
local Checkpoint = torch.class('Checkpoint')

local options = {
  {'-train_from', '',  [[If training from a checkpoint then this is the path to the pretrained model.]],
                         {valid=onmt.utils.ExtendedCmdLine.fileNullOrExists}},
  {'-continue', false, [[If training from a checkpoint, whether to continue the training in the same configuration or not.]]}
}

function Checkpoint.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Checkpoint')
end

function Checkpoint:__init(opt, model, optim, dicts)
  self.options = opt
  self.model = model
  self.optim = optim
  self.dicts = dicts

  self.savePath = self.options.save_model
end

function Checkpoint:save(filePath, info)
  info.learningRate = self.optim:getLearningRate()
  info.optimStates = self.optim:getStates()

  local data = {
    models = {},
    options = self.options,
    info = info,
    dicts = self.dicts
  }

  for k, v in pairs(self.model.models) do
    if v.serialize then
      data.models[k] = v:serialize()
    else
      data.models[k] = v
    end
  end

  torch.save(filePath, data)
end

--[[ Save the model and data in the middle of an epoch sorting the iteration. ]]
function Checkpoint:saveIteration(iteration, totalIteration, epochState, batchOrder, validPpl, validBleu, verbose)
  local info = {}
  info.iteration = iteration + 1
  info.epoch = epochState.epoch + iteration / totalIteration - 1
  info.batchOrder = batchOrder

  local filePath = string.format('%s_checkpoint_epoch%.2f_ppl=%.2f_bleu=%.2f.t7', self.savePath, info.epoch, validPpl, validBleu)

  if verbose then
    _G.logger:info('Saving checkpoint to \'' .. filePath .. '\'...')
  end

  -- Succeed serialization before overriding existing file
  self:save(filePath .. '.tmp', info)
  os.rename(filePath .. '.tmp', filePath)
end

function Checkpoint:saveEpoch(validPpl, validBleu, epochState, verbose)
  local info = {}
  info.validPpl = validPpl
  info.epoch = epochState.epoch + 1
  info.iteration = 1
  info.trainTimeInMinute = epochState:getTime() / 60

  local filePath = string.format('%s_epoch%d_ppl=%.2f,bleu=%.2f.t7', self.savePath, epochState.epoch, validPpl, validBleu)

  if verbose then
    _G.logger:info('Saving checkpoint to \'' .. filePath .. '\'...')
  end

  self:save(filePath, info)
end

function Checkpoint.loadFromCheckpoint(opt)
  local checkpoint = {}
  if opt.train_from:len() > 0 then
    _G.logger:info('Loading checkpoint \'' .. opt.train_from .. '\'...')

    checkpoint = torch.load(opt.train_from)

		opt.layers = checkpoint.options.layers
		opt.rnn_size = checkpoint.options.rnn_size
		opt.brnn = checkpoint.options.brnn
		opt.brnn_merge = checkpoint.options.brnn_merge
		opt.input_feed = checkpoint.options.input_feed
		opt.word_vec_size = checkpoint.options.word_vec_size
		opt.rnn_type = checkpoint.options.rnn_type
		opt.feat_merge = checkpoint.options.feat_merge
		opt.feat_vec_exponent = checkpoint.options.feat_vec_exponent
		opt.coverage = checkpoint.options.coverage
		opt.attention = checkpoint.options.attention
		opt.dropout = checkpoint.options.dropout

    -- Resume training from checkpoint
    if opt.continue then
      opt.optim = checkpoint.options.optim
      opt.learning_rate_decay = checkpoint.options.learning_rate_decay
      opt.start_decay_at = checkpoint.options.start_decay_at
      opt.curriculum = checkpoint.options.curriculum

      opt.learning_rate = checkpoint.info.learningRate
      opt.start_epoch = checkpoint.info.epoch
      opt.start_iteration = checkpoint.info.iteration

      _G.logger:info('Resuming training from epoch ' .. opt.start_epoch
                         .. ' at iteration ' .. opt.start_iteration .. '...')
    end
  end
  return checkpoint, opt
end

return Checkpoint
