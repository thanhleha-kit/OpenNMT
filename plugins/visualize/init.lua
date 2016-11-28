-- svg visualization of the neural network for educative purpose

require 'paths'
json = require('json')

local Extension = {
  hooks = {},
  model_opt = {},
  neuron_dim = 5,
  layer_spacing = 15,
  id = 0
}

local function to_table(t)
  local T = {}
  for i=1,t:size(1) do
    table.insert(T, math.floor(t[i]*1000)/1000)
  end
  return T
end

local function generateJSON(params)
  local file = io.open(paths.concat(Extension.dir,Extension.prefix..'-'..Extension.id..".json"), "w")
  if Extension.encoder_layers == nil then
    local model = Extension.model
    Extension.encoder_layers = {}
    Extension.word_vecs = {}
    for t=1, #model.encoder.network_clones do
      Extension.encoder_layers[t]={}
      model.encoder:net(t):apply(function(m)
        if m.name=='lstm' then table.insert(Extension.encoder_layers[t],m) end
        if m.name=='word_vecs' then table.insert(Extension.word_vecs,m) end
      end)
    end
  end
  if Extension.decoder_layers == nil then
    local model = Extension.model
    Extension.decoder_layers = {}
    for t=1, #model.decoder.network_clones do
      Extension.decoder_layers[t]={}
      model.decoder:net(t):apply(function(m)
        if m.name=='lstm' then table.insert(Extension.decoder_layers[t],m) end
      end)
    end
  end

  local net={}
  local batch = params.batch
  for t = 1,batch.source_length do
    local word=Extension.src_dict:lookup(batch.source_input[t][1])
    table.insert(net, {type='word', mod='src', idx=t, value=word})
    table.insert(net, {type='lkp', mod='src', idx=t, value=to_table(Extension.word_vecs[t].output[1])})

    local s = Extension.neuron_dim
    for i=1,Extension.model_opt.num_layers do
      table.insert(net, {type='lstm', mod='src', idx=t, level=i, value=to_table(Extension.encoder_layers[t][i].output[1][1])})
    end

  end

  for t = 1,batch.target_length do
    local h = Extension.decoder_height

    local s = Extension.neuron_dim
    for i=1,Extension.model_opt.num_layers do
      h=h-Extension.layer_spacing-Extension.neuron_dim
      table.insert(net, {type='lstm', mod='tgt', idx=t, level=i, value=to_table(Extension.decoder_layers[t][i].output[1][1])})
    end

    local word=Extension.meanings[t]
    table.insert(net, {type='word', mod='tgt', idx=t, value=word})
    word=Extension.targ_dict:lookup(batch.target_input[t][1])
    table.insert(net, {type='word', mod='ref', idx=t, value=word})
  end
  file:write(json.encode(net))
  file:close()
  Extension.id = Extension.id+1
end

-- record generator output
local function record_tok_generation(params)
  local max, index = params.pred[1]:max(1)
  Extension.meanings[params.t] = Extension.targ_dict:lookup(index[1])
end

local function modelInitalized(params)
  local model = params.model
  local opt = params.opt
  Extension.model_opt['rnn_size'] = opt.rnn_size
  Extension.model_opt['num_layers'] = opt.num_layers
  Extension.model = model
  Extension.encoder_layers=nil
  Extension.decoder_layers=nil
  Extension.src_dict = params.dataset.src_dict
  Extension.targ_dict = params.dataset.targ_dict
  Extension.meanings = {}
  Extension.decoder_height = opt.num_layers*(Extension.neuron_dim+Extension.layer_spacing)+2*Extension.layer_spacing+1*40+50
  Extension.height = Extension.decoder_height+opt.num_layers*(Extension.neuron_dim+Extension.layer_spacing)+3*Extension.layer_spacing+2*40+20
end

function Extension.init(opt)
  Extension.hooks['training:after_batch'] = generateJSON
  Extension.hooks['decoder:tok_generation'] = record_tok_generation
  Extension.hooks['model_initialized'] = modelInitalized
  Extension.dir = opt['visualize:dir']
  Extension.prefix = os.date("%y%m%d_%X")
  Extension.encoder_layers = {}
  Extension.id = 1
end

function Extension.registerOptions(cmd)
  cmd:option('-visualize:dir', '', [[directory where svg are stored]])
end

return Extension
