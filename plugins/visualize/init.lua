-- generate a JSON dump of the neural network for education visualization purpose
-- require torch json library (luarocks install --server=http://rocks.moonscript.org/manifests/amrhassan --local json4Lua)

require 'paths'
json = require('json')

local Extension = {
  hooks = {},
  model_opt = {},
  id = 0
}

local function to_table(t)
  local T = {}
  for i=1,t:size(1) do
    table.insert(T, math.floor(t[i]*500)/100)
  end
  return T
end

local function to_tableL(L,size)
  local T = {}
  for j=1,size do
    local R={}
    for i=1,4*size do
      table.insert(R, math.floor(math.abs(L[1][i][j]*50)))
    end
    for i=1,4 do
      table.insert(R, math.floor(math.abs(L[2][(j-1)*4+i]*50)))
    end
    for i=1,4*size do
      table.insert(R, math.floor(math.abs(L[3][i][j]*50)))
    end
    table.insert(T,R)
  end
  return T
end

local function generateJSON(params)
  if Extension.encoder_layers == nil then
    local model = Extension.model
    Extension.encoder_layers = {}
    Extension.word_vecs = {}
    for t=1, #model.encoder.network_clones do
      Extension.encoder_layers[t]={}
      model.encoder:net(t):apply(function(m)
        if m.name == 'lstm' then
          table.insert(Extension.encoder_layers[t],m)
        end
        if m.name == 'word_vecs' then table.insert(Extension.word_vecs,m) end
      end)
    end
  end
  if Extension.decoder_layers == nil then
    local model = Extension.model
    Extension.decoder_layers = {}
    Extension.softmax_attns = {}
    for t=1, #model.decoder.network_clones do
      Extension.decoder_layers[t]={}
      model.decoder:net(t):apply(function(m)
        if m.name == 'lstm' then
          table.insert(Extension.decoder_layers[t],m)
        end
        if m.name == 'softmax_attn' then table.insert(Extension.softmax_attns, m) end
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
      local p, gp=Extension.encoder_layers[1][i]:parameters()
      table.insert(net, {type='lstm', mod='src', lstmcell='params', idx=t, level=i, value=to_tableL(p,Extension.model_opt.rnn_size)})
    end
    table.insert(net, {type='lstm', mod='src', lstmcell='c', idx=t, level=Extension.model_opt.num_layers,
                                      value=to_table(Extension.encoder_layers[t][Extension.model_opt.num_layers].output[1][1])})
  end

  for t = 1,batch.target_length-1 do
    local tgtword=Extension.meanings[t]
    local refword=Extension.targ_dict:lookup(batch.target_input[t+1][1])
    if refword == '<blank>' then break end
    table.insert(net, {type='attn', idx=t, value=to_table(Extension.softmax_attns[t].output[1])})
    for i=1,Extension.model_opt.num_layers do
      local p, gp=Extension.decoder_layers[1][i]:parameters()
      table.insert(net, {type='lstm', mod='tgt', lstmcell='params', idx=t, level=i, value=to_tableL(p,Extension.model_opt.rnn_size)})
    end

    table.insert(net, {type='lstm', mod='tgt', lstmcell='c', idx=t, level=Extension.model_opt.num_layers,
                                     value=to_table(Extension.decoder_layers[t][Extension.model_opt.num_layers].output[1][1])})

    table.insert(net, {type='word', mod='tgt', idx=t, value=tgtword})
    table.insert(net, {type='word', mod='ref', idx=t, value=refword})
  end
  local file = io.open(paths.concat(Extension.dir,Extension.prefix..'-'..params.epoch..':'..params.idx..".json"), "w")
  local snapshot = { net=net, epoch=params.epoch, idx=params.idx, loss=params.loss, opt=Extension.model_opt}
  file:write(json.encode(snapshot))
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
  cmd:option('-visualize:dir', '', [[directory where json are saved]])
end

return Extension
