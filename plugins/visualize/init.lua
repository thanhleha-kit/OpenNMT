-- svg visualization of the neural network for educative purpose

require 'paths'

local Extension = {
  hooks = {},
  model_opt = {},
  neuron_dim = 5,
  layer_spacing = 15,
  id = 0
}

local function getColorActivation(v)
  local color
  if v > 0 then
    color = 'fill:#FFFF00'
  else
    color = 'fill:#00FFFF'
  end
  v = math.abs(v) * 10
  if v > 1 then v = 1 end
  return color .. ';opacity:' .. v
end

local function renderVector(px, py, t)
  local svg = "<rect x='"..(px-1).."' y='"..(py-1).."' width='42' height='42' stroke='white' stroke-width='1' />"
  for i=1, 10 do
    for j=1, 10 do
      local color=getColorActivation(t[1][i+(j-1)*10])
      svg = svg .. "<rect x='"..(px+(i-1)*4).."' y='"..(py+(j-1)*4).."' width='4' height='4' style='"..color..";stroke-width:0' />"
    end
  end
  return svg
end

-- protect html entities in word
local function protect(word)
  word = string.gsub(word, "&", "&amp;")
  word = string.gsub(word, "<", "&lt;")
  word = string.gsub(word, ">", "&gt;")
  return word
end

local function generateSVG(params)
  local file = io.open(paths.concat(Extension.dir,Extension.prefix..'-'..Extension.id..".json"), "w")
  file:write("[\n")
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

  local batch = params.batch
  for t = 1,batch.source_length do
    local word=protect(Extension.src_dict:lookup(batch.source_input[t][1]))
    local h=Extension.height-20
    file:write("  [ \"src_"..t.."\", \"<text x='"..((t-1)*55+30).."' y='"..h.."' text-anchor='middle' fill='white' font-size='10px'>"..word.."</text>\"],\n")
    h=h-10-Extension.layer_spacing-40
    file:write("  [ \"lkp_"..t.."\", \""..renderVector(10+(t-1)*55,h,Extension.word_vecs[t].output).."\"],\n")

    local s = Extension.neuron_dim
    for i=1,Extension.model_opt.num_layers do
      h=h-Extension.layer_spacing-Extension.neuron_dim
      file:write("  [ \"srclstm_"..i.."\", \"")
      for j=1,Extension.model_opt.rnn_size do
        -- if we are doing a batch, take only the first sentence
        local value=Extension.encoder_layers[t][i].output[1][1][j]
        local color=getColorActivation(value)
        file:write("<rect x='"..(s*j).."' y='"..h.."' width='"..s.."' height='"..s.."' style='"..color..";stroke-width:0' />")
      end
      file:write("\"],\n")
    end

    h=h-Extension.layer_spacing-40
    file:write("  [ \"contextsrc"..t.."\", \""..renderVector(10+(t-1)*55,h,Extension.encoder_layers[t][Extension.model_opt.num_layers].output[1]).."\"],\n")
  end

  for t = 1,batch.target_length do
    local h = Extension.decoder_height

    local s = Extension.neuron_dim
    for i=1,Extension.model_opt.num_layers do
      h=h-Extension.layer_spacing-Extension.neuron_dim
      file:write("  [ \"tgtlstm_"..i.."\", \"");
      for j=1,Extension.model_opt.rnn_size do
        -- if we are doing a batch, take only the first sentence
        local value=Extension.decoder_layers[t][i].output[1][1][j]
        local color=getColorActivation(value)
        file:write("<rect x='"..(s*j).."' y='"..h.."' width='"..s.."' height='"..s.."' style='"..color..";stroke-width:0' />")
      end
      file:write("\"],\n")
    end

    h=h-Extension.layer_spacing-40
    file:write("  [ \"contexttgt"..t.."\", \""..renderVector(10+(t-1)*55,h,Extension.decoder_layers[t][Extension.model_opt.num_layers].output[1]).."\"],\n")
    h=h-20
    local word=protect(Extension.meanings[t])
    file:write("  [ \"pred_"..t.."\", \"<text x='"..((t-1)*55+30).."'' y='"..h.."' text-anchor='middle' fill='red' font-size='10px'>"..word.."</text>\"],\n")
    h=h-20
    word=protect(Extension.targ_dict:lookup(batch.target_input[t][1]))
    file:write("  [ \"ref_"..t.."\", \"<text x='"..((t-1)*55+30).."'' y='"..h.."' text-anchor='middle' fill='blue' font-size='10px'>"..word.."</text>\"]")
    if t ~= batch.target_length then file:write(",") end
    file:write("\n")
  end
  file:write("]")
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
  Extension.hooks['training:after_batch'] = generateSVG
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
