import ipdb
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlxs2s.utils import check_file
import numpy as np

ACT2FN = {
    'silu': nn.SiLU,
    'gelu': nn.GELU,
}

AUDIO_TOKEN_INDEX = 151646  # hard code in Qwen2AudioConfig.__init__()

Qwen2Config = {
    'vocab_size': 151936,
    'hidden_size': 4096,
    'intermediate_size': 22016,
    'num_hidden_layers': 32,
    'num_attention_heads': 32,
    'num_key_value_heads': 32,
    'hidden_act': "silu",
    'max_position_embeddings': 32768,
    'initializer_range': 0.02,
    'rms_norm_eps': 1e-6,
    'use_cache': True,
    'tie_word_embeddings': False,
    'rope_theta':  10000.0,
    'use_sliding_window': False,
    'sliding_window': 4096,
    'max_window_layers': 28,
    'attention_dropout': 0.0,
    'pad_token_id': None,
}


def create_additive_causal_mask(N: int, offset: int = 0):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9

# class Qwen2RotaryEmbedding(nn.Module):
#     def __init__(self, dim, max_position_embeddings, base):
#         super().__init__()
#         inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
#         t = mx.arange(max_position_embeddings, dtype=mx.float32)
#         freqs = mx.outer(t, inv_freq)
#         emb = mx.concatenate([freqs, freqs], dim=-1)
#
#         self.cos_cached = emb.cos().astype(mx.bfloat16)
#         self.sin_cached = emb.sin().astype(mx.bfloat16)
#
#     def __call__(self, seq_len):
#         return (
#             self.cos_cached[:seq_len],
#             self.sin_cached[:seq_len]
#         )


class Qwen2AudioAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        is_decoder: bool = False,
        layer_idx=None,
        config=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_causal = False
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError((
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            ))
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def __call__(
        self,
        hidden_states,
    ):
        """Input shape: Batch x Time x Channel"""

        B, L, D = hidden_states.shape

        # TODO: fuse QKV for better performance
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scaling, mask=None
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        attn_output = self.out_proj(output)

        return attn_output


class Qwen2Attention(nn.Module):
    def __init__(self, config: dict, layer_idx):
        super().__init__()

        self.hidden_dim = config['hidden_size']
        self.n_heads = config['num_attention_heads']
        self.n_kv_heads = config['num_key_value_heads']

        self.layer_idx = layer_idx

        self.head_dim = self.hidden_dim // self.n_kv_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.rope = nn.RoPE(
                        self.head_dim,
                        traditional=False,
                        base=config['rope_theta'],
                        scale=1.0)

        self.mask = None
        # rope_scale = (
        #     1 / args.rope_scaling["factor"]
        #     if args.rope_scaling is not None and args.rope_scaling["type"] == "linear"
        #     else 1
        # )
        # rope_scale = 1
        # self.rotary_emb = Qwen2RotaryEmbedding(
        #     self.head_dim,
        #     max_position_embeddings=self.max_position_embeddings,
        #     base=self.rope_theta,
        # )

    def __call__(
        self,
        x: mx.array,
        mask=None,
        cache=None,
    ) -> mx.array:
        B, L, D = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # ipdb.set_trace()
        if cache is not None:
            raise RuntimeError("not implemented yet~")
        #     queries = self.rope(queries, offset=cache.offset)
        #     keys = self.rope(keys, offset=cache.offset)
        #     keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_dim)
        return self.o_proj(output)


class Qwen2AudioEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config['d_model']
        self.encoder_ffn_dim = config['encoder_ffn_dim']
        self.self_attn = Qwen2AudioAttention(
                                embed_dim=self.embed_dim,
                                num_heads=config['encoder_attention_heads'],
                                config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # self.dropout = 0.0
        # self.activation_dropout = 0.0
        self.activation_fn = nn.gelu
        self.fc1 = nn.Linear(self.embed_dim, self.encoder_ffn_dim)
        self.fc2 = nn.Linear(self.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen2MLP(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.intermediate_size = config['intermediate_size']
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config['hidden_act']]()

    def __call__(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


#  better choice: mlx.nn.RMSNorm
#
# class Qwen2RMSNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-6):
#         """
#         Qwen2RMSNorm is equivalent to T5LayerNorm
#         """
#         super().__init__()
#         # self.weight = nn.Parameter(mx.ones(hidden_size))
#         self.weight = mx.ones(hidden_size)
#         self.variance_epsilon = eps
#
#     def __call__(self, hidden_states):
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(mx.float32)
#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * \
#             mx.rsqrt(variance + self.variance_epsilon)
#         return self.weight * hidden_states.to(input_dtype)
#
#     def extra_repr(self):
#         return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()

        self.self_attn = Qwen2Attention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        # self.input_layernorm = Qwen2RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        self.input_layernorm = nn.RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        # self.post_attention_layernorm = Qwen2RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        self.post_attention_layernorm = nn.RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        # if output_attentions:
        #     outputs += (self_attn_weights,)
        #
        # if use_cache:
        #     outputs += (present_key_value,)

        # ipdb.set_trace()
        return outputs


class Qwen2AudioEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.num_mel_bins = config['num_mel_bins']  # 128
        self.encoder_layers = config['encoder_layers']  # 32
        self.encoder_attention_heads = config['encoder_attention_heads']  # 20
        self.encoder_ffn_dim = config['encoder_ffn_dim']  # 5120
        self.d_model = config['d_model']  # 1280
        self.activation_function = config['activation_function']  # gelu
        self.scale_embedding = config['scale_embedding']  # false
        self.max_source_positions = config['max_source_positions']  # 1500

        self.config = config

        self.dropout = 0.0
        self.layerdrop = 0.0

        # self.padding_idx = config.pad_token_id
        self.embed_scale = 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, self.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, self.d_model)
        # self.embed_positions.requires_grad_(False)

        self.layers = [Qwen2AudioEncoderLayer(config) for _ in range(self.encoder_layers)]
        self.layer_norm = nn.LayerNorm(self.d_model)
        # Ignore copy
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

    def __call__(self, input_features):
        inputs_embeds = nn.gelu(self.conv1(input_features))
        inputs_embeds = nn.gelu(self.conv2(inputs_embeds))

        out_len = inputs_embeds.shape[1]
        hidden_states = inputs_embeds + self.embed_positions.weight[:out_len,:]

        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states)

        # According to https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.AvgPool1d.html
        # Assuming an input of shape (N, L, C) and kernel_size is k, the output is a tensor of shape (N, Lout, C)
        hidden_states = self.avg_pooler(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        # return {'last_hidden_state': hidden_states}
        return hidden_states


class Qwen2AudioMultiModalProjector(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.linear = nn.Linear(config['audio_config']['d_model'], config['text_config']['hidden_size'], bias=True)

    def __call__(self, audio_features):
        return self.linear(audio_features)


class Qwen2Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # self.padding_idx = config['pad_token_id'
        self.vocab_size = config['vocab_size']

        self.embed_tokens = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.layers = [
            Qwen2DecoderLayer(config, layer_idx)
            for layer_idx in range(config['num_hidden_layers'])
        ]
        # self.norm = Qwen2RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        self.norm = nn.RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def __call__(
        self,
        inputs_embeds,
        attention_mask,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):

        hidden_states = inputs_embeds
        next_decoder_cache = None

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask,
                # position_ids=position_ids,
                # past_key_value=past_key_values,
                # output_attentions=output_attentions,
                # use_cache=use_cache,
                # cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]

            # if use_cache:
            #     next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            #
            # if output_attentions:
            #     all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        # if output_hidden_states:
        #     all_hidden_states += (hidden_states,)

        # next_cache = None
        # if use_cache:
        #     next_cache = next_decoder_cache.to_legacy_cache(
        #     ) if use_legacy_cache else next_decoder_cache

        # if not return_dict:
        #     return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        # return BaseModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attns,
        # )
        return hidden_states


class Qwen2ForCausalLM(nn.Module):
    # _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__()
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)

    # def get_input_embeddings(self):
    #     return self.model.embed_tokens
    #
    # def set_input_embeddings(self, value):
    #     self.model.embed_tokens = value
    #
    # def get_output_embeddings(self):
    #     return self.lm_head
    #
    # def set_output_embeddings(self, new_embeddings):
    #     self.lm_head = new_embeddings
    #
    # def set_decoder(self, decoder):
    #     self.model = decoder
    #
    # def get_decoder(self):
    #     return self.model

    def __call__(
        self,
        inputs_embeds,
        attention_mask,
        position_ids=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        # ipdb.set_trace()
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            # past_key_values=past_key_values,
            # use_cache=use_cache,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
            # cache_position=cache_position,
        )
        # return outputs

        hidden_states = outputs[:, -1:, :]  # (batch_size, seq_len, hidden_dim)
        logits = self.lm_head(hidden_states)
        # logits = logits.float()

        # ipdb.set_trace()
        
        return logits

        # loss = None
        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     shift_logits = shift_logits.view(-1, self.config.vocab_size)
        #     shift_labels = shift_labels.view(-1)
        #     # Enable model parallelism
        #     shift_labels = shift_labels.to(shift_logits.device)
        #     loss = loss_fct(shift_logits, shift_labels)
        #
        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output
        #
        # return CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )


class Qwen2AudioForConditionalGeneration(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        preprocessor_config_file = check_file(Path(model_path)/'config.json')
        with open(preprocessor_config_file, encoding="utf-8") as handle:
            config_kwargs = json.load(handle)
        Qwen2Config.update(config_kwargs['text_config'])
        config_kwargs['text_config'] = Qwen2Config
        config_kwargs['audio_token_index'] = AUDIO_TOKEN_INDEX
        # self.vocab_size = config_kwargs['text_config']['vocab_size']
        self.config = config_kwargs

        self.audio_tower = Qwen2AudioEncoder(config_kwargs['audio_config'])
        self.multi_modal_projector = Qwen2AudioMultiModalProjector(config_kwargs)
        self.language_model = Qwen2ForCausalLM(config_kwargs['text_config'])

        safetensors = [
            str(Path(model_path) / f'model-0000{i}-of-00005.safetensors')
            for i in [1, 2, 3, 4, 5]
        ]
        mx_weights = {}
        for tensor in safetensors:
            mx_weights.update(mx.load(str(tensor)))
        # qwen2 default layout of conv1.weight from safetensors is [out_channel, in_channel, kernel_size]
        # but mlx conv1 weight shape is [out_channel, kernel_size, in_channel]
        # so a trnaspose is needed before calling load_weights()
        mx_weights = {k: mx.swapaxes(v, 1, 2)
                    if "conv1.weight" in k or "conv2.weight" in k else v
                    for k, v in mx_weights.items()}

        # self.load_weights(list(mx_weights.items()))
        self.load_weights(list(mx_weights.items()))

    def __call__(self, input_ids, input_features, **kwargs):
        audio_index = np.where(np.array(input_ids) == self.config['audio_token_index'])
        # input_ids's dims is 2, so np.where() returns a tuple of size 2
        if audio_index[1].size != 1:
            raise ValueError((
                "Currently this demo only support 'ONE' audio stub, "
                "but you give {audio_index[0].size} audios"))

        special_index = audio_index[1].item()
        # ipdb.set_trace()
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        audio_features = self.multi_modal_projector(self.audio_tower(input_features))

        final_features = mx.concatenate([
            inputs_embeds[:, 0:special_index, :],
            audio_features,
            inputs_embeds[:, special_index+1:, :],
        ], axis=1)

        # ipdb.set_trace()
        feat_len = final_features.shape[1]
        causal_mask = create_additive_causal_mask(feat_len).astype(mx.bfloat16)
        ipdb.set_trace()
        out = self.language_model(inputs_embeds=final_features, attention_mask=causal_mask)
        return out

