import ipdb
import json
# import math
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlxs2s.utils import check_file

ACT2FN = {
    'silu': nn.SiLU,
    'gelu': nn.GELU,
}

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


class Qwen2AudioAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

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
        self.dropout = 0.0
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

    # Copied from transformers.models.bart.modeling_bart.BartAttention._shape with BART->whisper
    # def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
    #     return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions: bool = False,
        cache_position=None,
    ):
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self._shape(self.q_proj(
            hidden_states) * self.scaling, tgt_len, bsz)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self._shape(self.k_proj(current_states), -1, bsz)
            value_states = self._shape(self.v_proj(current_states), -1, bsz)
            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {
                        "cache_position": cache_position}
                )

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError((
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                ))
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_probs, value_states)

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError((
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            ))

        attn_output = attn_output.transpose(1, 2)
        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class Attention(nn.Module):
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()

        self.hidden_dim = config['hidden_size']
        self.n_heads = config['num_attention_heads']
        self.n_kv_heads = config['num_key_value_heads']

        self.layer_idx = layer_idx

        self.head_dim = self.hidden_dim // self.n_kv_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_dim, self.n_kv_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(
            self.hidden_dim, self.n_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(
            self.hidden_dim, self.n_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(
            self.n_kv_heads * self.head_dim, self.hidden_dim, bias=False)

        # rope_scale = (
        #     1 / args.rope_scaling["factor"]
        #     if args.rope_scaling is not None and args.rope_scaling["type"] == "linear"
        #     else 1
        # )
        rope_scale = 1
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=config['rope_theta'],
            scale=rope_scale,
        )

    def __call__(
        self,
        x: mx.array,
        mask=None,
        cache=None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(
            B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
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
        self.dropout = 0.0
        self.activation_dropout = 0.0
        self.activation_fn = nn.gelu
        self.fc1 = nn.Linear(self.embed_dim, self.encoder_ffn_dim)
        self.fc2 = nn.Linear(self.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # if hidden_states.dtype == torch.float16 and (
        #     torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        # ):
        #     clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        #     hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


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
        self.act_fn = ACT2FN[config['hidden_act']]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        # self.weight = nn.Parameter(mx.ones(hidden_size))
        self.weight = mx.ones(hidden_size)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(mx.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * \
            mx.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        self.hidden_size = config['hidden_size']

        self.self_attn = Attention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(
            config['hidden_size'], eps=config['rms_norm_eps'])
        self.post_attention_layernorm = Qwen2RMSNorm(
            config['hidden_size'], eps=config['rms_norm_eps'])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

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

        self.dropout = 0.0
        self.layerdrop = 0.0

        # self.padding_idx = config.pad_token_id
        self.embed_scale = 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, self.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(
            self.max_source_positions, self.d_model)
        # self.embed_positions.requires_grad_(False)

        self.layers = [Qwen2AudioEncoderLayer(
            config) for _ in range(self.encoder_layers)]
        self.layer_norm = nn.LayerNorm(self.d_model)
        # Ignore copy
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

    def __call__(self, mel, tokens):
        return self.decoder(tokens, self.encoder(mel))[0]


class Qwen2AudioMultiModalProjector(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.linear = nn.Linear(
            config['audio_config']['d_model'], config['text_config']['hidden_size'], bias=True)

    def forward(self, audio_features):
        hidden_states = self.linear(audio_features)
        return hidden_states


class Qwen2Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # self.padding_idx = config['pad_token_id'
        self.vocab_size = config['vocab_size']

        self.embed_tokens = nn.Embedding(
            config['vocab_size'], config['hidden_size'])
        self.layers = [
            Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config['num_hidden_layers'])
        ]
        self.norm = Qwen2RMSNorm(
            config['hidden_size'], eps=config['rms_norm_eps'])

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        use_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache) and not self.training:
            use_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length(
            ) if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache(
            ) if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2ForCausalLM(nn.Module):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__()
        self.model = Qwen2Model(config)
        self.vocab_size = config['vocab_size']
        self.lm_head = nn.Linear(
            config['hidden_size'], config['vocab_size'], bias=False)

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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else True

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen2AudioForConditionalGeneration(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        preprocessor_config_file = check_file(Path(model_path)/'config.json')
        with open(preprocessor_config_file, encoding="utf-8") as handle:
            config_kwargs = json.load(handle)
        Qwen2Config.update(config_kwargs['text_config'])
        config_kwargs['text_config'] = Qwen2Config

        self.audio_tower = Qwen2AudioEncoder(config_kwargs['audio_config'])
        self.multi_modal_projector = Qwen2AudioMultiModalProjector(
            config_kwargs)
        self.vocab_size = config_kwargs['text_config']['vocab_size']
        self.language_model = Qwen2ForCausalLM(config_kwargs['text_config'])

        safetensors = [
            str(Path(model_path) / f'model-0000{i}-of-00005.safetensors')
            for i in [1, 2, 3, 4, 5]
        ]
        mx_weights = {}
        for tensor in safetensors:
            mx_weights.update(mx.load(str(tensor)))
        # qwen2 default conv1 weight shape from safetensors is [out_channel, in_channel, kernel_size]
        # but mlx conv1 weight shape is [out_channel, kernel_size, in_channel]
        # so a trnaspose is needed before calling load_weights()
        mx_weights = {k: mx.swapaxes(v, 1, 2)
                    if "conv1.weight" in k or "conv2.weight" in k else v
                    for k, v in mx_weights.items()}

        # self.load_weights(list(mx_weights.items()))
        self.load_weights(list(mx_weights.items()))

        # print(mx_weights)
        # self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        # # set it to left by default, user can use setter to change padding_sides
       # self._padding_side = "left"
