import ipdb
# import time
import json
from pathlib import Path
from typing import Generator, Tuple
from functools import partial
import mlx.core as mx
import mlx.nn as nn
from .utils import (
        check_file,
        KVCache,
        create_attention_mask
)
from .sample_utils import top_p_sampling
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

    def __call__(
        self,
        x: mx.array,
        mask,
        kv_cache,
    ) -> mx.array:
        B, L, D = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if kv_cache is not None:
            queries = self.rope(queries, offset=kv_cache.offset)
            keys = self.rope(keys, offset=kv_cache.offset)
            keys, values = kv_cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # output = mx.fast.scaled_dot_product_attention(
        #     queries, keys, values, scale=self.scale, mask=mask
        # )
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values,
            scale=self.scale, mask=mask
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
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config['hidden_act']]()

    def __call__(self, hidden_state):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        )


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()

        self.self_attn = Qwen2Attention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = nn.RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])
        self.post_attention_layernorm = nn.RMSNorm(config['hidden_size'], eps=config['rms_norm_eps'])

    def __call__(
        self,
        hidden_states,
        attention_mask,
        kv_cache,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask,
            kv_cache
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


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

        # self.dropout = 0.0
        # self.layerdrop = 0.0

        # self.padding_idx = config.pad_token_id
        # self.embed_scale = 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, self.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(self.max_source_positions, self.d_model)
        self.layers = [Qwen2AudioEncoderLayer(config) for _ in range(self.encoder_layers)]
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

    def __call__(self, input_features):
        inputs_embeds = nn.gelu(self.conv1(input_features))
        inputs_embeds = nn.gelu(self.conv2(inputs_embeds))

        out_len = inputs_embeds.shape[1]
        hidden_states = inputs_embeds + self.embed_positions.weight[:out_len, :]

        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states)

        # According to https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.AvgPool1d.html
        # Assuming an input of shape (N, L, C) and kernel_size is k, the output is a tensor of shape (N, Lout, C)
        # hidden_states = self.avg_pooler(hidden_states)
        # hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.layer_norm(self.avg_pooler(hidden_states))

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
        self.n_kv_heads = config['num_key_value_heads']
        self.hidden_dim = config['hidden_size']
        self.head_dim = self.hidden_dim // self.n_kv_heads
        self.n_layers = config['num_hidden_layers']

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
        inputs: mx.array,
        kv_cache,
    ):
        hidden_states = inputs
        mask = create_attention_mask(hidden_states, kv_cache)

        if kv_cache is None:
            kv_cache = [None] * len(self.n_layers)

        for layer, c in zip(self.layers, kv_cache):
            hidden_states = layer(hidden_states, mask, c)

        return self.norm(hidden_states)


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: dict, gen_config: dict):
        super().__init__()
        hidden_size = config['hidden_size']
        vocab_size = config['vocab_size']
        self.gen_config = gen_config
        self.repetition_penalty = self.gen_config['repetition_penalty']
        self.temperature = self.gen_config['temperature']
        self.topK = self.gen_config['top_k']
        self.topP = self.gen_config['top_p']

        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        kv_cache 
    ):
        outputs = self.model(inputs, kv_cache)

        # hidden_states = outputs[:, -1, :]  # (batch_size, seq_len, hidden_dim) -> (batch_size, hidden_dim)
        return self.lm_head(outputs[:, -1, :])

        '''
        # apply penalty rectification
        score = mx.take(logits, input_ids)
        score = mx.where(score < 0, score * self.repetition_penalty, score / self.repetition_penalty)
        logits[0, input_ids[0]] = score  # TODO: alternative approach??

        # apply temperature
        logits = logits * (1.0 / self.temperature)

        # apply topK
        inds = mx.stop_gradient(mx.argpartition(-logits, kth=self.topK-1, axis=-1)[..., :self.topK])

        # apply topP
        inds = inds[:, ::-1]  # topK index with scores from low to high
        # ipdb.set_trace()
        topp_inds = (mx.cumsum(mx.softmax(mx.take_along_axis(logits, inds, axis=-1), axis=-1), axis=-1) > self.topP)

        # ipdb.set_trace()
        # return inds[0, -1]  # OK
        max_len = 1
        o_token = inds[0, -1].item()

        o_tokens = []
        o_tokens.append(o_token)

        B, L, H = inputs_embeds.shape
        cache_len = L
        # self.inputs_embeds = mx.zeros([B, L+128, H], dtype=mx.bfloat16)
        # self.inputs_embeds[:, L, :] = self.language_model.model.embed_tokens(inds[:, -1:])
        while (max_len < 128) and (o_token not in self.gen_config['eos_token_id']):
            print("enter")
            decode_input_embed = self.model.embed_tokens(inds[:, -1:])
            print(decode_input_embed.shape)
            hidden_states, kv_caches = self.model(
                inputs_embeds=decode_input_embed,
                attention_mask=None,
                kv_caches=kv_caches,
                cache_len=cache_len
            )
            cache_len += 1
            # hidden_states = outputs[:, -1, :]  # (batch_size, seq_len, hidden_dim) -> (batch_size, hidden_dim)
            logits = self.lm_head(hidden_states[:, -1, :])
            # apply penalty rectification
            score = mx.take(logits, input_ids)
            score = mx.where(score < 0, score * self.repetition_penalty, score / self.repetition_penalty)
            logits[0, input_ids[0]] = score  # TODO: alternative approach??
            # apply temperature
            logits = logits * (1.0 / self.temperature)
            # apply topK
            inds = mx.stop_gradient(mx.argpartition(-logits, kth=self.topK-1, axis=-1)[..., :self.topK])
            # apply topP
            inds = inds[:, ::-1]  # topK index with scores from low to high
            o_token = inds[0, -1].item()
            o_tokens.append(o_token)
            print(o_tokens)

        return o_tokens
        # inds = inds[:, -mx.sum(topp_inds, axis=-1):][:, ::-1]
        # final_logits = mx.take_along_axis(logits, inds, axis=-1)
        #
        # return final_logits
        '''

class Qwen2AudioForConditionalGeneration(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        config_file = check_file(Path(model_path)/'config.json')
        with open(config_file, encoding="utf-8") as handle:
            config_kwargs = json.load(handle)
        Qwen2Config.update(config_kwargs['text_config'])
        config_kwargs['text_config'] = Qwen2Config
        # config_kwargs['audio_token_index'] = AUDIO_TOKEN_INDEX
        # self.vocab_size = config_kwargs['text_config']['vocab_size']
        self.config = config_kwargs

        generation_config_file = check_file(Path(model_path)/'generation_config.json')
        with open(generation_config_file, encoding="utf-8") as handle:
            self.gen_config = json.load(handle)

        self.n_kv_heads = config_kwargs['text_config']['num_key_value_heads']
        self.hidden_dim = config_kwargs['text_config']['hidden_size']
        self.layers = config_kwargs['text_config']['num_hidden_layers']
        self.head_dim = self.hidden_dim // self.n_kv_heads

        self.repetition_penalty = self.gen_config['repetition_penalty']
        self.temperature = self.gen_config['temperature']
        self.topK = self.gen_config['top_k']
        self.topP = self.gen_config['top_p']
        self.eos_ids = self.gen_config['eos_token_id']

        self.audio_tower = Qwen2AudioEncoder(config_kwargs['audio_config'])
        self.multi_modal_projector = Qwen2AudioMultiModalProjector(config_kwargs)
        self.language_model = Qwen2ForCausalLM(config_kwargs['text_config'], self.gen_config)

        safetensors = [
            str(Path(model_path) / f'model-0000{i}-of-00005.safetensors')
            for i in [1, 2, 3, 4, 5]
        ]
        mx_weights = {}
        for tensor in safetensors:
            mx_weights.update(mx.load(str(tensor)))
        # qwen2's default conv1.weight layout from safetensors is [out_channel, in_channel, kernel_size]
        # but mlx conv1 weight shape is [out_channel, kernel_size, in_channel]
        # so a trnaspose is needed before calling load_weights()
        mx_weights = {k: mx.swapaxes(v, 1, 2)
                    if "conv1.weight" in k or "conv2.weight" in k else v
                    for k, v in mx_weights.items()}

        self.load_weights(list(mx_weights.items()))

    # @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
    def sample(self, input_ids, logits):
        logprobs = logits - mx.logsumexp(logits)

        score = mx.take(logprobs, input_ids)
        score = mx.where(score < 0, score * self.repetition_penalty, score / self.repetition_penalty)
        logprobs[0, input_ids[0]] = score  # TODO: alternative approach??

        # apply temperature
        logprobs = logprobs * (1.0 / self.temperature)
        probs = mx.softmax(logprobs, axis=-1)
        # apply topK
        sorted_indices = mx.argsort(probs, axis=-1).squeeze(0)[-self.topK:]
        sorted_probs = probs[..., sorted_indices]
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
        # apply topP
        top_probs = mx.where(
            cumulative_probs > 1 - self.topP,
            sorted_probs,
            0,
        )
        sorted_token = mx.random.categorical(mx.log(top_probs))
        token = sorted_indices[sorted_token]

        return token[None], probs[0, token]

    def context(self, input_ids, input_features, kv_cache):
        audio_index = np.where(np.array(input_ids) == AUDIO_TOKEN_INDEX)
        # input_ids's dims is 2, so np.where() returns a tuple of size 2
        if audio_index[1].size != 1:
            raise ValueError((
                "Currently this demo only support 'ONE' audio stub, "
                f"but you give {audio_index[1].size} audios"))

        special_index = audio_index[1].item()
        # ipdb.set_trace()
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        audio_features = self.multi_modal_projector(self.audio_tower(input_features))

        final_features = mx.concatenate([
            inputs_embeds[:, 0:special_index, :],
            audio_features,
            inputs_embeds[:, special_index+1:, :],
        ], axis=1)

        out = self.language_model(final_features, kv_cache)
        return out

    def forward(self, dec_id, kv_cache):
        inputs_embeds = self.language_model.model.embed_tokens(dec_id)
        return self.language_model(inputs_embeds, kv_cache)

    def generate_step(
        self,
        input_ids: mx.array,
        input_features: mx.array
    ):
        y = input_ids
        kv_cache = [
            KVCache(self.head_dim, self.n_kv_heads)
            for _ in range(self.layers)
        ]

        repetition_context = input_ids[0, :].tolist()  #TODO: multi-batch support

        logits = self.context(input_ids, input_features, kv_cache)
        y, logprobs = self.sample(input_ids, logits)
        mx.async_eval(y)
        # ipdb.set_trace()
        repetition_context.append(y.item())
        while True:
            logits = self.forward(y, kv_cache)
            next_y, next_logprobs = self.sample(input_ids, logits)
            mx.async_eval(next_y)
            yield y.item(), logprobs
            y, logprobs = next_y, next_logprobs
            repetition_context.append(y.item())

    def __call__(self, input_ids, input_features, max_decoder_tokens):
        output_tokens = []
        for (token, logprobs), n in zip(
            self.generate_step(input_ids, input_features),
            range(max_decoder_tokens),
        ):
            # print(f"DEBUG: {token}")
            output_tokens.append(token)
            if token in self.eos_ids:
                break

        return output_tokens
