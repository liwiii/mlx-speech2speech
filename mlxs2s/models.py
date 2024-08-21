import json
import math
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlxs2s.utils import check_file


class Qwen2AudioEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config['d_model']
        self.encoder_ffn_dim = config['encoder_ffn_dim']
        self.self_attn = mx.fast.scaled_dot_product_attention
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
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
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

        self.embed_positions = nn.Embedding(self.max_source_positions, self.d_model)
        # self.embed_positions.requires_grad_(False)

        self.layers = [Qwen2AudioEncoderLayer(config) for _ in range(self.encoder_layers)]
        self.layer_norm = nn.LayerNorm(self.d_model)
        # Ignore copy
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

    def __call__(self, mel, tokens):
        return self.decoder(tokens, self.encoder(mel))[0]


class Qwen2AudioForConditionalGeneration(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        preprocessor_config_file = check_file(Path(model_path)/'config.json')
        with open(preprocessor_config_file, encoding="utf-8") as handle:
            config_kwargs = json.load(handle)

        self.audio_encoder = Qwen2AudioEncoder(config_kwargs['audio_config'])
