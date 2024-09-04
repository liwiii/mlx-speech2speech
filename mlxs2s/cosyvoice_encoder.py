from typing import Tuple

# import torch
# import torch.utils.checkpoint as ckpt

from cosyvoice.transformer.convolution import ConvolutionModule
from cosyvoice.transformer.encoder_layer import TransformerEncoderLayer
from cosyvoice.transformer.encoder_layer import ConformerEncoderLayer
from cosyvoice.transformer.positionwise_feed_forward import PositionwiseFeedForward
from cosyvoice.utils.class_utils import (
    COSYVOICE_EMB_CLASSES,
    COSYVOICE_SUBSAMPLE_CLASSES,
    COSYVOICE_ATTENTION_CLASSES,
    COSYVOICE_ACTIVATION_CLASSES,
)
from cosyvoice.utils.mask import make_pad_mask
from cosyvoice.utils.mask import add_optional_chunk_mask

import mlx.core as mx
import mlx.nn as nn
import math

from .cosyvoice_attention import RelPositionMultiHeadedAttention
import torch


class EspnetRelPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Construct an PositionalEncoding object."""
        super(EspnetRelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: mx.array, offset=0):
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.position_encoding(size=x.size(1), offset=offset)
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset, size: int):
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - size + 1 : self.pe.size(1) // 2 + size,
        ]
        return pos_emb


class BaseSubsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset, size: int):
        return self.pos_enc.position_encoding(offset, size)


class LinearNoSubsampling(BaseSubsampling):
    def __init__(self,
                 idim: int,
                 odim: int,
                 pos_enc_class: nn.Module):
        super().__init__()
        self.out = nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.LayerNorm(odim, eps=1e-5),
        )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(
        self,
        x: mx.array,
        x_mask: mx.array,
        offset=0
    ):
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


# input_layer='linear',
# pos_enc_layer_type='rel_pos_espnet',
# selfattention_layer_type='rel_selfattn',
class BaseEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,  # 512
        output_size: int,  # 1024
        attention_heads: int,  # 16
        linear_units: int,  # 4096
        num_blocks: int,  # 6
        normalize_before: bool,  # True
        static_chunk_size: int,  # 1
        use_dynamic_chunk: bool,  # False
        use_dynamic_left_chunk: bool,  # False
    ):
        super().__init__()
        self._output_size = output_size
        self.embed = LinearNoSubsampling(
            input_size,
            output_size,
            EspnetRelPositionalEncoding(output_size,),
        )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        NOTE(xcsong):
            We pass the `__call__` method of the modules instead of `forward` to the
            checkpointing API because `__call__` attaches all the hooks of the module.
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)
        xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_layers(self, xs: torch.Tensor, chunk_masks: torch.Tensor,
                       pos_emb: torch.Tensor,
                       mask_pad: torch.Tensor) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    def forward_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        att_mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward just one chunk

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.

        """
        assert xs.size(0) == 1
        # tmp_masks is just for interface compatibility
        tmp_masks = torch.ones(1,
                               xs.size(1),
                               device=xs.device,
                               dtype=torch.bool)
        tmp_masks = tmp_masks.unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        # NOTE(xcsong): Before embed, shape(xs) is (b=1, time, mel-dim)
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)
        # NOTE(xcsong): After  embed, shape(xs) is (b=1, chunk_size, hidden-dim)
        elayers, cache_t1 = att_cache.size(0), att_cache.size(2)
        chunk_size = xs.size(1)
        attention_key_size = cache_t1 + chunk_size
        pos_emb = self.embed.position_encoding(offset=offset - cache_t1,
                                               size=attention_key_size)
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)
        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            # NOTE(xcsong): Before layer.forward
            #   shape(att_cache[i:i + 1]) is (1, head, cache_t1, d_k * 2),
            #   shape(cnn_cache[i])       is (b=1, hidden-dim, cache_t2)
            xs, _, new_att_cache, new_cnn_cache = layer(
                xs,
                att_mask,
                pos_emb,
                att_cache=att_cache[i:i + 1] if elayers > 0 else att_cache,
                cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache)
            # NOTE(xcsong): After layer.forward
            #   shape(new_att_cache) is (1, head, attention_key_size, d_k * 2),
            #   shape(new_cnn_cache) is (b=1, hidden-dim, cache_t2)
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
            r_cnn_cache.append(new_cnn_cache.unsqueeze(0))
        if self.normalize_before:
            xs = self.after_norm(xs)

        # NOTE(xcsong): shape(r_att_cache) is (elayers, head, ?, d_k * 2),
        #   ? may be larger than cache_t1, it depends on required_cache_size
        r_att_cache = torch.cat(r_att_cache, dim=0)
        # NOTE(xcsong): shape(r_cnn_cache) is (e, b=1, hidden-dim, cache_t2)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=0)

        return (xs, r_att_cache, r_cnn_cache)

    def forward_chunk_by_chunk(
        self,
        xs: torch.Tensor,
        decoding_chunk_size: int,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward input chunk by chunk with chunk_size like a streaming
            fashion

        Here we should pay special attention to computation cache in the
        streaming style forward chunk by chunk. Three things should be taken
        into account for computation in the current network:
            1. transformer/conformer encoder layers output cache
            2. convolution in conformer
            3. convolution in subsampling

        However, we don't implement subsampling cache for:
            1. We can control subsampling module to output the right result by
               overlapping input instead of cache left context, even though it
               wastes some computation, but subsampling only takes a very
               small fraction of computation in the whole model.
            2. Typically, there are several covolution layers with subsampling
               in subsampling module, it is tricky and complicated to do cache
               with different convolution layers with different subsampling
               rate.
            3. Currently, nn.Sequential is used to stack all the convolution
               layers in subsampling, we need to rewrite it to make it work
               with cache, which is not preferred.
        Args:
            xs (torch.Tensor): (1, max_len, dim)
            chunk_size (int): decoding chunk size
        """
        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        subsampling = self.embed.subsampling_rate
        context = self.embed.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = xs.size(1)
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = xs[:, cur:end, :]
            (y, att_cache,
             cnn_cache) = self.forward_chunk(chunk_xs, offset,
                                             required_cache_size, att_cache,
                                             cnn_cache)
            outputs.append(y)
            offset += y.size(1)
        ys = torch.cat(outputs, 1)
        masks = torch.ones((1, 1, ys.size(1)),
                           device=ys.device,
                           dtype=torch.bool)
        return ys, masks


class TransformerEncoder(BaseEncoder):
    """Transformer encoder module."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        key_bias: bool = True,
        selfattention_layer_type: str = "selfattn",
        activation_type: str = "relu",
        gradient_checkpointing: bool = False,
    ):
        """ Construct TransformerEncoder

        See Encoder for the meaning of each parameter.
        """
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         static_chunk_size, use_dynamic_chunk, global_cmvn,
                         use_dynamic_left_chunk, gradient_checkpointing)
        activation = COSYVOICE_ACTIVATION_CLASSES[activation_type]()
        self.encoders = torch.nn.ModuleList([
            TransformerEncoderLayer(
                output_size,
                COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](attention_heads,
                                                                      output_size,
                                                                      attention_dropout_rate,
                                                                      key_bias),
                PositionwiseFeedForward(output_size, linear_units,
                                        dropout_rate, activation),
                dropout_rate, normalize_before) for _ in range(num_blocks)
        ])


# input_layer='linear',
# pos_enc_layer_type='rel_pos_espnet',
# selfattention_layer_type='rel_selfattn',
class ConformerEncoder(BaseEncoder):
    def __init__(
        self,
        input_size: int,  # 512
        output_size: int,  # 1024
        attention_heads: int,  # 16
        linear_units: int,  # 4096
        num_blocks: int,  # 6
        normalize_before: bool,  # True
        use_cnn_module: bool,  # False
        macaron_style: bool,  # False
        use_dynamic_chunk: bool,  # False
        use_dynamic_left_chunk: bool,  # False
        static_chunk_size: int,  # 1
        positionwise_conv_kernel_size: int = 1,
        activation_type: str = "swish",
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        key_bias: bool = True,
    ):
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, normalize_before,
                         static_chunk_size, use_dynamic_chunk,
                         use_dynamic_left_chunk)
        activation = nn.silu()

        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            key_bias,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            activation,
        )
        # convolution module definition
        # convolution_layer_args = (
        #     output_size,
        #     cnn_module_kernel,
        #     activation,
        #     cnn_module_norm,
        #     causal)

        self.encoders = [
            ConformerEncoderLayer(
                output_size,
                RelPositionMultiHeadedAttention(*encoder_selfattn_layer_args),
                PositionwiseFeedForward(*positionwise_layer_args),
                None, None,
                normalize_before)
            for _ in range(num_blocks)
        ]
