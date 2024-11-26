# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Scaled dot-product attention."""

import typing
from typing import Literal, TypeAlias

from alphafold3_torch.torch.attention import attention_base as base
from alphafold3_torch.torch.attention import flash_attention as attention_triton
from alphafold3_torch.torch.attention import xla_attention
from alphafold3_torch.torch.common import triton_utils
import torch
from torch import Tensor
from torch.types import _bool, _device, _dtype, _int, _size
from torch.types import _dtype as DType

from typing import Float
from typing import List, Optional, Tuple, Union
import typeguard

Implementation: TypeAlias = Literal["cudnn"]


def dot_product_attention(
    query: Tensor, # "*B T H D"
    key: Tensor, # "*B t #H D"
    value: Tensor, # "*B t #H D"
    *,
    bias: Tensor, # "*#B #H #T #t"
    mask: Optional[Tensor], # "*#B #H #T #t"
    implementation: Implementation= None,
    logits_dtype: Optional[DType] = None,
    precision= None,
) -> Tensor: #"*B T H D"
  """Performs scaled dot-product attention.

  Scaled dot-product attention from "Attention is all you need"
  https://arxiv.org/abs/1706.03762.

  Computes self- or cross-attention. The following is computed:
  softmax(qk_scale * query @ key^T + bias) @ value.

  Supports both multi-head and multi-query attention
  (https://arxiv.org/abs/1911.02150).

  Arguments:
    query: Query array of shape `[batch, seq_len_q, num_heads, head_dim]`.
    key: Key array of shape `[batch, seq_len_kv, num_heads, head_dim]`.
      `num_heads` can be 1 for multi-query attention.
    value: Value array of shape `[batch, seq_len_kv, num_heads, head_dim]`.
      `num_heads` can be 1 for multi-query attention.
    bias: Optional bias array, broadcastable to shape `[batch, num_heads,
      seq_len_q, seq_len_kv]`.
    mask: Optional boolean mask, broadcastable to `[batch, num_heads, seq_len_q,
      seq_len_kv]`. Attention weights are masked out if the corresponding mask
      value is `False`.
    implementation: if `None` (default), an implementation is automatically
      chosen. 'xla' will use standard XLA and work on any platform, 'triton'
      will use a fused Triton GPU kernel, and 'cudnn' a cuDNN FlashAttention
      kernel. Only a subset of data types, shapes and GPUs are supported by
      'triton' and 'cudnn', with an exception thrown in this case.
    logits_dtype: Data type for attention logits (`query @ key^T`). If `None` is
      passed (the default), the accumulator type from the `query @ key^T` dot
      product will be used, which is FP32 for BF16/FP16/FP32 inputs. Note that
      this default increases the memory usage for BF16/FP16 inputs when using
      `implementation='xla'`, but does not increase memory usage when using
      `implementation='triton'`.
    precision: The precision for the dot products. Either `None` (default) which
      uses the default JAX precision for a backend; a tuple `(
      query_key_dot_precision, weights_value_dot_precision)` of
      `jax.lax.Precision` objects; or a single `jax.lax.Precision` object
      applied to both dot products.

  Returns:
    An array with the same shape as `query`.
  """

  if implementation is not None:
    named_args = typing.get_args(Implementation)
    if implementation not in named_args:
      raise ValueError(
          f"Unsupported named implementation. Must be one of {named_args}."
      )

  if implementation == "cudnn":
    if logits_dtype is not None:
      raise ValueError(
          "logits_dtype is not supported for cudnn implementation."
      )
    if precision is not None:
      raise NotImplementedError(
          "precision is not supported for cudnn implementation."
      )

    return torch.nn.functional.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        # bias=bias,
        attn_mask=mask,
        # implementation="cudnn",
    )