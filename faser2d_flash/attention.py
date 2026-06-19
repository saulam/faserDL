from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
except ImportError:
    flash_attn_varlen_qkvpacked_func = None


class PackedSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool,
        attention_dropout: float,
        projection_dropout: float,
        backend: str,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.backend = backend
        self.attention_dropout = float(attention_dropout)
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.projection = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(projection_dropout)

    @property
    def active_backend(self) -> str:
        return "flash_attn_varlen" if self.backend == "flash" else "torch_sdpa_explicit"

    def _flash(
        self,
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        if flash_attn_varlen_qkvpacked_func is None:
            raise RuntimeError(
                "FlashAttention is not importable. Run `python -m faser2d_flash.runtime "
                "--require-flash` inside the platon-flashattn environment."
            )
        if not qkv.is_cuda:
            raise RuntimeError(
                "The configured attention backend is FlashAttention, but tokens are on CPU. "
                "Use a CUDA GPU or explicitly set model.attention_backend=torch_sdpa for a "
                "small diagnostic run."
            )
        if qkv.dtype not in {torch.float16, torch.bfloat16}:
            raise RuntimeError(
                f"FlashAttention requires fp16 or bf16 activations, got {qkv.dtype}. "
                "Enable mixed precision in the training configuration."
            )
        if cu_seqlens.dtype != torch.int32:
            cu_seqlens = cu_seqlens.to(torch.int32)
        return flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen,
            dropout_p=self.attention_dropout if self.training else 0.0,
            causal=False,
        )

    def _torch_sdpa(
        self,
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        outputs = []
        dropout = self.attention_dropout if self.training else 0.0
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            sequence = qkv[start:end]
            q, k, v = sequence.unbind(dim=1)
            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=dropout,
                is_causal=False,
                scale=1.0 / math.sqrt(self.head_dim),
            )
            outputs.append(output.squeeze(0).transpose(0, 1))
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        qkv = self.qkv(x).view(x.shape[0], 3, self.num_heads, self.head_dim)
        if self.backend == "flash":
            output = self._flash(qkv, cu_seqlens, max_seqlen)
        else:
            output = self._torch_sdpa(qkv, cu_seqlens)
        output = output.reshape(x.shape[0], -1)
        return self.projection_dropout(self.projection(output))
