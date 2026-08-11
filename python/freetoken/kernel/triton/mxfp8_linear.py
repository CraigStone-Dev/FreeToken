"""MXFP8 (block-32 e8m0-scaled FP8) W8A16 dense linear, shared across models.

MiniMax-M3's modelopt MIXED_PRECISION checkpoint quantizes every dense projection
(attention q/k/v/o, the leading dense MLPs, the per-layer shared experts, the sparse
indexer projections) to **MXFP8**: ``weight`` fp8-e4m3 ``[N, K]`` plus a per-output-row,
per-32-input-column e8m0 scale ``weight_scale_inv`` (uint8 exponent codes ``[N, K//32]``;
the dequant multiplier is ``2**(code - 127)``, matching vLLM's
``modelopt`` MXFP8 semantics -- ``_mxfp8_e4m3_quantize_torch`` computes
``descale = exp2(code - 127)`` and ``w_bf16 = w_fp8 * descale``).

Keeping the weight MXFP8 and reading it directly in a W8A16 GEMV halves the decode
weight traffic vs a bf16 dequant at load -- decode is weight-bandwidth bound, same
motivation as ``fp8_pertensor_linear``, whose split-K structure the M==1 kernel
mirrors. The block-32 scale is loaded ONCE per (BLOCK_N, BLOCK_K//32) tile and
broadcast in registers -- a per-(n,k)-element scale gather + exp2 costs ~3x the
whole GEMV (PR#110 review: 424-768 GB/s vs ~2 TB/s for the same-structure
per-tensor-fp8 kernel).

Prefill / batched forward (M > 1) does NOT use a custom GEMM: it dequantizes the
weight to bf16 (``fp8 * 2**k`` is exact in bf16) and runs cuBLAS, the same
dequant-then-matmul precedent as the NVFP4 prefill path. The materialized weight is
a per-call transient; a fused inline-dequant GEMM never came within 30x of cuBLAS
on these shapes (PR#110 review), so the transient's extra HBM round-trip is the
fast option by a wide margin.

Numerics: the GEMV dequants fp8->f32 with the pow2 scale in fp32 and accumulates in
fp32; the M>1 path multiplies in bf16 via cuBLAS with fp32 accumulate (identical to
serving every other bf16 projection).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from freetoken.layers.base import BaseOP

from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view, e4m3_native_cx, e4m3_u8_to_f32

FP8 = torch.float8_e4m3fn
MXFP8_BLOCK = 32
_TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}

# Escape hatch: FREETOKEN_DEBUG_MXFP8_REF=1 swaps the kernels for a pure-torch
# dequant matmul (numeric reference / A-B debugging). Evaluated once.
_USE_REF = os.environ.get("FREETOKEN_DEBUG_MXFP8_REF") == "1"


def mxfp8_dequant(weight: torch.Tensor, scale_codes: torch.Tensor,
                  dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Dequant: ``w[n, k] * 2**(codes[n, k//32] - 127)`` -> ``dtype``.

    Serves the M>1 forward (dequant + cuBLAS), the load-time bf16 ablation path
    (``FREETOKEN_M3_*_MXFP8=0``) and the kernels' numeric reference in tests.
    ``weight`` ``[..., N, K]`` fp8-e4m3; ``scale_codes`` ``[..., N, K//32]`` uint8
    e8m0 exponent codes.
    """
    assert weight.shape[-1] % MXFP8_BLOCK == 0
    assert scale_codes.shape[-1] == weight.shape[-1] // MXFP8_BLOCK
    descale = torch.exp2(scale_codes.to(torch.float32) - 127.0)
    w = weight.to(torch.float32).view(*weight.shape[:-1], -1, MXFP8_BLOCK)
    return (w * descale.unsqueeze(-1)).view(weight.shape).to(dtype)


# ======================================================================================
# Decode (M==1) split-K GEMV: fp8 x bf16 reduction in fp32; the e8m0 scale is loaded
# once per (BLOCK_N, BLOCK_K//32) tile and broadcast in registers.
# ======================================================================================
@triton.jit
def _mxfp8_gemv_splitk_kernel(
    a_ptr, w_ptr, s_ptr, part_ptr, N, K, n_kb, kb_per,
    stride_ak, stride_wn, stride_wk, stride_sn, stride_sk, stride_pk, stride_pn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Each (pid_n, pid_k) computes the partial sum over ``kb_per`` BLOCK_K chunks for a
    BLOCK_N slice of outputs. BLOCK_K is a multiple of 32, so each chunk covers
    ``BLOCK_K // 32`` whole scale blocks: one ``[BLOCK_N, KB32]`` code load + exp2 per
    tile, broadcast over the 32-wide inner axis via a 3D register view."""
    KB32: tl.constexpr = BLOCK_K // 32
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    off_kb32 = tl.arange(0, KB32)
    kb_start = pid_k * kb_per
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(kb_per):
        kb = kb_start + i
        if kb < n_kb:
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(a_ptr + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
            if e4m3_native_cx():
                w = tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float32)
            else:
                w = e4m3_u8_to_f32(tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0,
                ))
            codes = tl.load(
                s_ptr + offs_n[:, None] * stride_sn
                + (kb * KB32 + off_kb32[None, :]) * stride_sk,
                mask=n_mask[:, None] & ((kb * KB32 + off_kb32[None, :]) * 32 < K),
                other=127,
            ).to(tl.float32)  # [BLOCK_N, KB32]
            scale = tl.exp2(codes - 127.0)
            prod = tl.reshape(w * a[None, :], (BLOCK_N, KB32, 32))
            acc += tl.sum(tl.sum(prod, axis=2) * scale, axis=1)
    tl.store(part_ptr + pid_k * stride_pk + offs_n * stride_pn, acc, mask=n_mask)


@triton.jit
def _splitk_reduce_kernel(
    part_ptr, out_ptr, N, SPLIT_K: tl.constexpr,
    stride_pk, stride_pn, BLOCK: tl.constexpr, OUT: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + k * stride_pk + offs * stride_pn, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(OUT), mask=mask)


def _gemv(a: torch.Tensor, weight: torch.Tensor, scale_codes: torch.Tensor,
          out_dtype: torch.dtype) -> torch.Tensor:
    """M==1 split-K GEMV. ``a`` [K] bf16; ``weight`` [N, K] fp8; ``scale_codes``
    [N, K//32] uint8."""
    N, K = weight.shape
    BLOCK_K = 128
    n_kb = triton.cdiv(K, BLOCK_K)
    BLOCK_N = 16
    n_tiles = triton.cdiv(N, BLOCK_N)
    split_k = max(1, min(1536 // n_tiles, n_kb))
    split_k = 1 << (split_k.bit_length() - 1)  # pow2 -> stable reduction order
    kb_per = triton.cdiv(n_kb, split_k)
    part = torch.empty((split_k, N), dtype=torch.float32, device=a.device)
    _mxfp8_gemv_splitk_kernel[(n_tiles, split_k)](
        a, weight, scale_codes, part, N, K, n_kb, kb_per,
        a.stride(0), weight.stride(0), weight.stride(1),
        scale_codes.stride(0), scale_codes.stride(1),
        part.stride(0), part.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=1,
    )
    out = torch.empty(N, dtype=out_dtype, device=a.device)
    _splitk_reduce_kernel[(triton.cdiv(N, 256),)](
        part, out, N, split_k, part.stride(0), part.stride(1),
        BLOCK=256, OUT=_TL_DTYPE[out_dtype if out_dtype in _TL_DTYPE else torch.bfloat16],
        num_warps=2,
    )
    return out


def mxfp8_linear(
    x: torch.Tensor, weight: torch.Tensor, scale_codes: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``y = x @ dequant(weight, scale_codes)^T``. Decode (M=1) -> split-K GEMV;
    M>1 -> bf16 dequant + cuBLAS (see the module docstring). ``weight`` [N, K]
    fp8-e4m3; ``scale_codes`` [N, K//32] uint8 e8m0 exponent codes (dequant
    multiplier ``2**(code - 127)``)."""
    *lead, K = x.shape
    N = weight.shape[0]
    assert K % MXFP8_BLOCK == 0 and scale_codes.shape == (N, K // MXFP8_BLOCK)
    if _USE_REF:  # numeric-reference fallback (debug / A-B)
        w = mxfp8_dequant(weight, scale_codes, dtype=torch.float32)
        out = (x.reshape(-1, K).float() @ w.t()).to(x.dtype).reshape(*lead, N)
    elif x.numel() // K == 1:
        w8 = e4m3_kernel_view(weight)
        out = _gemv(x.reshape(K), w8, scale_codes, x.dtype).reshape(*lead, N)
    else:
        # Per-call bf16 transient (pow2 descale is lossless in bf16) + cuBLAS.
        w = mxfp8_dequant(weight, scale_codes, dtype=x.dtype)
        out = torch.nn.functional.linear(x.reshape(-1, K), w).reshape(*lead, N)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


# ======================================================================================
# BaseOP linear layers (TP=1, replicated). Buffers: fp8 ``weight`` + uint8
# ``weight_scale_inv`` (the checkpoint's e8m0 exponent codes, loaded verbatim).
# ======================================================================================
class Mxfp8Linear(BaseOP):
    """Replicated MXFP8 linear: fp8-e4m3 ``weight`` ``[out, in]`` + uint8 e8m0
    ``weight_scale_inv`` ``[out, in // 32]``. Fused projections concatenate several
    checkpoint projections along the output dim; the scales are per-output-row so the
    concatenation is exact."""

    def __init__(self, in_features: int, out_features: int, has_bias: bool = False):
        assert in_features % MXFP8_BLOCK == 0
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty(out_features, in_features, dtype=FP8)
        self.weight_scale_inv = torch.empty(
            out_features, in_features // MXFP8_BLOCK, dtype=torch.uint8
        )
        self.bias = torch.empty(out_features) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mxfp8_linear(x, self.weight, self.weight_scale_inv, self.bias)


__all__ = ["FP8", "MXFP8_BLOCK", "Mxfp8Linear", "mxfp8_linear", "mxfp8_dequant"]
