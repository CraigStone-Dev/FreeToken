"""Quant-aware dense-linear factories for Muse Glimmer.

The checkpoint is either plain bf16 or compressed-tensors NVFP4 on every text Linear
(``attn_quant``/``dense_quant`` are set together by parse_config), so the dispatch is a
two-way choice between the shared W4A16 kernels and the TP-aware bf16 layers.
"""

from __future__ import annotations


def make_col_merged(config, in_f: int, output_sizes: list[int], has_bias: bool = False):
    if getattr(config, "attn_quant", "none") == "nvfp4":
        from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseColMerged

        return Nvfp4DenseColMerged(in_f, output_sizes, has_bias)
    from freetoken.layers import LinearColParallelMerged

    return LinearColParallelMerged(in_f, output_sizes, has_bias=has_bias)


def make_replicated(config, in_f: int, out_f: int, has_bias: bool = False):
    if getattr(config, "attn_quant", "none") == "nvfp4":
        from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseLinear

        return Nvfp4DenseLinear(in_f, out_f, has_bias)
    from freetoken.layers import LinearReplicated

    return LinearReplicated(in_f, out_f, has_bias=has_bias)


__all__ = ["make_col_merged", "make_replicated"]
