"""Tiny block-FP8 qwen3_5_moe checkpoint generator for TP loader tests / E2E boots.

Shrinks the Qwen3.6-35B-A3B-FP8 geometry (4 layers: 3 GDN + 1 full-attn, few experts)
while keeping hidden 2048, all per-layer dims, and -- by default -- the FULL vocab
(248320), so lm_head / embed_tokens exercise the exact vocab-parallel shapes the TP=2
loader bug hit. Weights are random: the point is to load/boot/serve, not to generate
coherent text.

Used by ``test_qwen3_5_tp_quant.py`` (small vocab, CPU loader test) and manually for
full E2E boots (full vocab, ``ft serve --tensor-parallel-size 2``).
"""
from __future__ import annotations

import json
import os

import torch

# Per-layer dims are the REAL Qwen3.6-35B-A3B-FP8 values (block-fp8, 128x128 scales):
H = 2048            # hidden
M = 512             # moe / shared intermediate
Q = 8192            # full-attn q_proj rows (16 heads * 256 * 2, attn output gate)
KV = 512            # k / v rows (2 heads * 256)
ODIM = 4096         # o_proj input (16 heads * 256; the output gate halves q before o_proj)
O = 2048            # o_proj / out_proj rows
QKV = 8192          # GDN in_proj_qkv rows (2 * 16 * 128 + 32 * 128)
Z = 4096            # GDN in_proj_z rows (32 * 128)
CONV = 8192         # GDN conv1d channels
NV = 32             # GDN value heads (A_log / dt_bias / in_proj_ba rows)
KDIM = 128          # GDN norm dim
HEAD_DIM = 256      # full-attn head dim
N_Q_HEADS = 16
N_KV_HEADS = 2


def make_tiny_fp8_ckpt(
    out_dir: str,
    *,
    vocab: int = 248320,
    layers: int = 4,
    experts: int = 32,
    seed: int = 0,
) -> str:
    """Write a minimal block-FP8 checkpoint (config.json + single safetensors + index)."""
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)

    def fp8(shape):
        return (torch.randn(*shape) * 0.05).to(torch.float8_e4m3fn)

    def scale(shape):
        return (torch.rand(*shape) * 0.02 + 0.03).to(torch.bfloat16)

    def bf16(shape, s=0.02):
        return (torch.randn(*shape) * s).to(torch.bfloat16)

    t: dict[str, torch.Tensor] = {}
    t["lm_head.weight"] = bf16((vocab, H))
    t["model.language_model.embed_tokens.weight"] = bf16((vocab, H))
    t["model.language_model.norm.weight"] = bf16((H,))
    for li in range(layers):
        pre = f"model.language_model.layers.{li}"
        t[f"{pre}.input_layernorm.weight"] = bf16((H,))
        t[f"{pre}.post_attention_layernorm.weight"] = bf16((H,))
        t[f"{pre}.mlp.gate.weight"] = bf16((experts, H))
        t[f"{pre}.mlp.shared_expert_gate.weight"] = bf16((1, H))
        for p in ("gate", "up"):
            t[f"{pre}.mlp.shared_expert.{p}_proj.weight"] = fp8((M, H))
            t[f"{pre}.mlp.shared_expert.{p}_proj.weight_scale_inv"] = scale((M // 128, H // 128))
        t[f"{pre}.mlp.shared_expert.down_proj.weight"] = fp8((H, M))
        t[f"{pre}.mlp.shared_expert.down_proj.weight_scale_inv"] = scale((H // 128, M // 128))
        for e in range(experts):
            ep = f"{pre}.mlp.experts.{e}"
            for p in ("gate", "up"):
                t[f"{ep}.{p}_proj.weight"] = fp8((M, H))
                t[f"{ep}.{p}_proj.weight_scale_inv"] = scale((M // 128, H // 128))
            t[f"{ep}.down_proj.weight"] = fp8((H, M))
            t[f"{ep}.down_proj.weight_scale_inv"] = scale((H // 128, M // 128))
        if li % 4 != 3:  # GDN (linear attention)
            g = f"{pre}.linear_attn"
            t[f"{g}.A_log"] = bf16((NV,))
            t[f"{g}.dt_bias"] = bf16((NV,))
            t[f"{g}.conv1d.weight"] = bf16((CONV, 1, 4))
            t[f"{g}.in_proj_a.weight"] = bf16((NV, H))
            t[f"{g}.in_proj_b.weight"] = bf16((NV, H))
            t[f"{g}.in_proj_qkv.weight"] = fp8((QKV, H))
            t[f"{g}.in_proj_qkv.weight_scale_inv"] = scale((QKV // 128, H // 128))
            t[f"{g}.in_proj_z.weight"] = fp8((Z, H))
            t[f"{g}.in_proj_z.weight_scale_inv"] = scale((Z // 128, H // 128))
            t[f"{g}.norm.weight"] = bf16((KDIM,))
            t[f"{g}.out_proj.weight"] = fp8((O, Z))
            t[f"{g}.out_proj.weight_scale_inv"] = scale((O // 128, Z // 128))
        else:  # full attention
            a = f"{pre}.self_attn"
            for p, rows in (("q", Q), ("k", KV), ("v", KV)):
                t[f"{a}.{p}_proj.weight"] = fp8((rows, H))
                t[f"{a}.{p}_proj.weight_scale_inv"] = scale((rows // 128, H // 128))
            t[f"{a}.o_proj.weight"] = fp8((O, ODIM))
            t[f"{a}.o_proj.weight_scale_inv"] = scale((O // 128, ODIM // 128))
            t[f"{a}.q_norm.weight"] = bf16((HEAD_DIM,))
            t[f"{a}.k_norm.weight"] = bf16((HEAD_DIM,))

    cfg = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": H,
            "num_hidden_layers": layers,
            "layer_types": [
                "linear_attention" if i % 4 != 3 else "full_attention" for i in range(layers)
            ],
            "full_attention_interval": 4,
            "num_attention_heads": N_Q_HEADS,
            "num_key_value_heads": N_KV_HEADS,
            "head_dim": HEAD_DIM,
            "attn_output_gate": True,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "rope_type": "default",
            },
            "linear_num_key_heads": 16,
            "linear_num_value_heads": NV,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": experts,
            "num_experts_per_tok": min(8, experts),
            "moe_intermediate_size": M,
            "shared_expert_intermediate_size": M,
            "vocab_size": vocab,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": False,
            "use_cache": True,
        },
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    }
    json.dump(cfg, open(f"{out_dir}/config.json", "w"), indent=1)
    save_file(t, f"{out_dir}/model.safetensors")
    json.dump(
        {
            "metadata": {"total_size": sum(x.numel() * x.element_size() for x in t.values())},
            "weight_map": {k: "model.safetensors" for k in t},
        },
        open(f"{out_dir}/model.safetensors.index.json", "w"),
    )
    return out_dir


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "tiny-fp8-ckpt"
    vocab = int(sys.argv[2]) if len(sys.argv) > 2 else 248320
    make_tiny_fp8_ckpt(out, vocab=vocab)
    print(f"wrote {out} (vocab={vocab})")
