"""Muse-Glimmer-30B config parsing and serving-mode resolution.

Drives ``muse_glimmer.parse_config`` off a synthetic HF config shaped exactly like
meta-models/Muse-Glimmer-30B's (multimodal wrapper: text tower in ``text_config``,
weights under ``model.language_model.``) and checks every decision the plan pinned:
the [SWA x3, full] layer split with NoPE full layers, the folded qk scale, the two
norm eps values, the logit post-processing scalars, the compressed-tensors NVFP4
quant modes, and the pool/backend resolution.
"""

from __future__ import annotations

import pytest

from freetoken.attention.base import AttnType
from freetoken.models.muse_glimmer.config import parse_config


class _Cfg:
    """Attribute-access shim over a dict (what AutoConfig hands parse_config)."""

    def __init__(self, data: dict):
        for k, v in data.items():
            setattr(self, k, _Cfg(v) if isinstance(v, dict) and k == "text_config" else v)


def _hf_config(num_layers: int = 52, quantized: bool = False) -> _Cfg:
    pattern = ["sliding_attention"] * 3 + ["full_attention"]
    thetas = [500000.0] * 3 + [0.0]
    reps = (num_layers + 3) // 4
    data = {
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "model_type": "muse_glimmer",
        "image_token_id": 200092,
        "text_config": {
            "hidden_size": 6656,
            "intermediate_size": 19968,
            "num_hidden_layers": num_layers,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "vocab_size": 202048,
            "max_position_embeddings": 131072,
            "hidden_activation": "silu",
            "rms_norm_eps": 1e-5,
            "post_norm_eps": 1e-8,
            "qk_scale_factor": 3.87,
            "final_logit_softcapping": 20.0,
            "output_multiplier": 0.19611613513818404,
            "sliding_window": 2048,
            "tie_word_embeddings": False,
            "layer_types": (pattern * reps)[:num_layers],
            "layer_rope_theta": (thetas * reps)[:num_layers],
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
            "model_type": "muse_glimmer_text",
        },
    }
    if quantized:
        data["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "format": "nvfp4-pack-quantized",
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                        "strategy": "tensor_group",
                    },
                }
            },
        }
    return _Cfg(data)


def test_parse_config_full_model():
    cfg = parse_config(_hf_config())

    assert cfg.num_layers == 52
    assert (cfg.num_qo_heads, cfg.num_kv_heads, cfg.head_dim) == (32, 2, 128)
    assert cfg.hidden_size == 6656 and cfg.intermediate_size == 19968
    assert cfg.vocab_size == 202048 and not cfg.tie_word_embeddings
    assert cfg.hidden_act == "silu" and not cfg.is_moe
    assert cfg.rms_norm_eps == 1e-5 and cfg.post_norm_eps == 1e-8
    assert cfg.use_qk_norm
    # q = qk_norm(q) * 3.87 with standard 1/sqrt(d) attention: folded into sm_scale.
    assert cfg.attn_sm_scale == pytest.approx(3.87 * 128**-0.5)
    assert cfg.final_logit_softcapping == 20.0
    assert cfg.output_multiplier == pytest.approx(0.19611613513818404)
    assert cfg.embedding_scale is None  # NormedEmbedding, not Gemma's sqrt(hidden) scale
    assert cfg.vision_config is None  # served text-only

    groups = {g.name: g for g in cfg.attention_groups}
    assert groups["swa"].layer_ids == tuple(i for i in range(52) if (i + 1) % 4 != 0)
    assert groups["full"].layer_ids == tuple(i for i in range(52) if (i + 1) % 4 == 0)
    assert groups["swa"].sliding_window == 2048
    assert groups["swa"].rotary_config.base == 500000.0
    # NoPE full layers: base 0.0 is the skip-rope marker read by the attention module.
    assert groups["full"].rotary_config.base == 0.0
    assert cfg.rotary_config.base == 500000.0  # top-level config carries the real rope

    specs = {s.name: s for s in cfg.kv_cache_group_specs()}
    assert specs["swa"].attn_type == AttnType.SWA and specs["swa"].sliding_window == 2048
    assert specs["full"].attn_type == AttnType.FULL and specs["full"].sliding_window is None
    assert all((s.num_kv_heads, s.head_dim) == (2, 128) for s in specs.values())

    # BF16 checkpoint: nothing quantized.
    assert cfg.attn_quant == "none" and cfg.dense_quant == "none"
    assert cfg.lm_head_quant == "none"


def test_parse_config_nvfp4_checkpoint():
    cfg = parse_config(_hf_config(quantized=True))
    # compressed-tensors NVFP4 quantizes every text Linear (attention incl. the gate,
    # and the MLP); lm_head / embeddings / norms stay bf16 (the ignore list).
    assert cfg.attn_quant == "nvfp4" and cfg.dense_quant == "nvfp4"
    assert cfg.lm_head_quant == "none"


def test_per_layer_theta_beats_shared_rope_theta():
    # layer_rope_theta is the source of truth: a hypothetical checkpoint giving the
    # full layers a real theta must not be forced to NoPE.
    hf = _hf_config(num_layers=8)
    hf.text_config.layer_rope_theta = [1e6] * 8
    cfg = parse_config(hf)
    groups = {g.name: g for g in cfg.attention_groups}
    assert groups["full"].rotary_config.base == 1e6
    assert groups["swa"].rotary_config.base == 1e6


def test_pool_family_and_backend_resolution():
    from freetoken.engine.engine import _required_attn_types, _resolve_auto_attention_backend
    from freetoken.kvcache import resolve_pool_class
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

    cfg = parse_config(_hf_config())
    assert resolve_pool_class(cfg) is HybridSWAKVCache
    required = _required_attn_types(cfg)
    assert required == frozenset({AttnType.FULL, AttnType.SWA})
    # SWA restricts serving to the triton backend (the only one in the capability
    # matrix that consumes per-call sliding windows), same as gemma4.
    assert _resolve_auto_attention_backend(required, False) == "triton"


def test_registry_resolves_architecture():
    from freetoken.models.register import get_model_spec

    spec = get_model_spec("MuseGlimmerForConditionalGeneration")
    assert spec.module == "freetoken.models.muse_glimmer"
    assert spec.model_cls == "MuseGlimmerForCausalLM"


def test_aot_table_covers_the_checkpoints():
    from freetoken.kernel.aot_models import SUPPORTED_MODELS

    entry = next(m for m in SUPPORTED_MODELS if m.architecture == "MuseGlimmerForConditionalGeneration")
    assert entry.hidden_size == 6656
    assert entry.kv_groups == ((2, 128),)
    assert entry.expert_formats == ()  # dense
    assert "RedHatAI/Muse-Glimmer-30B-NVFP4" in entry.aliases


def test_weight_rename_and_fusion():
    import torch

    from freetoken.models.muse_glimmer.weight import _rename, _try_fuse

    # Text tower renamed, vision dropped, lm_head untouched.
    assert _rename("model.language_model.layers.0.self_attn.q_proj.weight") == (
        "model.layers.0.self_attn.q_proj.weight"
    )
    assert _rename("model.language_model.embed_tokens.weight") == "model.embed_tokens.weight"
    assert _rename("lm_head.weight") == "lm_head.weight"
    assert _rename("model.vision_tower.layers.0.attn.q_proj.weight") is None
    assert _rename("model.vision_adapter.fc1.weight") is None
    assert _rename("model.vision_projection.weight") is None

    # q/k/v + the attention gate fuse into qkvg_proj in declaration order.
    buf: dict = {}
    parts = {
        "q_proj": torch.full((4, 2), 0.0),
        "k_proj": torch.full((2, 2), 1.0),
        "v_proj": torch.full((2, 2), 2.0),
        "gate_proj": torch.full((4, 2), 3.0),
    }
    fused = None
    for name, tensor in parts.items():
        out = _try_fuse(f"model.layers.0.self_attn.{name}.weight", tensor, buf)
        assert out is not None
        if out != ():
            fused = out
    assert fused is not None and not buf
    key, tensor = fused
    assert key == "model.layers.0.self_attn.qkvg_proj.weight"
    assert tensor.shape == (12, 2)
    assert tensor[0, 0] == 0.0 and tensor[4, 0] == 1.0 and tensor[6, 0] == 2.0
    assert tensor[8, 0] == 3.0

    # The MLP's own gate_proj is a different fusion (gate|up), not the attention one.
    buf2: dict = {}
    assert _try_fuse("model.layers.0.mlp.gate_proj.weight", torch.zeros(3, 2), buf2) == ()
    out = _try_fuse("model.layers.0.mlp.up_proj.weight", torch.ones(3, 2), buf2)
    key2, tensor2 = out
    assert key2 == "model.layers.0.mlp.gate_up_proj.weight" and tensor2.shape == (6, 2)


def test_model_state_dict_matches_loader_keys():
    import torch

    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    from freetoken.models.muse_glimmer.model import MuseGlimmerForCausalLM

    cfg = parse_config(_hf_config(num_layers=4))
    model = MuseGlimmerForCausalLM(cfg)
    keys = set(model.state_dict().keys())
    layer0 = {k for k in keys if k.startswith("model.layers.0.")}
    assert layer0 == {
        "model.layers.0.self_attn.qkvg_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.pre_feedforward_layernorm.weight",
        "model.layers.0.post_feedforward_layernorm.weight",
    }
    # Weightless norms (embed_norm, qk_norm) must not demand checkpoint tensors.
    assert "model.embed_norm.weight" not in keys
    assert not any("qk_norm" in k for k in keys)
    assert {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"} <= keys

    # Rope built only where a real theta exists: sliding layers yes, NoPE full no.
    layers = model.model.layers.op_list
    assert layers[0].self_attn.rotary is not None and layers[0].self_attn.is_swa
    assert layers[3].self_attn.rotary is None and not layers[3].self_attn.is_swa
    assert layers[0].self_attn.attn_spec.sliding_window == 2048
    assert layers[3].self_attn.attn_spec.sliding_window is None
    assert layers[0].self_attn.attn_spec.sm_scale == pytest.approx(3.87 * 128**-0.5)

    # NVFP4 build swaps every text Linear for the W4A16 kernels.
    from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseColMerged, Nvfp4DenseLinear

    qcfg = parse_config(_hf_config(num_layers=4, quantized=True))
    qmodel = MuseGlimmerForCausalLM(qcfg)
    attn = qmodel.model.layers.op_list[0].self_attn
    mlp = qmodel.model.layers.op_list[0].mlp
    assert isinstance(attn.qkvg_proj, Nvfp4DenseColMerged)
    assert isinstance(attn.o_proj, Nvfp4DenseLinear)
    assert isinstance(mlp.gate_up_proj, Nvfp4DenseColMerged)
    assert isinstance(mlp.down_proj, Nvfp4DenseLinear)
    assert type(qmodel.lm_head).__name__ == "ParallelLMHead"  # lm_head stays bf16
    del model, qmodel, torch
