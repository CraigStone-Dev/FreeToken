"""CPU unit tests for the TP>1 sharding of qwen4_exp (Qwen3.8-Flash-Next).

Covers the weight-loader shard math (fused qkv / in_proj / gate_up, row-parallel
o_proj / out_proj / down_proj, vocab-parallel embed / lm_head, GDN A_log / dt_bias /
conv1d, replicated PLE / indexer / HC) and, end to end, that ``iter_weights`` at TP=2
emits exactly the shapes of the model's TP=2 state dict.

Run:  pytest tests/models/test_qwen4_exp_tp.py
"""

import json
import os

import torch

import freetoken.distributed.info as _tpinfo
from freetoken.distributed.info import DistributedInfo


def _set_tp(rank: int, size: int):
    _tpinfo._TP_INFO = DistributedInfo(rank, size)


# ======================================================================================
# Tiny config / checkpoint
# ======================================================================================

def _tiny_hf_config():
    from types import SimpleNamespace

    return SimpleNamespace(
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=SimpleNamespace(
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            num_hidden_layers=4,
            layer_types=[
                "linear_attention", "full_attention",
                "linear_attention", "full_attention",
            ],
            vocab_size=128,
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            intermediate_size=0,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
            max_position_embeddings=128,
            rope_theta=10000.0,
            partial_rotary_factor=0.5,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
            output_gate_type="sigmoid",
            hc_count=2,
            hc_lowrank=16,
            # 1-indexed: layer 0 (a linear_attention layer) carries the PLE.
            ple_layer_ids=[1],
            ple_embed_dim=64,
            ple_conv_kernel_size=4,
            ngram_size=3,
            heads_per_ngram=2,
            ngram_vocab_size_base=64,
            make_ngram_vocab_size_divisible_by=8,
            split_ngram_parts=2,
            eos_token_id=0,
            indexer_n_heads=2,
            indexer_kv_heads=1,
            indexer_head_dim=32,
            indexer_budget=8,
            indexer_compress_ratio=2,
        ),
    )


def _tiny_config():
    from freetoken.models.qwen4_exp.config import parse_config

    return parse_config(_tiny_hf_config())


def _model_state_shapes(rank: int, size: int):
    _set_tp(rank, size)
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

    model = Qwen4ExpForCausalLM(_tiny_config())
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def _make_tiny_ckpt(folder: str) -> None:
    """Write a single-shard checkpoint in the checkpoint (pre-fusion) key format."""
    from safetensors.torch import save_file

    cfg = _tiny_config()
    H = cfg.hidden_size
    qo = cfg.num_qo_heads * cfg.head_dim          # 256
    kv = cfg.num_kv_heads * cfg.head_dim          # 128
    g = cfg.linear_attention_group()
    key_dim = g.num_key_heads * g.key_head_dim    # 16
    value_dim = g.num_value_heads * g.value_head_dim  # 32
    conv_dim = 2 * key_dim + value_dim            # 64
    inter = cfg.shared_expert_intermediate_size   # 32
    hc = cfg.qwen4_args.hc_count
    lowrank = cfg.qwen4_args.hc_lowrank
    ple_dim = cfg.qwen4_args.ple_embed_dim
    num_experts = cfg.num_experts
    moe_inter = cfg.moe_intermediate_size

    t: dict[str, torch.Tensor] = {}
    P = "model.language_model."
    t[P + "embed_tokens.weight"] = torch.randn(cfg.vocab_size, H)
    t["lm_head.weight"] = torch.randn(cfg.vocab_size, H)  # top-level (ForCausalLM)
    # top-level hyper-connection mixer (no injection, never fused)
    t[P + "hyper_connection_mixer.hc_norm.weight"] = torch.randn(hc * H)
    t[P + "hyper_connection_mixer.input_mix_weight_down.weight"] = torch.randn(lowrank, hc * H)
    t[P + "hyper_connection_mixer.input_mix_weight_up.weight"] = torch.randn(hc * H, lowrank)
    for i in range(cfg.num_layers):
        lp = f"{P}layers.{i}."
        if cfg.is_linear_layer(i):
            t[lp + "linear_attn.in_proj_qkv.weight"] = torch.randn(conv_dim, H)
            t[lp + "linear_attn.in_proj_z.weight"] = torch.randn(value_dim, H)
            t[lp + "linear_attn.in_proj_b.weight"] = torch.randn(g.num_value_heads, H)
            t[lp + "linear_attn.in_proj_a.weight"] = torch.randn(g.num_value_heads, H)
            t[lp + "linear_attn.conv1d.weight"] = torch.randn(conv_dim, 1, g.conv_kernel_dim)
            t[lp + "linear_attn.A_log"] = torch.randn(g.num_value_heads)
            t[lp + "linear_attn.dt_bias"] = torch.randn(g.num_value_heads)
            t[lp + "linear_attn.norm.weight"] = torch.randn(g.value_head_dim)
            t[lp + "linear_attn.out_proj.weight"] = torch.randn(H, value_dim)
        else:
            t[lp + "self_attn.q_proj.weight"] = torch.randn(2 * qo, H)
            t[lp + "self_attn.k_proj.weight"] = torch.randn(kv, H)
            t[lp + "self_attn.v_proj.weight"] = torch.randn(kv, H)
            t[lp + "self_attn.o_proj.weight"] = torch.randn(qo, H)
            t[lp + "self_attn.q_norm.weight"] = torch.randn(cfg.head_dim)
            t[lp + "self_attn.k_norm.weight"] = torch.randn(cfg.head_dim)
            a = cfg.qwen4_args
            t[lp + "self_attn.indexer.index_qk_proj.weight"] = torch.randn(
                (a.index_n_heads + a.index_kv_heads) * a.index_head_dim, H)
            t[lp + "self_attn.indexer.q_layernorm.weight"] = torch.randn(a.index_head_dim)
            t[lp + "self_attn.indexer.k_layernorm.weight"] = torch.randn(a.index_head_dim)

        t[lp + "mlp.gate.weight"] = torch.randn(num_experts, H)
        t[lp + "mlp.shared_expert_gate.weight"] = torch.randn(1, H)
        t[lp + "mlp.shared_expert.gate_proj.weight"] = torch.randn(inter, H)
        t[lp + "mlp.shared_expert.up_proj.weight"] = torch.randn(inter, H)
        t[lp + "mlp.shared_expert.down_proj.weight"] = torch.randn(H, inter)
        # routed experts: the loader must skip these (offload source banks)
        for e in range(2):
            t[lp + f"mlp.experts.{e}.gate_proj.weight"] = torch.randn(moe_inter, H)
            t[lp + f"mlp.experts.{e}.up_proj.weight"] = torch.randn(moe_inter, H)
            t[lp + f"mlp.experts.{e}.down_proj.weight"] = torch.randn(H, moe_inter)

        for hc_name in ("attn_hyper_connection", "mlp_hyper_connection"):
            t[lp + f"{hc_name}.hc_norm.weight"] = torch.randn(hc * H)
            t[lp + f"{hc_name}.input_mix_weight_down.weight"] = torch.randn(lowrank, hc * H)
            t[lp + f"{hc_name}.block_inject_weight.weight"] = torch.randn(hc, hc * H)
            t[lp + f"{hc_name}.input_mix_weight_up.weight"] = torch.randn(hc * H, lowrank)

        if i in cfg.qwen4_args.ple_layer_ids:
            num_heads = cfg.qwen4_args.heads_per_ngram * (cfg.qwen4_args.ngram_size - 1)
            t[lp + "ple.ple_embedding.layer_multipliers"] = torch.arange(cfg.qwen4_args.ngram_size, dtype=torch.int64)
            t[lp + "ple.ple_embedding.ngram_heads_vocab_sizes"] = torch.full(
                (num_heads,), 64, dtype=torch.int64)
            t[lp + "ple.ple_embedding.ngram_heads_offsets"] = torch.arange(num_heads, dtype=torch.int64)
            t[lp + "ple.key_proj.weight"] = torch.randn(hc * H, ple_dim)
            t[lp + "ple.value_proj.weight"] = torch.randn(H, ple_dim)
            t[lp + "ple.norm_key.weight"] = torch.randn(hc * H)
            t[lp + "ple.norm_query.weight"] = torch.randn(hc * H)
            t[lp + "ple.norm_conv.weight"] = torch.randn(hc * H)
            t[lp + "ple.conv1d.weight"] = torch.randn(hc * H, 1, cfg.qwen4_args.ple_conv_kernel_size)
            # n-gram table + scale: the loader must skip these (load_ple_table owns them)
            t[lp + "ple.ple_embedding.ngram_embedding.shard_0.weight"] = torch.randn(4, ple_dim)
            t[lp + "ple.ple_embedding.ngram_embedding.weight_scale"] = torch.tensor(0.5)

    os.makedirs(folder, exist_ok=True)
    save_file(t, os.path.join(folder, "model.safetensors"))
    hf = _tiny_hf_config()
    with open(os.path.join(folder, "config.json"), "w") as fh:
        json.dump(vars(hf), fh, default=lambda o: vars(o))


# ======================================================================================
# Shard-math
# ======================================================================================

def test_shard_tp():
    from freetoken.models.qwen4_exp.weight import _shard_tp

    t = torch.arange(32).reshape(8, 4)
    s0, s1 = _shard_tp(t, rank=0, world_size=2, dim=0), _shard_tp(t, rank=1, world_size=2, dim=0)
    assert s0.shape == (4, 4) and s1.shape == (4, 4)
    torch.testing.assert_close(torch.cat([s0, s1]), t)
    assert _shard_tp(t, rank=0, world_size=1, dim=0).equal(t)
    assert _shard_tp(torch.arange(32).reshape(4, 8), rank=0, world_size=2, dim=1).shape == (4, 4)


def test_shard_tp_parts():
    from freetoken.models.qwen4_exp.weight import _shard_tp_parts

    t = torch.arange(48).reshape(12, 4)
    s0 = _shard_tp_parts(t, (4, 4, 4), rank=0, world_size=2)
    s1 = _shard_tp_parts(t, (4, 4, 4), rank=1, world_size=2)
    assert s0.shape == (6, 4)
    for i in range(3):
        torch.testing.assert_close(torch.cat([s0[i * 2:i * 2 + 2], s1[i * 2:i * 2 + 2]]), t[i * 4:i * 4 + 4])


def test_shard_tp_parts_replicate():
    from freetoken.models.qwen4_exp.weight import _shard_tp_parts

    t = torch.arange(16).reshape(8, 2)
    kw = dict(tensor=t, part_sizes=(4, 4), world_size=4, local_part_sizes=(2, 4))
    s0, s1, s2, s3 = [_shard_tp_parts(rank=r, **kw) for r in range(4)]
    assert s0.shape == (6, 2)
    torch.testing.assert_close(s0[:2], s1[:2])   # ranks 0,1 share kv head 0
    torch.testing.assert_close(s2[:2], s3[:2])   # ranks 2,3 share kv head 1
    assert not torch.equal(s0[:2], s2[:2])
    torch.testing.assert_close(s0[2:], s2[2:])   # replicated part identical


def test_in_proj_conv_part_sharded_per_subpart():
    """The in_proj conv part is itself [q | k | v] (q/k each key_dim); each sub-part must be
    sharded individually so the model's [local_key | local_key | local_value] split lines up.
    Regression: chunking the whole conv part as one part misaligns k/v at TP>1 (garbage GDN)."""
    from freetoken.models.qwen4_exp.weight import _shard_tp_parts

    cfg = _tiny_config()
    g = cfg.linear_attention_group()
    key_dim = g.num_key_heads * g.key_head_dim
    value_dim = g.num_value_heads * g.value_head_dim
    v = g.num_value_heads
    q, k, vv, z, b, a = [torch.randn(n, 8) for n in
                         (key_dim, key_dim, value_dim, value_dim, v, v)]
    fused = torch.cat([q, k, vv, z, b, a], dim=0)
    # the part sizes iter_weights actually passes
    s0 = _shard_tp_parts(fused, (key_dim, key_dim, value_dim, value_dim, v, v),
                         rank=0, world_size=2)
    s1 = _shard_tp_parts(fused, (key_dim, key_dim, value_dim, value_dim, v, v),
                         rank=1, world_size=2)
    exp0 = torch.cat([q[:key_dim // 2], k[:key_dim // 2], vv[:value_dim // 2],
                      z[:value_dim // 2], b[:v // 2], a[:v // 2]], dim=0)
    exp1 = torch.cat([q[key_dim // 2:], k[key_dim // 2:], vv[value_dim // 2:],
                      z[value_dim // 2:], b[v // 2:], a[v // 2:]], dim=0)
    torch.testing.assert_close(s0, exp0)
    torch.testing.assert_close(s1, exp1)


def test_maybe_shard_q4_rules():
    from freetoken.models.qwen4_exp.weight import _maybe_shard_q4

    cfg = _tiny_config()
    tp = DistributedInfo(0, 2)
    # row-parallel: input dim sharded (H=256, qo=256, value_dim=32, inter=32)
    w = torch.randn(256, 256)
    s = _maybe_shard_q4("model.layers.1.self_attn.o_proj.weight", w, cfg, tp)
    assert s.shape == (256, 128)
    s = _maybe_shard_q4("model.layers.0.linear_attn.out_proj.weight", torch.randn(256, 32), cfg, tp)
    assert s.shape == (256, 16)
    s = _maybe_shard_q4("model.layers.0.mlp.shared_expert.down_proj.weight", torch.randn(256, 32), cfg, tp)
    assert s.shape == (256, 16)
    # vocab-parallel / per-head: dim 0
    assert _maybe_shard_q4("model.lm_head.weight", torch.randn(128, 256), cfg, tp).shape == (64, 256)
    assert _maybe_shard_q4("model.layers.0.linear_attn.A_log", torch.randn(4), cfg, tp).shape == (2,)
    assert _maybe_shard_q4("model.layers.0.linear_attn.dt_bias", torch.randn(4), cfg, tp).shape == (2,)
    # GDN conv1d: per-part (key|key|value) = (16, 16, 32) -> (8, 8, 16)
    s = _maybe_shard_q4("model.layers.0.linear_attn.conv1d.weight", torch.randn(64, 1, 4), cfg, tp)
    assert s.shape == (32, 1, 4)
    # replicated: PLE conv1d, indexer, norms, HC
    for name, t in [
        ("model.layers.0.ple.conv1d.weight", torch.randn(512, 1, 4)),
        ("model.layers.1.self_attn.indexer.index_qk_proj.weight", torch.randn(96, 256)),
        ("model.layers.1.self_attn.q_norm.weight", torch.randn(64)),
        ("model.layers.0.attn_hyper_connection.input_mix_weight_up.weight", torch.randn(16, 16)),
    ]:
        assert _maybe_shard_q4(name, t, cfg, tp).shape == t.shape, name
    # tp=1: identity
    tp1 = DistributedInfo(0, 1)
    assert _maybe_shard_q4("model.lm_head.weight", torch.randn(128, 256), cfg, tp1).shape == (128, 256)


# ======================================================================================
# End to end: iter_weights(TP=2) shapes == model state dict (TP=2)
# ======================================================================================

def test_iter_weights_tp2_matches_model_state_dict():
    import tempfile

    from freetoken.models.qwen4_exp.weight import iter_weights

    _set_tp(0, 1)
    with tempfile.TemporaryDirectory() as d:
        _make_tiny_ckpt(d)
        for rank in (0, 1):
            _set_tp(rank, 2)
            got = {k: tuple(v.shape) for k, v in iter_weights(
                d, torch.device("cpu"), include_moe_experts=False, include_non_moe=True)}
            want = _model_state_shapes(rank, 2)
            for k, v in got.items():
                assert k in want, f"unexpected loader key {k}"
                assert v == want[k], f"{k}: loader {v} vs model {want[k]}"
            # every model key must be covered, except the PLE table backends (attached by
            # load_host_tables, not the loader), the stacked routed-expert banks (offload
            # source banks / resident expert load, never the dense loader), and dummy fill
            missing = {k for k in want
                       if k not in got and "ngram_embedding" not in k and ".mlp.experts." not in k}
            assert not missing, f"loader missed {sorted(missing)}"
    _set_tp(0, 1)


def test_iter_weights_tp1_unchanged():
    import tempfile

    from freetoken.models.qwen4_exp.weight import iter_weights

    _set_tp(0, 1)
    with tempfile.TemporaryDirectory() as d:
        _make_tiny_ckpt(d)
        got = {k: tuple(v.shape) for k, v in iter_weights(
            d, torch.device("cpu"), include_moe_experts=False, include_non_moe=True)}
        want = _model_state_shapes(0, 1)
        for k, v in got.items():
            assert k in want, f"unexpected loader key {k}"
            assert v == want[k], f"{k}: loader {v} vs model {want[k]}"
