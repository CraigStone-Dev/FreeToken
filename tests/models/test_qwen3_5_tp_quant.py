"""TP=2 quantized loader tests for qwen3_5_moe (block-fp8 + modelopt mixed checkpoints).

Covers the loader-side TP sharding: ``_shard_dense_part`` (per-head column parts),
``_shard_fp8_tensor`` (weight + weight_scale_inv), the end-to-end ``iter_weights``
sharding for both checkpoint kinds, and the TP-sharded resident fp8 expert banks.
"""
from __future__ import annotations

import tempfile

import pytest
import torch

import freetoken.distributed.info as _tpinfo
from freetoken.distributed.info import DistributedInfo
from freetoken.models.qwen3_5_moe.config import parse_config
from freetoken.models.qwen3_5_moe.weight import (
    _resident_fp8_experts,
    _shard_dense_part,
    _shard_fp8_tensor,
    iter_weights,
)
from freetoken.utils.hf import cached_load_hf_config

from tests.models.tiny_fp8_ckpt import make_tiny_fp8_ckpt, make_tiny_mixed_ckpt


def _set_tp(rank: int, size: int):
    _tpinfo._TP_INFO = DistributedInfo(rank, size)


@pytest.fixture(autouse=True)
def _clear_tp():
    yield
    _tpinfo._TP_INFO = None


# Tiny config matching the tiny checkpoints (16 q heads, 2 kv heads, 16 k / 32 v GDN heads)
@pytest.fixture(scope="module")
def cfg():
    d = tempfile.mkdtemp()
    ckpt = make_tiny_fp8_ckpt(d, vocab=512, layers=4, experts=8)
    return parse_config(cached_load_hf_config(ckpt))


# ======================================================================================
# _shard_dense_part (per-head column parts; weight or its //128 scale)
# ======================================================================================
def test_shard_dense_part_qkv_tp2(cfg):
    _set_tp(0, 2)
    q, k, v = 8192, 512, 512
    w = torch.randn(q + k + v, 2048)
    r = _shard_dense_part("model.layers.3.self_attn.q_proj", w[:q], cfg)
    assert r.shape == (q // 2, 2048)
    assert torch.equal(r, w[:q][:q // 2])
    _set_tp(1, 2)
    r1 = _shard_dense_part("model.layers.3.self_attn.q_proj", w[:q], cfg)
    assert torch.equal(r1, w[:q][q // 2:])


def test_shard_dense_part_kv_replicate_tp4(cfg):
    # 2 kv heads over 4 ranks: rank 0/1 get head 0, rank 2/3 get head 1
    kv = 512
    w = torch.randn(kv, 2048)
    _set_tp(0, 4)
    r0 = _shard_dense_part("model.layers.3.self_attn.k_proj", w, cfg)
    _set_tp(1, 4)
    r1 = _shard_dense_part("model.layers.3.self_attn.k_proj", w, cfg)
    _set_tp(2, 4)
    r2 = _shard_dense_part("model.layers.3.self_attn.k_proj", w, cfg)
    _set_tp(3, 4)
    r3 = _shard_dense_part("model.layers.3.self_attn.k_proj", w, cfg)
    assert r0.shape == (256, 2048)
    assert torch.equal(r0, r1)
    assert torch.equal(r0, w[:256])
    assert torch.equal(r2, r3)
    assert torch.equal(r2, w[256:])


def test_shard_dense_part_in_proj_qkv_expands(cfg):
    _set_tp(0, 2)
    key_dim = 16 * 128
    value_dim = 32 * 128
    w = torch.randn(2 * key_dim + value_dim, 2048)
    r = _shard_dense_part("model.layers.0.linear_attn.in_proj_qkv", w, cfg)
    # local: q 8 heads * 256, k 8 heads * 128, v 16 heads * 128
    assert r.shape == (key_dim // 2 + key_dim // 2 + value_dim // 2, 2048)
    assert torch.equal(r[:1024], w[:1024])                      # q local
    assert torch.equal(r[1024:2048], w[2048:3072])              # k local
    assert torch.equal(r[2048:], w[4096:6144])                  # v local
    # the //128 scale carries the same part structure
    s = torch.randn((2 * key_dim + value_dim) // 128, 2048 // 128)
    rs = _shard_dense_part("model.layers.0.linear_attn.in_proj_qkv", s, cfg)
    assert rs.shape == ((key_dim + key_dim + value_dim) // 2 // 128, 2048 // 128)


def test_shard_dense_part_z_ba(cfg):
    _set_tp(1, 2)
    z = torch.randn(4096, 2048)
    r = _shard_dense_part("model.layers.0.linear_attn.in_proj_z", z, cfg)
    assert r.shape == (2048, 2048)
    assert torch.equal(r, z[2048:])
    b = torch.randn(32, 2048)
    rb = _shard_dense_part("model.layers.0.linear_attn.in_proj_b", b, cfg)
    assert torch.equal(rb, b[16:])


# ======================================================================================
# _shard_fp8_tensor (weight + weight_scale_inv, all part kinds)
# ======================================================================================
def test_shard_fp8_tensor_column_and_row(cfg):
    _set_tp(0, 2)
    # column part weight + scale
    w = torch.randn(8192, 2048)
    assert _shard_fp8_tensor("model.layers.3.self_attn.q_proj.weight", w, cfg).shape == (4096, 2048)
    s = torch.randn(64, 16)
    assert _shard_fp8_tensor("model.layers.3.self_attn.q_proj.weight_scale_inv", s, cfg).shape == (32, 16)
    # row-parallel: input dim
    w = torch.randn(2048, 4096)
    assert _shard_fp8_tensor("model.layers.3.self_attn.o_proj.weight", w, cfg).shape == (2048, 2048)
    s = torch.randn(16, 32)
    assert _shard_fp8_tensor("model.layers.3.self_attn.o_proj.weight_scale_inv", s, cfg).shape == (16, 16)
    # vocab-parallel
    w = torch.randn(512, 2048)
    assert _shard_fp8_tensor("lm_head.weight", w, cfg).shape == (256, 2048)
    # per-head vectors (A_log is per value head: 32 -> 16 at TP=2)
    assert _shard_fp8_tensor("model.layers.0.linear_attn.A_log", torch.randn(32), cfg).shape == (16,)
    # replicated (norms) pass through
    n = torch.randn(2048)
    assert torch.equal(_shard_fp8_tensor("model.layers.0.input_layernorm.weight", n, cfg), n)


def test_shard_fp8_tensor_in_proj_qkv_scale(cfg):
    _set_tp(1, 2)
    key_dim, value_dim = 16 * 128, 32 * 128
    s = torch.randn((2 * key_dim + value_dim) // 128, 16)
    r = _shard_fp8_tensor("model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv", s, cfg)
    assert r.shape == ((key_dim + value_dim // 2) // 128, 16)


# ======================================================================================
# iter_weights end-to-end (block-fp8 + mixed)
# ======================================================================================
def _iter_shapes(ckpt, rank, size, **kw):
    _set_tp(rank, size)
    return {n: tuple(t.shape) for n, t in iter_weights(
        ckpt, torch.device("cpu"), include_moe_experts=False, include_non_moe=True, **kw)}


def test_fp8_iter_weights_tp2_sharding():
    d = tempfile.mkdtemp()
    ckpt = make_tiny_fp8_ckpt(d, vocab=512, layers=4, experts=8)
    s0 = _iter_shapes(ckpt, 0, 2)
    s1 = _iter_shapes(ckpt, 1, 2)
    assert set(s0) == set(s1)
    # GDN layer (0): column parts halved, row parts input-halved, A_log/conv1d halved
    assert s0["model.layers.0.linear_attn.in_proj_qkvz.weight"] == (6144, 2048)
    assert s0["model.layers.0.linear_attn.in_proj_qkvz.weight_scale_inv"] == (48, 16)
    assert s0["model.layers.0.linear_attn.in_proj_ba.weight"] == (32, 2048)  # b|a: 16+16
    assert s0["model.layers.0.linear_attn.out_proj.weight"] == (2048, 2048)
    assert s0["model.layers.0.linear_attn.A_log"] == (16,)
    assert s0["model.layers.0.linear_attn.conv1d.weight"] == (4096, 1, 4)
    # full-attn layer (3): qkv halved per part (kv replicated at TP=2), o_proj input-halved
    assert s0["model.layers.3.self_attn.qkv_proj.weight"] == (4608, 2048)
    assert s0["model.layers.3.self_attn.o_proj.weight"] == (2048, 2048)
    # shared expert + vocab
    assert s0["model.layers.0.mlp.shared_expert.gate_up_proj.weight"] == (512, 2048)
    assert s0["model.layers.0.mlp.shared_expert.down_proj.weight"] == (2048, 256)
    assert s0["lm_head.weight"] == (256, 2048)
    assert s0["model.embed_tokens.weight"] == (256, 2048)
    # TP=1 is unchanged (full shapes)
    s1x = _iter_shapes(ckpt, 0, 1)
    assert s1x["model.layers.0.linear_attn.in_proj_qkvz.weight"] == (12288, 2048)
    assert s1x["model.layers.0.linear_attn.in_proj_ba.weight"] == (64, 2048)  # b|a: 32+32
    assert s1x["model.layers.0.linear_attn.A_log"] == (32,)


def test_mixed_iter_weights_tp2_sharding():
    d = tempfile.mkdtemp()
    ckpt = make_tiny_mixed_ckpt(d, vocab=512, layers=4, experts=8)
    s0 = _iter_shapes(ckpt, 0, 2)
    s1 = _iter_shapes(ckpt, 1, 2)
    assert set(s0) == set(s1)
    # per-tensor fp8 fused parts + nvfp4 dense + bf16 parts all shard the same way
    assert s0["model.layers.0.linear_attn.in_proj_qkvz.weight"] == (6144, 2048)
    assert s0["model.layers.0.linear_attn.in_proj_ba.weight"] == (32, 2048)  # b|a: 16+16
    assert s0["model.layers.3.self_attn.qkv_proj.weight"] == (4608, 2048)
    assert s0["model.layers.3.self_attn.o_proj.weight"] == (2048, 2048)
    # shared expert is native NVFP4 (packed [O, IN//2]): rows halved (gate|up parts)
    assert s0["model.layers.0.mlp.shared_expert.gate_up_proj.weight"] == (512, 1024)
    assert s0["lm_head.weight"] == (256, 2048)


# ======================================================================================
# resident fp8 expert banks (TP-sharded on the intermediate dim)
# ======================================================================================
def test_resident_fp8_experts_tp2_sharding():
    d = tempfile.mkdtemp()
    ckpt = make_tiny_fp8_ckpt(d, vocab=512, layers=4, experts=8)
    cfg = parse_config(cached_load_hf_config(ckpt))
    banks = {}
    for rank in (0, 1):
        _set_tp(rank, 2)
        for name, t in _resident_fp8_experts(ckpt, cfg):
            banks.setdefault(name, {})[rank] = t
    # every layer yields the four sharded banks
    assert len(banks) == 4 * 4
    # layer 0: verify the sharded values against the raw checkpoint pieces
    from safetensors import safe_open
    with safe_open(f"{ckpt}/model.safetensors", framework="pt") as f:
        gate = f.get_tensor("model.language_model.layers.0.mlp.experts.0.gate_proj.weight")
        up = f.get_tensor("model.language_model.layers.0.mlp.experts.0.up_proj.weight")
        down = f.get_tensor("model.language_model.layers.0.mlp.experts.0.down_proj.weight")
    r0 = banks["model.layers.0.mlp.experts.gate_up_proj"][0]
    r1 = banks["model.layers.0.mlp.experts.gate_up_proj"][1]
    assert r0.shape == (8, 512, 2048)  # E=8, 2*i=512 (i=256), H
    # rank 0 holds gate rows [0:256] / up rows [0:256]; rank 1 the halves
    # rank 0 holds gate rows [0:256) / up rows [0:256); rank 1 the halves
    assert torch.equal(r0[0, :256].float(), gate[:256].float())
    assert torch.equal(r1[0, :256].float(), gate[256:].float())
    assert torch.equal(r0[0, 256:].float(), up[:256].float())
    assert torch.equal(r1[0, 256:].float(), up[256:].float())
    d0 = banks["model.layers.0.mlp.experts.down_proj"][0]
    d1 = banks["model.layers.0.mlp.experts.down_proj"][1]
    assert d0.shape == (8, 2048, 256)
    assert torch.equal(d0[0].float(), down[:, :256].float())
    assert torch.equal(d1[0].float(), down[:, 256:].float())
    for name, per_rank in banks.items():
        assert set(per_rank) == {0, 1}
        if name.endswith("gate_up_scale_inv"):
            assert per_rank[0].shape == (8, 4, 16)
        elif name.endswith("down_scale_inv"):
            assert per_rank[0].shape == (8, 16, 2)


def test_resident_fp8_experts_tp1_full():
    d = tempfile.mkdtemp()
    ckpt = make_tiny_fp8_ckpt(d, vocab=512, layers=4, experts=8)
    cfg = parse_config(cached_load_hf_config(ckpt))
    _set_tp(0, 1)
    shapes = {n: tuple(t.shape) for n, t in _resident_fp8_experts(ckpt, cfg)}
    assert shapes["model.layers.0.mlp.experts.gate_up_proj"] == (8, 1024, 2048)
    assert shapes["model.layers.0.mlp.experts.down_proj"] == (8, 2048, 512)
