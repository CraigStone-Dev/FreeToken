import torch

import freetoken.distributed.info as _tpinfo
from freetoken.distributed.info import DistributedInfo
from freetoken.models.qwen3_5_moe.weight import (
    _maybe_shard,
    _shard_nvfp4_emit,
    _shard_tp,
    _shard_tp_parts,
)


def _set_tp(rank: int, size: int):
    _tpinfo._TP_INFO = DistributedInfo(rank, size)


def test_shard_tp():
    t = torch.arange(32).reshape(8, 4)
    s0, s1 = _shard_tp(t, rank=0, world_size=2, dim=0), _shard_tp(t, rank=1, world_size=2, dim=0)
    assert s0.shape == (4, 4) and s1.shape == (4, 4)
    torch.testing.assert_close(torch.cat([s0, s1]), t)
    assert _shard_tp(t, rank=0, world_size=1, dim=0).equal(t)
    assert _shard_tp(torch.arange(32).reshape(4, 8), rank=0, world_size=2, dim=1).shape == (4, 4)


def test_shard_tp_parts():
    t = torch.arange(48).reshape(12, 4)
    s0 = _shard_tp_parts(t, (4, 4, 4), rank=0, world_size=2)
    s1 = _shard_tp_parts(t, (4, 4, 4), rank=1, world_size=2)
    assert s0.shape == (6, 4)
    for i in range(3):
        torch.testing.assert_close(torch.cat([s0[i*2:i*2+2], s1[i*2:i*2+2]]), t[i*4:i*4+4])


def test_shard_tp_parts_replicate():
    t = torch.arange(16).reshape(8, 2)
    local = (2, 4)
    kw = dict(tensor=t, part_sizes=(4, 4), world_size=4, local_part_sizes=local)
    s0, s1, s2, s3 = [_shard_tp_parts(rank=r, **kw) for r in range(4)]
    assert s0.shape == (6, 2)
    torch.testing.assert_close(s0[:2], s1[:2])   # ranks 0,1 share head 0
    torch.testing.assert_close(s2[:2], s3[:2])   # ranks 2,3 share head 1
    assert not torch.equal(s0[:2], s2[:2])        # different heads
    torch.testing.assert_close(s0[2:], s2[2:])    # replicated part identical


def test_maybe_shard_tp1_passthrough():
    _set_tp(0, 1)
    try:
        # world_size=1: every key returns the tensor unchanged (no clone)
        for name in (
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.linear_attn.out_proj.weight",
            "model.layers.0.mlp.shared_expert.down_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.layers.0.linear_attn.A_log",
            "model.layers.0.linear_attn.dt_bias",
            "model.layers.0.linear_attn.conv1d.weight",
        ):
            t = torch.zeros(4, 4)
            assert _maybe_shard(name, t) is t
    finally:
        _tpinfo._TP_INFO = None


def test_maybe_shard_tp2():
    _set_tp(0, 2)
    try:
        # row-parallel: shard the input dim
        t = torch.arange(32).reshape(4, 8)
        s = _maybe_shard("model.layers.0.self_attn.o_proj.weight", t)
        assert s.shape == (4, 4)
        torch.testing.assert_close(s, t[:, :4])
        # vocab-parallel: shard the row dim
        s = _maybe_shard("lm_head.weight", t)
        assert s.shape == (2, 8)
        torch.testing.assert_close(s, t[:2])
        # per-head vectors
        v = torch.arange(8).reshape(4, 2)
        s = _maybe_shard("model.layers.0.linear_attn.A_log", v)
        assert s.shape == (2, 2)
        torch.testing.assert_close(s, v[:2])
        # stacked resident experts: gate/up sharded independently on dim 1
        e = torch.arange(16).reshape(2, 4, 2)  # [E=2, 2*I=4, H=2]
        s = _maybe_shard("model.layers.0.mlp.experts.gate_up_proj", e)
        assert s.shape == (2, 2, 2)
        # rank 0: first half of gate (rows 0) + first half of up (rows 2)
        torch.testing.assert_close(s, torch.cat([e[:, :1, :], e[:, 2:3, :]], dim=1))
        d = torch.arange(2 * 2 * 4).reshape(2, 2, 4)  # [E=2, H=2, I=4]
        s = _maybe_shard("model.layers.0.mlp.experts.down_proj", d)
        assert s.shape == (2, 2, 2)
        torch.testing.assert_close(s[0], d[0][:, :2])
    finally:
        _tpinfo._TP_INFO = None


def test_shard_nvfp4_emit_tp1_passthrough():
    _set_tp(0, 1)
    try:
        w = torch.zeros(8, 4, dtype=torch.uint8)
        s = torch.zeros(8, 1, dtype=torch.float8_e4m3fn)
        g = torch.zeros(8, dtype=torch.float16)
        emit = [("x.weight", w), ("x.weight_scale", s), ("x.weight_global", g)]
        assert _shard_nvfp4_emit(emit, "model.layers.0.mlp.shared_expert.gate_up_proj") is emit
    finally:
        _tpinfo._TP_INFO = None


def test_shard_nvfp4_emit_gate_up_tp2():
    _set_tp(0, 2)
    try:
        # gate_up_proj: [2*I, H//2] packed, [2*I, H//16] scale, [2*I] global; I=4, H=16
        w = torch.arange(8 * 8).reshape(8, 8).to(torch.uint8)
        s = torch.arange(8 * 2).reshape(8, 2).to(torch.float8_e4m3fn)
        g = torch.arange(8, dtype=torch.float16)
        emit = [("x.gate_up_proj.weight", w), ("x.gate_up_proj.weight_scale", s),
                ("x.gate_up_proj.weight_global", g)]
        out = _shard_nvfp4_emit(emit, "model.layers.0.mlp.shared_expert.gate_up_proj")
        ow, os_, og = (t for _, t in out)
        assert ow.shape == (4, 8) and os_.shape == (4, 2) and og.shape == (4,)
        # gate (rows 0..3) and up (rows 4..7) each halved
        torch.testing.assert_close(ow[:2], w[:2])
        torch.testing.assert_close(ow[2:], w[4:6])
        torch.testing.assert_close(og, torch.cat([g[:2], g[4:6]]))
    finally:
        _tpinfo._TP_INFO = None


def test_shard_nvfp4_emit_down_tp2():
    _set_tp(1, 2)
    try:
        # down_proj: [H, I//2] packed, [H, I//16] scale, [H] global; H=4, I=16
        w = torch.arange(4 * 8).reshape(4, 8).to(torch.uint8)
        s = torch.arange(4 * 2).reshape(4, 2).to(torch.float8_e4m3fn)
        g = torch.arange(4, dtype=torch.float16)
        emit = [("x.down_proj.weight", w), ("x.down_proj.weight_scale", s),
                ("x.down_proj.weight_global", g)]
        out = _shard_nvfp4_emit(emit, "model.layers.0.mlp.shared_expert.down_proj")
        ow, os_, og = (t for _, t in out)
        assert ow.shape == (4, 4) and os_.shape == (4, 1) and og.shape == (4,)
        torch.testing.assert_close(ow, w[:, 4:])  # rank 1 takes the second half of I
        torch.testing.assert_close(os_, s[:, 1:])
        torch.testing.assert_close(og, g)  # global keeps full output rows
    finally:
        _tpinfo._TP_INFO = None


def test_shard_nvfp4_emit_lm_head_replicated():
    _set_tp(0, 2)
    try:
        w = torch.arange(8 * 8).reshape(8, 8).to(torch.uint8)
        s = torch.arange(8 * 2).reshape(8, 2).to(torch.float8_e4m3fn)
        g = torch.arange(8, dtype=torch.float16)
        emit = [("lm_head.weight", w), ("lm_head.weight_scale", s), ("lm_head.weight_global", g)]
        out = _shard_nvfp4_emit(emit, "lm_head")
        assert out is emit  # Nvfp4LMHead is replicated
    finally:
        _tpinfo._TP_INFO = None


def test_shard_tp_parts_3d_conv():
    # conv1d.weight is [conv_dim, 1, K]; per-part sharding slices the channel dim
    t = torch.arange(24).reshape(12, 1, 2)
    s0 = _shard_tp_parts(t, (4, 4, 4), rank=0, world_size=2)
    s1 = _shard_tp_parts(t, (4, 4, 4), rank=1, world_size=2)
    assert s0.shape == (6, 1, 2)
    # per-part: rank r gets rows [r*2, r*2+2) of each 4-row part
    torch.testing.assert_close(s0, torch.cat([t[0:2], t[4:6], t[8:10]], dim=0))
    torch.testing.assert_close(s1, torch.cat([t[2:4], t[6:8], t[10:12]], dim=0))
