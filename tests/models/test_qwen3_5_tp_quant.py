"""CPU unit tests for the TP>1 quantized-checkpoint sharding (qwen3_5_moe).

Covers the weight-loader shard math (fused fp8 qkv / in_proj_qkvz with GQA KV-head
replication, block-fp8 per-part + scale sharding) and the row-parallel / column-merged
quantized linear layers (via the pure-torch FREETOKEN_DEBUG_FP8_REF reference path).

Run:  FREETOKEN_DEBUG_FP8_REF=1 pytest tests/models/test_qwen3_5_tp_quant.py
"""

import os

import torch
import torch.distributed as dist

os.environ.setdefault("FREETOKEN_DEBUG_FP8_REF", "1")

import freetoken.distributed.info as _tpinfo  # noqa: E402
from freetoken.distributed.info import DistributedInfo  # noqa: E402


def _set_tp(rank: int, size: int):
    _tpinfo._TP_INFO = DistributedInfo(rank, size)


def _ensure_single_rank_group():
    """Single-process gloo group: makes DistributedCommunicator.all_reduce a no-op
    (world_size==1), so a two-rank TP test can sum the partials by hand."""
    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method="file:///tmp/ft_tp_quant_test_init",
                                rank=0, world_size=1)


FP8 = torch.float8_e4m3fn


def _rand_fp8(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * 0.1).to(FP8)


# ======================================================================================
# _pt_fp8_fuse (mixed-precision checkpoint: per-tensor FP8 attn/GDN)
# ======================================================================================
def test_pt_fp8_fuse_qkv_shard_tp2():
    from freetoken.models.qwen3_5_moe.weight import _pt_fp8_fuse

    # Qwen3.6-35B dims: q 4096 (16 heads x 256 x 2 for the gate), k/v 512 (2 heads x 256).
    K = 2048
    q = _rand_fp8((4096, K), 1)
    k = _rand_fp8((512, K), 2)
    v = _rand_fp8((512, K), 3)
    sq, sk, sv = 0.2, 0.3, 0.4
    act = torch.tensor(0.205)

    local_kv = 256  # 2 KV heads / 2 ranks
    out = {}
    for rank in (0, 1):
        buf = {}
        _set_tp(rank, 2)
        e = _pt_fp8_fuse("model.layers.3.self_attn.q_proj", q, torch.tensor(sq), act,
                         buf, DistributedInfo(rank, 2), (None, local_kv, local_kv))
        assert e == []
        e = _pt_fp8_fuse("model.layers.3.self_attn.k_proj", k, torch.tensor(sk), act,
                         buf, DistributedInfo(rank, 2), (None, local_kv, local_kv))
        assert e == []
        e = _pt_fp8_fuse("model.layers.3.self_attn.v_proj", v, torch.tensor(sv), act,
                         buf, DistributedInfo(rank, 2), (None, local_kv, local_kv))
        assert e is not None and e != []
        out[rank] = dict(e)

    w0, w1 = out[0]["model.layers.3.self_attn.qkv_proj.weight"], \
        out[1]["model.layers.3.self_attn.qkv_proj.weight"]
    assert w0.shape == (2048 + 256 + 256, K) and w1.shape == w0.shape
    torch.testing.assert_close(w0[:2048], q[:2048])
    torch.testing.assert_close(w0[2048:2304], k[:256])
    torch.testing.assert_close(w0[2304:], v[:256])
    torch.testing.assert_close(w1[:2048], q[2048:])
    torch.testing.assert_close(w1[2048:2304], k[256:])
    torch.testing.assert_close(w1[2304:], v[256:])
    # per-row scale: each part's scalar across its local rows
    s0 = out[0]["model.layers.3.self_attn.qkv_proj.weight_scale"]
    assert s0.shape == (2560,)
    torch.testing.assert_close(s0[:2048], torch.full((2048,), sq, dtype=torch.float32))
    torch.testing.assert_close(s0[2048:2304], torch.full((256,), sk, dtype=torch.float32))
    torch.testing.assert_close(s0[2304:], torch.full((256,), sv, dtype=torch.float32))
    # shared input_scale (max) is unchanged
    torch.testing.assert_close(out[0]["model.layers.3.self_attn.qkv_proj.input_scale"],
                               torch.tensor(0.205))


def test_pt_fp8_fuse_qkv_tp1_unchanged():
    from freetoken.models.qwen3_5_moe.weight import _pt_fp8_fuse

    q = _rand_fp8((4096, 64), 1)
    k = _rand_fp8((512, 64), 2)
    v = _rand_fp8((512, 64), 3)
    buf = {}
    _set_tp(0, 1)
    e = _pt_fp8_fuse("l.self_attn.q_proj", q, torch.tensor(0.2), None,
                     buf, DistributedInfo(0, 1), None)
    e = _pt_fp8_fuse("l.self_attn.k_proj", k, torch.tensor(0.3), None,
                     buf, DistributedInfo(0, 1), None)
    e = _pt_fp8_fuse("l.self_attn.v_proj", v, torch.tensor(0.4), None,
                     buf, DistributedInfo(0, 1), None)
    d = dict(e)
    torch.testing.assert_close(d["l.self_attn.qkv_proj.weight"],
                               torch.cat([q, k, v], dim=0))


def test_pt_fp8_fuse_qkvz_shard_tp2():
    from freetoken.models.qwen3_5_moe.weight import _pt_fp8_fuse

    # GDN: qkv = key_dim(2048) + key_dim(2048) + value_dim(4096); z = value_dim(4096).
    K = 2048
    qkv = _rand_fp8((8192, K), 7)
    z = _rand_fp8((4096, K), 8)
    buf = {}
    _set_tp(1, 2)
    e = _pt_fp8_fuse("l.linear_attn.in_proj_qkv", qkv, torch.tensor(0.1), None,
                     buf, DistributedInfo(1, 2), None)
    assert e == []
    e = _pt_fp8_fuse("l.linear_attn.in_proj_z", z, torch.tensor(0.5), None,
                     buf, DistributedInfo(1, 2), None)
    d = dict(e)
    w = d["l.linear_attn.in_proj_qkvz.weight"]
    assert w.shape == (1024 + 1024 + 2048 + 2048, K)
    torch.testing.assert_close(w[:1024], qkv[1024:2048])        # rank 1: q[1024:2048]
    torch.testing.assert_close(w[1024:2048], qkv[3072:4096])    # k[1024:2048]
    torch.testing.assert_close(w[2048:4096], qkv[4096 + 2048:])  # v[2048:4096]
    torch.testing.assert_close(w[4096:], z[2048:])              # z[2048:4096]
    s = d["l.linear_attn.in_proj_qkvz.weight_scale"]
    torch.testing.assert_close(s[:4096], torch.full((4096,), 0.1, dtype=torch.float32))
    torch.testing.assert_close(s[4096:], torch.full((2048,), 0.5, dtype=torch.float32))


def test_pt_fp8_fuse_qkv_kv_replicate_tp4():
    from freetoken.models.qwen3_5_moe.weight import _pt_fp8_fuse

    # 2 KV heads at TP=4 -> each KV head replicated on 2 ranks.
    K = 64
    q = _rand_fp8((4096, K), 1)
    k = _rand_fp8((512, K), 2)
    v = _rand_fp8((512, K), 3)
    local_kv = 256  # 1 head per rank, replicated
    out = {}
    for rank in range(4):
        buf = {}
        _set_tp(rank, 4)
        for name, t, s in (("q_proj", q, 0.2), ("k_proj", k, 0.3), ("v_proj", v, 0.4)):
            e = _pt_fp8_fuse(f"l.self_attn.{name}", t, torch.tensor(s), None,
                             buf, DistributedInfo(rank, 4), (None, local_kv, local_kv))
        out[rank] = dict(e)["l.self_attn.qkv_proj.weight"]
    # local layout: q 4096/4=1024, then k 256, then v 256
    # ranks 0,1 share KV head 0; ranks 2,3 share KV head 1
    torch.testing.assert_close(out[0][1024:1280], out[1][1024:1280])
    torch.testing.assert_close(out[2][1024:1280], out[3][1024:1280])
    torch.testing.assert_close(out[0][1024:1280], k[:256])
    torch.testing.assert_close(out[2][1024:1280], k[256:])
    torch.testing.assert_close(out[0][1280:], out[1][1280:])
    torch.testing.assert_close(out[0][1280:], v[:256])
    # q is a plain /4 split
    torch.testing.assert_close(out[0][:1024], q[:1024])
    torch.testing.assert_close(out[3][:1024], q[3072:])


# ======================================================================================
# _fp8_shard_part (block-fp8 checkpoint: 128x128 block scales)
# ======================================================================================
def test_fp8_shard_part_qkv_weight_and_scale():
    from freetoken.models.qwen3_5_moe.weight import _fp8_shard_part

    class Gdn:
        num_key_heads, key_head_dim = 16, 128
        num_value_heads, value_head_dim = 32, 128

    K, KB = 2048, 2048 // 128
    q = _rand_fp8((4096, K), 1)
    k = _rand_fp8((512, K), 2)
    v = _rand_fp8((512, K), 3)
    qs = torch.randn(4096 // 128, KB)
    ks = torch.randn(512 // 128, KB)
    vs = torch.randn(512 // 128, KB)
    qkv_local = (None, 256, 256)
    tp1 = DistributedInfo(1, 2)

    for suf, parts, locals_ in (
        (".weight", (q, k, v), (2048, 256, 256)),
        (".weight_scale_inv", (qs, ks, vs), (16, 2, 2)),
    ):
        got = [
            _fp8_shard_part(".self_attn.qkv_proj", i, t, suf, tp1, qkv_local, Gdn())
            for i, t in enumerate(parts)
        ]
        for g, t, local in zip(got, parts, locals_):
            assert g.shape[0] == local, (g.shape, local)
    # rank 1 takes the second half of every part
    g = _fp8_shard_part(".self_attn.qkv_proj", 0, q, ".weight", tp1, qkv_local, Gdn())
    torch.testing.assert_close(g, q[2048:])
    g = _fp8_shard_part(".self_attn.qkv_proj", 1, ks, ".weight_scale_inv", tp1, qkv_local, Gdn())
    torch.testing.assert_close(g, ks[2:])  # rank 1: second of two scale blocks


def test_fp8_shard_part_qkvz_expands():
    from freetoken.models.qwen3_5_moe.weight import _fp8_shard_part

    class Gdn:
        num_key_heads, key_head_dim = 16, 128
        num_value_heads, value_head_dim = 32, 128

    K = 2048
    qkv = _rand_fp8((8192, K), 5)
    z = _rand_fp8((4096, K), 6)
    tp0 = DistributedInfo(0, 2)
    w = _fp8_shard_part(".linear_attn.in_proj_qkvz", 0, qkv, ".weight", tp0, None, Gdn())
    assert w.shape == (1024 + 1024 + 2048, K)
    torch.testing.assert_close(w[:1024], qkv[:1024])
    torch.testing.assert_close(w[1024:2048], qkv[2048:3072])
    torch.testing.assert_close(w[2048:], qkv[4096:6144])
    wz = _fp8_shard_part(".linear_attn.in_proj_qkvz", 1, z, ".weight", tp0, None, Gdn())
    torch.testing.assert_close(wz, z[:2048])
    # gate_up groups stay full (replicated model)
    gu = torch.randn(1024, K)
    torch.testing.assert_close(
        _fp8_shard_part(".mlp.shared_expert.gate_up_proj", 0, gu, ".weight", tp0, None, Gdn()),
        gu)


# ======================================================================================
# Row-parallel / column-merged quantized linears (pure-torch reference path)
# ======================================================================================
def test_fp8_pertensor_row_parallel_two_rank_sum():
    from freetoken.kernel.triton.fp8_pertensor_linear import (
        Fp8PerTensorLinear, Fp8PerTensorRowParallel,
    )

    N, K = 256, 512
    g = torch.Generator().manual_seed(0)
    w_bf16 = torch.randn(N, K, generator=g) * 0.05
    scale = 0.25
    w_fp8 = (w_bf16 / scale).to(FP8)
    x = torch.randn(7, K, dtype=torch.bfloat16, generator=g)

    full = Fp8PerTensorLinear(K, N)
    full.weight.copy_(w_fp8)
    full.weight_scale.copy_(torch.full((N,), scale))
    full._uniform_scale = True
    ref = full.forward(x)

    partials = []
    _ensure_single_rank_group()
    for rank in (0, 1):
        _set_tp(rank, 2)
        layer = Fp8PerTensorRowParallel(K, N)
        assert layer.weight.shape == (N, K // 2)
        assert layer.weight_scale.shape == (N,)
        shard = w_fp8[:, rank * K // 2:(rank + 1) * K // 2]
        layer.weight.copy_(shard)
        layer.weight_scale.copy_(torch.full((N,), scale))
        layer._uniform_scale = True
        partials.append(layer.forward(x[:, rank * K // 2:(rank + 1) * K // 2]))
    # all_reduce is the identity on the single-rank group: the two partials must sum
    # to the full (replicated) computation.
    torch.testing.assert_close(partials[0] + partials[1], ref, atol=2e-2, rtol=2e-2)


def test_fp8_pertensor_col_merged_local_sizes():
    from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorColMerged

    _set_tp(1, 2)
    layer = Fp8PerTensorColMerged(2048, [4096, 512, 512],
                                  local_output_sizes=[2048, 256, 256])
    assert layer.weight.shape == (2560, 2048)
    assert layer.weight_scale.shape == (2560,)
    # TP=1 default unchanged
    _set_tp(0, 1)
    full = Fp8PerTensorColMerged(2048, [4096, 512, 512])
    assert full.weight.shape == (5120, 2048)


def test_nvfp4_row_parallel_shapes():
    from freetoken.kernel.triton.nvfp4_linear import (
        Nvfp4DenseColMerged, Nvfp4DenseRowParallel,
    )

    _set_tp(0, 2)
    row = Nvfp4DenseRowParallel(2048, 2048)
    assert row.weight.shape == (2048, 1024 // 2)
    assert row.weight_scale.shape == (2048, 1024 // 16)
    assert row.weight_global.shape == (2048,)
    col = Nvfp4DenseColMerged(2048, [512, 512], local_output_sizes=[256, 256])
    assert col.weight.shape == (512, 2048 // 2)
    assert col.weight_scale.shape == (512, 2048 // 16)
    # load_state_dict round-trips the sharded layout (K-major repack)
    row.load_state_dict({
        "weight": torch.randint(0, 255, (2048, 512), dtype=torch.uint8),
        "weight_scale": torch.randint(1, 10, (2048, 64), dtype=torch.uint8)
        .view(torch.float8_e4m3fn),
        "weight_global": torch.rand(2048, dtype=torch.float16),
    })
    assert row.weight.shape == (1024 // 8, 2048)  # K-major [K//8, N]
    assert row.weight_scale.shape == (1024 // 16, 2048)
