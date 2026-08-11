"""Parity tests for the MiniMax-M3 block-sparse attention kernels.

The pure-torch reference below implements the semantics pinned against the vLLM
reference implementation (see ``scratch/m3_ref/`` and
``kernel/triton/minimax_m3_sparse.py``'s module docstring):

* per-index-head block scores: max over the block's 128 positions of
  ``dot(index_q, index_k)``, causal-masked, no scale;
* top-k selection with forced init/local blocks (score boosts 1e30/1e29, local
  wins overlaps), -1 padding past the visible block count;
* GQA attend over the selected blocks only (each kv head follows its own index
  head), causal inside the newest block.

The physical layout mirrors serving: a row-flat slab addressed through a
per-request ``block_rows`` base-row table built from a random page permutation,
so the tests exercise the gather addressing, not just the math.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required for the M3 sparse kernels", allow_module_level=True)

from freetoken.kernel.triton.minimax_m3_sparse import (
    SPARSE_BLOCK_SIZE,
    m3_index_decode,
    m3_index_score_prefill,
    m3_index_topk_prefill,
    m3_sparse_attn_decode,
    m3_sparse_attn_prefill,
)

BLK = SPARSE_BLOCK_SIZE
DEV = "cuda"


# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------
def ref_block_scores(
    iq: torch.Tensor,  # [T, H, D] queries at absolute positions q_pos
    ik: torch.Tensor,  # [S, D] index keys for positions 0..S-1
    q_pos: torch.Tensor,  # [T] absolute positions
) -> torch.Tensor:
    """[H, T, nblocks] fp32; max over each 128-block, causal, -inf invisible."""
    scores = torch.einsum("thd,sd->hts", iq.float(), ik.float())
    s_pos = torch.arange(ik.shape[0], device=iq.device)
    scores = scores.masked_fill(s_pos[None, None, :] > q_pos[None, :, None], float("-inf"))
    nblocks = (ik.shape[0] + BLK - 1) // BLK
    pad = nblocks * BLK - ik.shape[0]
    if pad:
        scores = torch.nn.functional.pad(scores, (0, pad), value=float("-inf"))
    return scores.view(*scores.shape[:2], nblocks, BLK).amax(dim=-1)


def ref_select(
    block_scores: torch.Tensor,  # [H, T, nblocks]
    q_pos: torch.Tensor,  # [T]
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> list[list[set[int]]]:
    """Per (head, token) SET of selected block ids (order is score-descending in
    the kernels, but only membership is contract -- the attend softmax is
    order-invariant)."""
    H, T, nblocks = block_scores.shape
    out: list[list[set[int]]] = []
    for h in range(H):
        rows = []
        for t in range(T):
            nb = int(q_pos[t]) // BLK + 1
            s = block_scores[h, t, :nb].clone()
            if init_blocks:
                s[: min(init_blocks, nb)] = 1e30
            if local_blocks:
                s[max(0, nb - local_blocks):] = 1e29
            k = min(topk, nb)
            rows.append(set(torch.topk(s, k).indices.tolist()))
        out.append(rows)
    return out


def ref_sparse_attend(
    q: torch.Tensor,  # [T, HQ, D] queries at q_pos
    k: torch.Tensor,  # [S, KVH, D]
    v: torch.Tensor,  # [S, KVH, D]
    selection: list[list[set[int]]],  # [KVH][T] block-id sets
    q_pos: torch.Tensor,  # [T]
    sm_scale: float,
) -> torch.Tensor:
    T, HQ, D = q.shape
    S, KVH, _ = k.shape
    group = HQ // KVH
    out = torch.empty(T, HQ, D, dtype=torch.float32, device=q.device)
    s_pos = torch.arange(S, device=q.device)
    for t in range(T):
        for kh in range(KVH):
            sel = sorted(selection[kh][t])
            pos_mask = torch.zeros(S, dtype=torch.bool, device=q.device)
            for b in sel:
                pos_mask[b * BLK : min((b + 1) * BLK, S)] = True
            pos_mask &= s_pos <= int(q_pos[t])
            idx = pos_mask.nonzero(as_tuple=True)[0]
            kk = k[idx, kh].float()  # [n, D]
            vv = v[idx, kh].float()
            for g in range(group):
                h = kh * group + g
                logits = (q[t, h].float() @ kk.t()) * sm_scale
                out[t, h] = torch.softmax(logits, dim=-1) @ vv
    return out


def make_layout(nblocks: int, seed: int = 0):
    """A shuffled physical layout: logical block i lives at page perm[i]."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(nblocks + 3, generator=g)[:nblocks]  # +3 spare pages
    total_pages = nblocks + 3
    block_rows = (perm * BLK).to(torch.int32).to(DEV)
    return block_rows, total_pages


def scatter_rows(x: torch.Tensor, block_rows: torch.Tensor, total_pages: int):
    """Place logical rows [S, ...] into a physical slab [total_pages*BLK, ...]."""
    S = x.shape[0]
    slab = torch.zeros(total_pages * BLK, *x.shape[1:], dtype=x.dtype, device=x.device)
    for b in range((S + BLK - 1) // BLK):
        n = min(BLK, S - b * BLK)
        base = int(block_rows[b])
        slab[base : base + n] = x[b * BLK : b * BLK + n]
    return slab


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
H_IDX, D_IDX = 4, 128
HQ, KVH, D = 16, 4, 128  # scaled-down GQA (group 4) to keep the reference fast
TOPK, INIT, LOCAL = 4, 0, 1
SCALE = D**-0.5


@pytest.mark.parametrize("kv_len,q_len", [(BLK * 7 + 55, 300), (BLK * 2, BLK * 2), (90, 90)])
def test_prefill_index_score_and_topk(kv_len: int, q_len: int):
    torch.manual_seed(0)
    prefix = kv_len - q_len
    nblocks = (kv_len + BLK - 1) // BLK
    iq = torch.randn(q_len, H_IDX, D_IDX, device=DEV, dtype=torch.bfloat16)
    ik = torch.randn(kv_len, D_IDX, device=DEV, dtype=torch.bfloat16)
    q_pos = torch.arange(prefix, kv_len, device=DEV)

    block_rows, total_pages = make_layout(nblocks)
    ik_slab = scatter_rows(ik, block_rows, total_pages)

    cu = torch.tensor([0, q_len], dtype=torch.int32, device=DEV)
    seq = torch.tensor([kv_len], dtype=torch.int32, device=DEV)
    pre = torch.tensor([prefix], dtype=torch.int32, device=DEV)
    score = m3_index_score_prefill(
        iq, ik_slab, block_rows.view(1, -1), cu, seq, pre, q_len, kv_len
    )
    ref = ref_block_scores(iq, ik, q_pos)
    for t in range(q_len):
        nb = int(q_pos[t]) // BLK + 1
        got = score[:, t, :nb].float()
        want = ref[:, t, :nb]
        assert torch.allclose(got, want, atol=0.35, rtol=0.02), (t, (got - want).abs().max())

    topk_idx = m3_index_topk_prefill(score, cu, pre, q_len, TOPK, INIT, LOCAL)
    ref_sel = ref_select(ref, q_pos, TOPK, INIT, LOCAL)
    for h in range(H_IDX):
        for t in range(q_len):
            nb = int(q_pos[t]) // BLK + 1
            k = min(TOPK, nb)
            got = topk_idx[h, t].tolist()
            assert set(got[:k]) == ref_sel[h][t], (h, t, got, ref_sel[h][t])
            assert all(i == -1 for i in got[k:]), (h, t, got)


@pytest.mark.parametrize("kv_lens", [[BLK * 6 + 17, 70, BLK * 3]])
def test_decode_index_topk(kv_lens: list[int]):
    torch.manual_seed(1)
    bs = len(kv_lens)
    max_nb = max((L + BLK - 1) // BLK for L in kv_lens)
    iq = torch.randn(bs, H_IDX, D_IDX, device=DEV, dtype=torch.bfloat16)
    block_rows = torch.zeros(bs, max_nb, dtype=torch.int32, device=DEV)
    slabs = []
    total = 0
    layouts = []
    for i, L in enumerate(kv_lens):
        nb = (L + BLK - 1) // BLK
        br, pages = make_layout(nb, seed=i)
        layouts.append((br, pages))
        block_rows[i, :nb] = br + total * BLK
        total += pages
    slab = torch.zeros(total * BLK, D_IDX, device=DEV, dtype=torch.bfloat16)
    iks = []
    off = 0
    for i, L in enumerate(kv_lens):
        ik = torch.randn(L, D_IDX, device=DEV, dtype=torch.bfloat16)
        iks.append(ik)
        br, pages = layouts[i]
        sub = scatter_rows(ik, br, pages)
        slab[off * BLK : (off + pages) * BLK] = sub
        off += pages
    seq = torch.tensor(kv_lens, dtype=torch.int32, device=DEV)
    topk_idx = m3_index_decode(iq, slab, block_rows, seq, max_nb, TOPK, INIT, LOCAL)
    for i, L in enumerate(kv_lens):
        q_pos = torch.tensor([L - 1], device=DEV)
        ref = ref_block_scores(iq[i : i + 1], iks[i], q_pos)
        sel = ref_select(ref, q_pos, TOPK, INIT, LOCAL)
        nb = (L + BLK - 1) // BLK
        k = min(TOPK, nb)
        for h in range(H_IDX):
            got = topk_idx[h, i].tolist()
            assert set(got[:k]) == sel[h][0], (i, h, got, sel[h][0])
            assert all(x == -1 for x in got[k:]), (i, h, got)


def _attend_case(kv_len: int, q_len: int, seed: int):
    torch.manual_seed(seed)
    prefix = kv_len - q_len
    nblocks = (kv_len + BLK - 1) // BLK
    q = torch.randn(q_len, HQ, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(kv_len, KVH, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(kv_len, KVH, D, device=DEV, dtype=torch.bfloat16)
    iq = torch.randn(q_len, H_IDX, D_IDX, device=DEV, dtype=torch.bfloat16)
    ik = torch.randn(kv_len, D_IDX, device=DEV, dtype=torch.bfloat16)
    q_pos = torch.arange(prefix, kv_len, device=DEV)
    block_rows, total_pages = make_layout(nblocks, seed=seed)
    return q, k, v, iq, ik, q_pos, block_rows, total_pages, nblocks, prefix


@pytest.mark.parametrize("kv_len,q_len", [(BLK * 7 + 55, 200), (BLK * 2, BLK * 2), (77, 77)])
def test_prefill_attend(kv_len: int, q_len: int):
    q, k, v, iq, ik, q_pos, block_rows, total_pages, nblocks, prefix = _attend_case(
        kv_len, q_len, seed=2
    )
    k_slab = scatter_rows(k, block_rows, total_pages)
    v_slab = scatter_rows(v, block_rows, total_pages)
    ik_slab = scatter_rows(ik, block_rows, total_pages)

    cu = torch.tensor([0, q_len], dtype=torch.int32, device=DEV)
    seq = torch.tensor([kv_len], dtype=torch.int32, device=DEV)
    pre = torch.tensor([prefix], dtype=torch.int32, device=DEV)
    score = m3_index_score_prefill(
        iq, ik_slab, block_rows.view(1, -1), cu, seq, pre, q_len, kv_len
    )
    topk_idx = m3_index_topk_prefill(score, cu, pre, q_len, TOPK, INIT, LOCAL)

    out = torch.empty_like(q)
    m3_sparse_attn_prefill(
        q, k_slab, v_slab, topk_idx, block_rows.view(1, -1), cu, seq, pre,
        q_len, SCALE, out,
    )

    ref_scores = ref_block_scores(iq, ik, q_pos)
    sel = ref_select(ref_scores, q_pos, TOPK, INIT, LOCAL)
    ref = ref_sparse_attend(q, k, v, sel, q_pos, SCALE)
    err = (out.float() - ref).abs().max().item()
    assert err < 3e-2, err


def test_prefill_attend_equals_dense_when_topk_covers():
    """kv_len <= topk * 128 -> the selection covers every visible block, so the
    sparse attend must equal dense causal attention exactly."""
    kv_len = q_len = BLK * TOPK  # 4 blocks, topk 4
    q, k, v, iq, ik, q_pos, block_rows, total_pages, nblocks, prefix = _attend_case(
        kv_len, q_len, seed=3
    )
    k_slab = scatter_rows(k, block_rows, total_pages)
    v_slab = scatter_rows(v, block_rows, total_pages)
    ik_slab = scatter_rows(ik, block_rows, total_pages)
    cu = torch.tensor([0, q_len], dtype=torch.int32, device=DEV)
    seq = torch.tensor([kv_len], dtype=torch.int32, device=DEV)
    pre = torch.tensor([prefix], dtype=torch.int32, device=DEV)
    score = m3_index_score_prefill(
        iq, ik_slab, block_rows.view(1, -1), cu, seq, pre, q_len, kv_len
    )
    topk_idx = m3_index_topk_prefill(score, cu, pre, q_len, TOPK, INIT, LOCAL)
    out = torch.empty_like(q)
    m3_sparse_attn_prefill(
        q, k_slab, v_slab, topk_idx, block_rows.view(1, -1), cu, seq, pre,
        q_len, SCALE, out,
    )
    # dense causal reference
    group = HQ // KVH
    ref = torch.empty(q_len, HQ, D, device=DEV, dtype=torch.float32)
    s_pos = torch.arange(kv_len, device=DEV)
    for h in range(HQ):
        kh = h // group
        logits = (q[:, h].float() @ k[:, kh].float().t()) * SCALE
        logits = logits.masked_fill(s_pos[None, :] > q_pos[:, None], float("-inf"))
        ref[:, h] = torch.softmax(logits, dim=-1) @ v[:, kh].float()
    err = (out.float() - ref).abs().max().item()
    assert err < 3e-2, err


@pytest.mark.parametrize("kv_lens", [[BLK * 6 + 17, 70, BLK * 3, BLK * 9]])
def test_decode_attend(kv_lens: list[int]):
    torch.manual_seed(4)
    bs = len(kv_lens)
    max_nb = max((L + BLK - 1) // BLK for L in kv_lens)
    q = torch.randn(bs, HQ, D, device=DEV, dtype=torch.bfloat16)
    iq = torch.randn(bs, H_IDX, D_IDX, device=DEV, dtype=torch.bfloat16)
    block_rows = torch.zeros(bs, max_nb, dtype=torch.int32, device=DEV)
    total = 0
    layouts = []
    for i, L in enumerate(kv_lens):
        nb = (L + BLK - 1) // BLK
        br, pages = make_layout(nb, seed=10 + i)
        layouts.append((br, pages))
        block_rows[i, :nb] = br + total * BLK
        total += pages
    k_slab = torch.zeros(total * BLK, KVH, D, device=DEV, dtype=torch.bfloat16)
    v_slab = torch.zeros_like(k_slab)
    ik_slab = torch.zeros(total * BLK, D_IDX, device=DEV, dtype=torch.bfloat16)
    ks, vs, iks = [], [], []
    off = 0
    for i, L in enumerate(kv_lens):
        kk = torch.randn(L, KVH, D, device=DEV, dtype=torch.bfloat16)
        vv = torch.randn(L, KVH, D, device=DEV, dtype=torch.bfloat16)
        ik = torch.randn(L, D_IDX, device=DEV, dtype=torch.bfloat16)
        ks.append(kk); vs.append(vv); iks.append(ik)
        br, pages = layouts[i]
        k_slab[off * BLK : (off + pages) * BLK] = scatter_rows(kk, br, pages)
        v_slab[off * BLK : (off + pages) * BLK] = scatter_rows(vv, br, pages)
        ik_slab[off * BLK : (off + pages) * BLK] = scatter_rows(ik, br, pages)
        off += pages
    seq = torch.tensor(kv_lens, dtype=torch.int32, device=DEV)
    topk_idx = m3_index_decode(iq, ik_slab, block_rows, seq, max_nb, TOPK, INIT, LOCAL)
    out = torch.empty_like(q)
    m3_sparse_attn_decode(q, k_slab, v_slab, topk_idx, block_rows, seq, SCALE, out)

    for i, L in enumerate(kv_lens):
        q_pos = torch.tensor([L - 1], device=DEV)
        ref_scores = ref_block_scores(iq[i : i + 1], iks[i], q_pos)
        sel = ref_select(ref_scores, q_pos, TOPK, INIT, LOCAL)
        ref = ref_sparse_attend(q[i : i + 1], ks[i], vs[i], sel, q_pos, SCALE)
        err = (out[i].float() - ref[0]).abs().max().item()
        assert err < 3e-2, (i, err)
