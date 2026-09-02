# Branch notes (CraigStone-Dev mirror)

This fork mirrors the FreeToken work for Qwen3.5/3.6 MoE on 2×24 GB
Blackwell GPUs (Rudi / Bandit). Two feature branches:

## `feat/qwen3-5-tp-support` — TP=2 for Qwen3.5/3.6 MoE

Upstreamed as [FlashML-org/FreeToken#104](https://github.com/FlashML-org/FreeToken/pull/104).
The PR branch lives in `RuixiangMa/FreeToken` (at `3c8b281`); this mirror
carries 6 follow-up commits on top: quantized-TP completion, `lm_head`
sharding fixes, the TP=2 loader regression test, and A/B verification
scripts.

- Removes the `TP=1 only` gates in `qwen3_5_moe` weight loading (bf16,
  block-FP8, and NVFP4 paths).
- Shards attention (column/row-parallel), GDN, quantized expert banks
  (FP8/NVFP4), and `lm_head` (vocab-parallel).
- Partitions CPU-executor cores per TP rank.
- E2E-verified at TP=1/2/4 (offload + fused modes) with TP1/TP2 A/B output
  equivalence.

## `feat/moe-disk-tier` — NVMe-backed MoE experts (stacked on the TP branch)

Adds a third memory tier below RAM: **VRAM ← RAM ← NVMe**.

- `--moe-disk-tier on --expert-ram-experts N` keeps N experts/layer in
  pinned RAM; the rest live in the checkpoint on NVMe.
- Misses are fetched with `preadv` through a per-thread pinned staging ring
  into the RAM bank, feeding the existing GPU LRU.
- Includes the staging-buffer reuse-race fix (`1d989e9`) and the
  rank-CUDA-device fix (`2a505d7`) — the latter only manifests on fast
  NVMe (4.3 GB/s), where preadv can overtake an in-flight H2D copy.
- E2E: Rudi (1.8 GB/s vdisk) + Bandit (4.3 GB/s NVMe) sweeps; verify run
  8604/8604 slot rows clean.

Not yet upstreamed. Rebase onto `main` after #104 merges — upstream PR #112
(per-layer host-bank residency) rewrites the same MoE files
(`offload_cache.py`, `host_banks.py`, `expert_banks.py`, `args.py`), so
expect merge work plus a full E2E re-verify.
