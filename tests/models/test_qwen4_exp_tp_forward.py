"""Forward equivalence: qwen4_exp (Qwen3.8) TP=1 vs TP=2, same weights, same prefill.

The shape test (test_qwen4_exp_tp.py) proves the loader emits the right tensor SHAPES; it
cannot catch split-order / shard-order bugs. This script loads the same synthetic checkpoint
through ``iter_weights`` at TP=1 and TP=2, runs an identical 48-token prefill on each, and
compares the final-position logits plus every layer's hidden state.

Run (2-GPU box, from the repo root):
    torchrun --nproc_per_node=1 tests/models/test_qwen4_exp_tp_forward.py --tp 1 --out /tmp/q4tp1.pt
    torchrun --nproc_per_node=2 tests/models/test_qwen4_exp_tp_forward.py --tp 2 --out /tmp/q4tp2.pt
    python tests/models/test_qwen4_exp_tp_forward.py --compare /tmp/q4tp1.pt /tmp/q4tp2.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # repo root, for `tests`

HIDDEN = 256
VOCAB = 128
NUM_LAYERS = 4
SEQ_LEN = 48
PAGE_SIZE = 64
SEED = 1234


def _tiny_hf_config():
    """TP=2-divisible geometry; GDN head dims 32 (the fla kernels reject 8).
    Q4TP_LAYERS=full,full,full,full overrides the layer types (QSA isolation)."""
    lt = os.environ.get("Q4TP_LAYERS")
    layer_types = lt.split(",") if lt else [
        "linear_attention", "full_attention",
        "linear_attention", "full_attention",
    ]
    ple_ids = [] if os.environ.get("Q4TP_PLE") == "0" else [1]
    return SimpleNamespace(
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=SimpleNamespace(
            hidden_size=HIDDEN,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            num_hidden_layers=NUM_LAYERS,
            layer_types=layer_types,
            vocab_size=VOCAB,
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
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            output_gate_type="sigmoid",
            hc_count=2,
            hc_lowrank=16,
            # 1-indexed: layer 0 (a linear_attention layer) carries the PLE.
            ple_layer_ids=ple_ids,
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
            indexer_head_dim=64,
            indexer_budget=8,
            indexer_compress_ratio=2,
        ),
    )


def _make_ckpt(folder: str) -> None:
    """One-shard checkpoint in the pre-fusion key format, deterministic, PLE table included."""
    from safetensors.torch import save_file

    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.models.qwen4_exp.ple import derive_ngram_hash_constants

    torch.manual_seed(SEED)
    cfg = parse_config(_tiny_hf_config())
    H = cfg.hidden_size
    qo = cfg.num_qo_heads * cfg.head_dim
    kv = cfg.num_kv_heads * cfg.head_dim
    g = cfg.linear_attention_group()
    key_dim = g.num_key_heads * g.key_head_dim
    value_dim = g.num_value_heads * g.value_head_dim
    conv_dim = 2 * key_dim + value_dim
    inter = cfg.shared_expert_intermediate_size
    hc = cfg.qwen4_args.hc_count
    lowrank = cfg.qwen4_args.hc_lowrank
    ple_dim = cfg.qwen4_args.ple_embed_dim
    num_experts = cfg.num_experts
    moe_inter = cfg.moe_intermediate_size
    a = cfg.qwen4_args

    t: dict[str, torch.Tensor] = {}
    P = "model.language_model."
    t[P + "embed_tokens.weight"] = torch.randn(VOCAB, H)
    t["lm_head.weight"] = torch.randn(VOCAB, H)
    t[P + "hyper_connection_mixer.hc_norm.weight"] = torch.randn(hc * H)
    t[P + "hyper_connection_mixer.input_mix_weight_down.weight"] = torch.randn(lowrank, hc * H)
    t[P + "hyper_connection_mixer.input_mix_weight_up.weight"] = torch.randn(hc * H, lowrank)
    for i in range(NUM_LAYERS):
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
            t[lp + "self_attn.indexer.index_qk_proj.weight"] = torch.randn(
                (a.index_n_heads + a.index_kv_heads) * a.index_head_dim, H)
            t[lp + "self_attn.indexer.q_layernorm.weight"] = torch.randn(a.index_head_dim)
            t[lp + "self_attn.indexer.k_layernorm.weight"] = torch.randn(a.index_head_dim)
        t[lp + "mlp.gate.weight"] = torch.randn(num_experts, H)
        t[lp + "mlp.shared_expert_gate.weight"] = torch.randn(1, H)
        t[lp + "mlp.shared_expert.gate_proj.weight"] = torch.randn(inter, H)
        t[lp + "mlp.shared_expert.up_proj.weight"] = torch.randn(inter, H)
        t[lp + "mlp.shared_expert.down_proj.weight"] = torch.randn(H, inter)
        for hc_name in ("attn_hyper_connection", "mlp_hyper_connection"):
            t[lp + f"{hc_name}.hc_norm.weight"] = torch.randn(hc * H)
            t[lp + f"{hc_name}.input_mix_weight_down.weight"] = torch.randn(lowrank, hc * H)
            t[lp + f"{hc_name}.block_inject_weight.weight"] = torch.randn(hc, hc * H)
            t[lp + f"{hc_name}.input_mix_weight_up.weight"] = torch.randn(hc * H, lowrank)
        if i in cfg.qwen4_args.ple_layer_ids:
            mult, sizes, offsets = derive_ngram_hash_constants(
                vocab_size=VOCAB, ngram_size=a.ngram_size,
                num_ngram_heads=a.num_ngram_heads,
                ngram_vocab_size_base=a.ngram_vocab_size_base, ple_layer_index=0)
            total = sum(sizes)
            head_dim = a.ngram_head_dim
            per = total // a.split_ngram_parts
            assert per * a.split_ngram_parts == total, "table rows must split evenly"
            table = torch.randn(total, head_dim).to(torch.float8_e4m3fn)
            for s in range(a.split_ngram_parts):
                t[lp + f"ple.ple_embedding.ngram_embedding.shard_{s}.weight"] = \
                    table[s * per:(s + 1) * per].contiguous()
            t[lp + "ple.ple_embedding.ngram_embedding.weight_scale"] = torch.tensor(0.5)
            t[lp + "ple.ple_embedding.layer_multipliers"] = torch.tensor(mult, dtype=torch.int64)
            t[lp + "ple.ple_embedding.ngram_heads_vocab_sizes"] = torch.tensor(sizes, dtype=torch.int64)
            t[lp + "ple.ple_embedding.ngram_heads_offsets"] = torch.tensor(offsets, dtype=torch.int64)
            t[lp + "ple.key_proj.weight"] = torch.randn(hc * H, ple_dim)
            t[lp + "ple.value_proj.weight"] = torch.randn(H, ple_dim)
            t[lp + "ple.norm_key.weight"] = torch.randn(hc * H)
            t[lp + "ple.norm_query.weight"] = torch.randn(hc * H)
            t[lp + "ple.norm_conv.weight"] = torch.randn(hc * H)
            t[lp + "ple.conv1d.weight"] = torch.randn(hc * H, 1, a.ple_conv_kernel_size)
    os.makedirs(folder, exist_ok=True)
    save_file(t, os.path.join(folder, "model.safetensors"))
    hf = _tiny_hf_config()
    with open(os.path.join(folder, "config.json"), "w") as fh:
        json.dump(vars(hf), fh, default=lambda o: vars(o))


def _stable_seed(name: str) -> int:
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)


def run(tp: int, out: str | None) -> None:
    # single-process fallback when not launched by torchrun
    if "RANK" not in os.environ:
        os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0",
                          MASTER_ADDR="127.0.0.1", MASTER_PORT="29517")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    size = dist.get_world_size()
    assert size == tp, f"world size {size} != --tp {tp}"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    from freetoken.core import Batch, Context, Req, set_global_ctx
    from freetoken.distributed.info import DistributedInfo
    from freetoken.kvcache import create_kvcache_pool
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.attention.linear import build_fla_metadata
    from freetoken.attention.qsa_sparse import QSASparseAttnBackend
    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM
    from freetoken.models.qwen4_exp.ple import PinnedUVATable
    from freetoken.models.qwen4_exp.weight import iter_weights

    import freetoken.distributed.info as _tpinfo
    _tpinfo._TP_INFO = DistributedInfo(rank, size)

    cfg = parse_config(_tiny_hf_config())

    ckpt = os.environ.get("Q4TP_CKPT")
    assert ckpt, "set Q4TP_CKPT to the synthetic checkpoint dir"

    # ---- model: meta init (as the engine), loader weights, deterministic fill of the routed experts ----
    from freetoken.layers.rotary import set_rope_device
    from freetoken.utils.torch_utils import torch_dtype

    set_rope_device(device)
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = Qwen4ExpForCausalLM(cfg)
    state = {}
    for key, w in iter_weights(ckpt, device, include_moe_experts=False, include_non_moe=True):
        state[key] = w
    want = model.state_dict()
    full: dict[str, torch.Tensor] = {}
    for name, meta in want.items():
        if name in state:
            full[name] = state[name].to(device=device, dtype=meta.dtype)
        elif meta.dtype.is_floating_point:
            # Routed experts: the resident MoE is TP-sharded (col-parallel gate_up on the
            # intermediate dim, row-parallel down_proj). Fill from a FULL-size deterministic
            # tensor and slice per rank so TP=1 and TP=2 hold the same logical weights.
            gen = torch.Generator(device=device).manual_seed(_stable_seed(name))
            if name.endswith("experts.gate_up_proj"):
                # per-expert layout is [gate | up]; each half is col-parallel independently
                inter_local = meta.shape[1] // 2
                fg = torch.randn((meta.shape[0], inter_local * size, meta.shape[2]),
                                 generator=gen, device=device, dtype=meta.dtype)
                fu = torch.randn((meta.shape[0], inter_local * size, meta.shape[2]),
                                 generator=gen, device=device, dtype=meta.dtype)
                sl = slice(rank * inter_local, (rank + 1) * inter_local)
                full[name] = torch.cat([fg[:, sl, :], fu[:, sl, :]], dim=1)
            elif name.endswith("experts.down_proj"):
                f = torch.randn((meta.shape[0], meta.shape[1], meta.shape[2] * size),
                                 generator=gen, device=device, dtype=meta.dtype)
                full[name] = f[:, :, rank * meta.shape[2]:(rank + 1) * meta.shape[2]].contiguous()
            else:
                full[name] = torch.randn(meta.shape, generator=gen, device=device, dtype=meta.dtype)
        else:
            full[name] = torch.zeros(meta.shape, device=device, dtype=meta.dtype)
    extra = set(state) - set(want)
    assert not extra, f"unexpected loader keys: {sorted(extra)}"
    model.load_state_dict(full)

    # ---- PLE table: read the shards straight (skip load_ple_table's O_DIRECT: overlayfs) ----
    if model.model.ple_layers:
        from safetensors import safe_open

        with safe_open(os.path.join(ckpt, "model.safetensors"), framework="pt", device="cpu") as f:
            keys = sorted(k for k in f.keys() if "ngram_embedding.shard_" in k)
            table_tensor = torch.cat([f.get_tensor(k) for k in keys], dim=0).contiguous()
            scale_key = next(k for k in f.keys() if k.endswith("ngram_embedding.weight_scale"))
            table_scale = float(f.get_tensor(scale_key))
        table_tensor = table_tensor.pin_memory()
        for ple in model.model.ple_layers:
            ple.ple_embedding.attach_table(PinnedUVATable(table_tensor, table_scale))

    # ---- pools + context ----
    num_req_slots = 2  # 1 request + 1 dummy
    pool = create_kvcache_pool(
        model_config=cfg, num_pages=3,  # 2 usable + 1 dummy, as create_kv_pool does
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16, device=device, num_req_slots=num_req_slots)
    page_table = torch.zeros((num_req_slots, 2 * PAGE_SIZE), dtype=torch.int32, device=device)
    page_table[0, :2 * PAGE_SIZE] = torch.arange(2 * PAGE_SIZE, dtype=torch.int32, device=device)
    page_table[1].fill_(2 * PAGE_SIZE)  # dummy page
    pool.attach_page_table(page_table)
    lpool = LinearStatePool(
        group=cfg.linear_attention_group(), num_slots=num_req_slots,
        dtype=torch.bfloat16, device=device, tp_size=size, slot_states=cfg.slot_states)
    ctx = Context(page_size=PAGE_SIZE)
    ctx.page_table = page_table
    ctx.kv_cache = pool
    ctx.linear_state_pool = lpool
    set_global_ctx(ctx)  # the backend reads the pool off the ctx at construction
    from freetoken.moe import create_moe_backend

    ctx.moe_backend = create_moe_backend(cfg.moe_backend)
    ctx.attn_backend = QSASparseAttnBackend(cfg)

    # ---- one prefill request ----
    gen = torch.Generator().manual_seed(7)
    ids = torch.randint(1, VOCAB, (SEQ_LEN,), generator=gen, dtype=torch.int32)
    req = Req(input_ids=ids, table_idx=0, cached_len=0, output_len=1,
              uid=0, sampling_params=None, cache_handle=None)
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = [req]
    batch.input_ids = ids.to(device)
    batch.positions = torch.arange(SEQ_LEN, dtype=torch.int32, device=device)
    batch.out_loc = page_table[0, :SEQ_LEN]
    batch.linear_table_idx = torch.zeros(SEQ_LEN, dtype=torch.int32, device=device)
    batch.fla_metadata = build_fla_metadata(batch, device)
    ctx.attn_backend.prepare_metadata(batch)

    # ---- forward, capturing per-layer hidden states ----
    layer_outs: dict[str, torch.Tensor] = {}
    wrapped = []
    def _capture(op, name, method="forward"):
        orig = getattr(op, method)

        def _f(*a, **kw):
            out = orig(*a, **kw)
            t = out[0] if isinstance(out, tuple) else out
            layer_outs[name] = t.detach().cpu()
            return out
        setattr(op, method, _f)
        wrapped.append((op, method, orig))

    _capture(model.model.embed_tokens, "emb")
    for i, layer in enumerate(model.model.layers.op_list):
        if layer.ple is not None:
            _capture(layer.ple, f"layer{i}.ple")
        _capture(layer.attn_hyper_connection, f"layer{i}.attn_mix", "mix")
        _capture(layer.linear_attn if layer._is_linear else layer.self_attn, f"layer{i}.attn")
        _capture(layer.attn_hyper_connection, f"layer{i}.attn_comb", "combine")
        _capture(layer.mlp_hyper_connection, f"layer{i}.mlp_mix", "mix")
        _capture(layer.mlp.gate, f"layer{i}.router")
        _capture(layer.mlp.shared_expert, f"layer{i}.shared")
        _capture(layer.mlp.experts, f"layer{i}.routed")
        _capture(layer.mlp, f"layer{i}.mlp")
        _capture(layer, f"layer{i}")
    with torch.no_grad(), ctx.forward_batch(batch):
        logits = model.forward()
    for op, method, orig in wrapped:
        setattr(op, method, orig)

    # layer0's routed-expert weights, FULL (unsharded), for the fp64 CPU MoE cross-check
    gu = model.state_dict()["model.layers.0.mlp.experts.gate_up_proj"]
    dp = model.state_dict()["model.layers.0.mlp.experts.down_proj"]
    if size > 1:
        inter_local = gu.shape[1] // 2
        gu_parts = [torch.empty_like(gu) for _ in range(size)]
        dist.all_gather(gu_parts, gu.contiguous())
        gu = torch.cat([p[:, :inter_local] for p in gu_parts] +
                       [p[:, inter_local:] for p in gu_parts], dim=1)
        dp_parts = [torch.empty_like(dp) for _ in range(size)]
        dist.all_gather(dp_parts, dp.contiguous())
        dp = torch.cat(dp_parts, dim=2)
    result = {
        "logits_last": logits[-1].float().cpu(),
        "logits_all": logits.float().cpu(),
        "layers": layer_outs,
        "experts_gate_up": gu.cpu(),
        "experts_down": dp.cpu(),
    }
    if rank == 0 and out:
        torch.save(result, out)
        print(f"[tp={tp}] saved {out}: logits {tuple(logits.shape)}, "
              f"layers {sorted(layer_outs)}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def compare(a_path: str, b_path: str) -> None:
    a = torch.load(a_path, map_location="cpu")
    b = torch.load(b_path, map_location="cpu")
    ok = True
    for key in ("logits_last", "logits_all"):
        d = (a[key] - b[key]).abs()
        scale = a[key].abs().max().item()
        print(f"{key}: max_abs={d.max().item():.4g} scale={scale:.4g} "
              f"rel={d.max().item() / max(scale, 1e-6):.4g}")
        if d.max().item() > 0.05 * max(scale, 1.0):
            ok = False
    for name in sorted(a["layers"]):
        da, db = a["layers"][name], b["layers"][name]
        # per-layer hidden is [T, hc*hidden]; compare the final position
        d = (da[-1] - db[-1]).abs()
        scale = da[-1].abs().max().item()
        print(f"{name}: max_abs={d.max().item():.4g} scale={scale:.4g} "
              f"rel={d.max().item() / max(scale, 1e-6):.4g}")
        if d.max().item() > 0.05 * max(scale, 1.0):
            ok = False
    # MoE router top-k flips: with random weights the top-2 logit gap is often within
    # bf16 noise, so a flip changes the selected experts and the MoE output discontinuously.
    for name in sorted(k for k in a["layers"] if k.endswith(".router")):
        ra, rb = a["layers"][name], b["layers"][name]
        ia = ra.topk(2, dim=-1).indices.sort(-1).values
        ib = rb.topk(2, dim=-1).indices.sort(-1).values
        flips = (ia != ib).any(-1).sum().item()
        gap = (ra.topk(2, dim=-1).values[:, -2] - ra.topk(2, dim=-1).values[:, -1]).abs()
        print(f"{name}: top2 flips={flips}/{ra.shape[0]} min_gap={gap.min().item():.4g}")
    print("EQUIVALENT" if ok else "DIVERGED")
    sys.exit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--compare", nargs=2, metavar=("TP1", "TP2"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    assert args.tp is not None, "--tp required unless --compare"
    run(args.tp, args.out)


if __name__ == "__main__":
    main()
