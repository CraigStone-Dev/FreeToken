"""Qwen3.5 / 3.6 / 3.8 checkpoint reader.

The dense pass reads every Linear module under the scheme the checkpoint's QuantConfig gives it, the same answer the model built its buffers from, so bf16, block-fp8, ModelOpt and llm-compressor exports in any mix all land as the model's state dict. Routed experts are read by the offload cache (``nvfp4_expert_spec`` / ``iter_expert_pieces``); bf16 stacked experts, resident block-fp8 experts and resident NVFP4 experts come from here.

Under TP>1 every tensor the reader emits is already the rank's slice of the model's TP-local buffers: column-merged projections shard per part on the output dim (KV heads replicate when num_kv < tp), row-parallel projections shard the input dim, embed/lm_head the vocab, GDN per-head vectors and conv1d per sub-part, and the resident expert banks the intermediate dim.
"""

from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4
from freetoken.layers.quantization import QuantConfig, QuantKind, QuantScheme, get_quant_config
from freetoken.models.config import LinearGatedDeltaGroupConfig, VISION_KEY_PREFIXES
from freetoken.models.loader import ShardReader, iter_weight_files
from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec
from freetoken.models.register import ModelSpec, get_model_spec
from freetoken.utils import cached_load_hf_config, div_ceil, div_even
from tqdm import tqdm

from .config import parse_config
from freetoken.models.qwen3_vl.weight import rename_vl_prefix

# bf16 checkpoints store the routed experts pre-stacked per layer
_STACKED_EXPERT_RE = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)$")
# per-expert tensors of a quantized checkpoint: the offload cache's expert reader takes these
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
# the ``model.language_model.`` anchor excludes the MTP head's ``mtp.layers.N.mlp.experts.*``
_EXPERT_KEY_RE = (
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>{kinds})$"
)
# role -> the expert bank reader's canonical (ModelOpt) tensor kind
_BANK_KINDS = {"weight": "weight", "weight_scale": "weight_scale", "weight_global": "weight_scale_2"}

# Gemma-style (1+weight) RMSNorm weights; the GDN gated norm (linear_attn.norm) is a plain weight*x norm
_GEMMA_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
)
# leaves the model builds as Linear layers: only their tensors are read under the QuantConfig, the rest passes through as stored
_LINEAR_LEAVES = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    "gate_proj", "up_proj", "down_proj", "gate", "shared_expert_gate", "lm_head",
})
# activation scales of modules whose scheme carries no input_scale role
_DROPPED_SUFFIXES = frozenset({"input_scale", "input_global_scale"})
_ELEM_DTYPES = {"e4m3": torch.float8_e4m3fn, "e2m1": torch.uint8}
_QUANT_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8, torch.int8)


# ======================================================================================
# TP sharding (TP>1): every tensor the reader emits is the rank's slice of the model's
# TP-local buffers. Column-merged projections shard per part on the output dim (the GDN
# recurrence needs whole heads, so the in_proj qkv part splits into q | k | v first and KV
# heads replicate when num_kv < tp); row-parallel projections shard the input dim;
# embed/lm_head the vocab; per-head vectors and conv1d per sub-part.
# ======================================================================================

def _shard_tp(tensor: torch.Tensor, *, rank: int, world_size: int, dim: int) -> torch.Tensor:
    """Simple chunk sharding along ``dim``."""
    if world_size == 1:
        return tensor
    return tensor.chunk(world_size, dim=dim)[rank].clone()


def _shard_tp_parts(
    tensor: torch.Tensor,
    part_sizes: tuple[int, ...],
    *,
    rank: int,
    world_size: int,
    local_part_sizes: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Per-part column-parallel sharding (dim=0). Each part is sharded independently,
    matching ``LinearColParallelMerged``'s per-output-size sharding.

    When ``local_part_sizes`` is provided, parts where ``local < full`` but
    ``local * world != full`` are replicated by head index (``rank * heads // world``),
    for GQA KV head replication when ``num_kv < tp_size``."""
    if world_size == 1:
        return tensor
    shards: list[torch.Tensor] = []
    offset = 0
    for i, full_size in enumerate(part_sizes):
        chunk = tensor[offset:offset + full_size]
        local_size = full_size // world_size if local_part_sizes is None else local_part_sizes[i]
        if local_size >= full_size:
            shards.append(chunk.clone())
        elif local_size * world_size == full_size:
            shards.append(chunk.chunk(world_size, dim=0)[rank].clone())
        else:
            num_heads = full_size // local_size
            head_idx = rank * num_heads // world_size
            shards.append(chunk[head_idx * local_size:(head_idx + 1) * local_size].clone())
        offset += full_size
    return torch.cat(shards, dim=0)


def _linear_group(config) -> LinearGatedDeltaGroupConfig | None:
    """The GDN attention group (num_key_heads / num_value_heads / head dims), or None."""
    for group in config.attention_groups:
        if isinstance(group, LinearGatedDeltaGroupConfig):
            return group
    return None


def _shard_dense_part(base: str, tensor: torch.Tensor, config) -> torch.Tensor:
    """TP-shard one dense-projection part along the output dim (dim=0, per head), for the
    column-merged linears (attention q/k/v, GDN in_proj qkv|z|b|a, gate/up). ``base`` is the
    checkpoint part name sans ``.weight``; the same call shards a part's weight or any of its
    per-output-row scale roles (weight_scale_inv at //128 rows, weight_scale at //16). The GDN
    in_proj_qkv part is split into its q | k | v sub-parts first so each shards independently
    (the recurrence needs whole heads); KV heads replicate when num_kv < tp. Row-parallel /
    vocab-parallel parts are handled by ``_DenseReader._shard_part``, not here."""
    tp = get_tp_info()
    if tp.size == 1:
        return tensor
    rank, world = tp.rank, tp.size
    if base.endswith(".k_proj") or base.endswith(".v_proj"):
        num_kv = config.num_kv_heads
        local_kv = div_even(num_kv, world, allow_replicate=True)
        if local_kv * world == num_kv:
            return _shard_tp(tensor, rank=rank, world_size=world, dim=0)
        rows_per_head = tensor.shape[0] // num_kv
        head_idx = rank * num_kv // world
        return tensor[head_idx * rows_per_head:(head_idx + 1) * rows_per_head].clone()
    if base.endswith(".in_proj_qkv"):
        g = _linear_group(config)
        key_dim = g.num_key_heads * g.key_head_dim
        value_dim = g.num_value_heads * g.value_head_dim
        parts = (key_dim, key_dim, value_dim)
        if tensor.shape[0] != sum(parts):
            parts = tuple(p // 128 for p in parts)  # weight_scale_inv
        q, k, v = (tensor[:parts[0]], tensor[parts[0]:parts[0] + parts[1]],
                   tensor[parts[0] + parts[1]:])
        num_kv = g.num_key_heads
        local_kv = div_even(num_kv, world, allow_replicate=True)
        if local_kv * world == num_kv:
            k = _shard_tp(k, rank=rank, world_size=world, dim=0)
        else:
            rows_per_head = k.shape[0] // num_kv
            head_idx = rank * num_kv // world
            k = k[head_idx * rows_per_head:(head_idx + 1) * rows_per_head].clone()
        return torch.cat([
            _shard_tp(q, rank=rank, world_size=world, dim=0),
            k,
            _shard_tp(v, rank=rank, world_size=world, dim=0),
        ], dim=0)
    return _shard_tp(tensor, rank=rank, world_size=world, dim=0)


def _shard_vocab(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Vocab-parallel shard matching ``VocabParallelEmbedding``'s div_ceil ranges (the last
    rank may hold fewer rows); every per-vocab-row role (weight / scale / global) slices alike."""
    if world_size == 1 or tensor.ndim == 0:
        return tensor
    per = div_ceil(tensor.shape[0], world_size)
    start = per * rank
    return tensor[start:min(start + per, tensor.shape[0])].clone()


def _maybe_shard(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Apply TP sharding to a non-fused (passthrough) weight based on its state-dict key
    suffix. Fused projections are sharded per part by the dense reader; conv1d.weight is also
    handled at the yield site (it needs the sub-part sizes)."""
    tp = get_tp_info()
    if tp.size == 1:
        return tensor
    if name.endswith((".o_proj.weight", ".down_proj.weight", ".out_proj.weight")):
        return _shard_tp(tensor, rank=tp.rank, world_size=tp.size, dim=1)
    if name.endswith(("embed_tokens.weight", "lm_head.weight")):
        return _shard_vocab(tensor, tp.rank, tp.size)
    if name.endswith(("A_log", "dt_bias")):
        return _shard_tp(tensor, rank=tp.rank, world_size=tp.size, dim=0)
    # Routed expert weights (3D stacked: [num_experts, ...]) — resident mode only.
    # gate_up_proj is [num_experts, 2*intermediate, hidden] = [gate, up] fused on dim=1.
    # Must shard gate and up independently (like _shard_tp_parts on dim=1).
    # down_proj is [num_experts, hidden, intermediate] — single dim, simple chunk on dim=2.
    if name.endswith("experts.gate_up_proj"):
        intermediate = tensor.shape[1] // 2
        gate = tensor[:, :intermediate, :]
        up = tensor[:, intermediate:, :]
        return torch.cat([
            gate.chunk(tp.size, dim=1)[tp.rank].clone(),
            up.chunk(tp.size, dim=1)[tp.rank].clone(),
        ], dim=1)
    if name.endswith("experts.down_proj"):
        return _shard_tp(tensor, rank=tp.rank, world_size=tp.size, dim=2)
    return tensor


def _rename(raw_name: str) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith("mtp."):
        return None
    # static KV-cache scales of the quantizers; the KV cache runs in the engine's dtype
    if raw_name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
        return None
    return rename_vl_prefix(raw_name)


def _is_gemma_norm(name: str) -> bool:
    return name == "model.norm.weight" or name.endswith(_GEMMA_NORM_SUFFIXES)


def _per_row_scale(scale: torch.Tensor, rows: int) -> torch.Tensor:
    """Per-tensor scalar or per-channel ``[rows, 1]`` fp8 scale -> fp32 ``[rows]``; any other count is refused rather than broadcast onto the wrong rows."""
    flat = scale.reshape(-1).to(torch.float32)
    if flat.numel() == 1:
        return flat.expand(rows).contiguous()
    if flat.numel() != rows:
        raise ValueError(
            f"fp8 weight_scale has {flat.numel()} elements for a weight with {rows} output rows "
            f"(shape {tuple(scale.shape)}); expected 1 or {rows}"
        )
    return flat.contiguous()


def _dequant_nvfp4(weight: torch.Tensor, weight_scale: torch.Tensor, weight_global: torch.Tensor) -> torch.Tensor:
    """Packed NVFP4 -> bf16 on CUDA (the kernel is GPU-only, the converter reads on CPU), returned on the caller's device."""
    device = weight.device
    if device.type != "cuda":
        weight, weight_scale, weight_global = (t.to("cuda") for t in (weight, weight_scale, weight_global))
    slots = torch.zeros(1, dtype=torch.int32, device=weight.device)
    out = dequant_nvfp4(
        weight.unsqueeze(0).contiguous(), weight_scale.unsqueeze(0).contiguous(), weight_global.unsqueeze(0),
        slots, dtype=torch.bfloat16,
    )[0]
    return out.to(device)


def _dequant(scheme: QuantScheme, part: dict[str, torch.Tensor]) -> torch.Tensor:
    """bf16 weight of a module the checkpoint quantized but the family serves unquantized."""
    weight = part["weight"]
    if scheme.kind is QuantKind.FP8_TENSOR:
        return (weight.to(torch.float32) * part["weight_scale"][:, None]).to(torch.bfloat16)
    if scheme.kind is QuantKind.FP8_BLOCK:
        from freetoken.kernel.triton.fp8_block_linear import dequant_block_fp8

        return dequant_block_fp8(weight, part["weight_scale_inv"])
    if scheme.kind is QuantKind.NVFP4:
        return _dequant_nvfp4(weight, part["weight_scale"], part["weight_global"])
    raise NotImplementedError(f"no bf16 dequantization for {scheme}")


class _DenseReader:
    """Routes each Linear tensor to the buffer its module's scheme declares; packed projections are concatenated per role once every part is in.

    Under TP>1 each part is sharded to the rank's slice before the scales are validated and
    the parts concatenated (``_shard_part``), so the emitted buffers are the model's TP-local
    ones: column-merged per part on the output dim, row-parallel on the input dim, lm_head on
    the vocab."""

    def __init__(self, quant: QuantConfig | None, spec: ModelSpec, config) -> None:
        self.quant = quant
        self.config = config
        self.groups = {fused: parts for fused, parts in spec.packed_modules_mapping if fused != "experts"}
        self.by_part: dict[str, list[tuple[str, int]]] = {}
        for fused, parts in self.groups.items():
            for idx, part in enumerate(parts):
                self.by_part.setdefault(part, []).append((fused, idx))
        # target module -> (part count, {part: {role: tensor}}, {part: the roles its module stores})
        self.pending: dict[str, tuple[int, dict[int, dict[str, torch.Tensor]], dict[int, set[str]], QuantScheme | None]] = {}

    def scheme(self, module: str) -> QuantScheme | None:
        return None if self.quant is None else self.quant.scheme_for(module)

    def stored(self, module: str) -> QuantScheme | None:
        """The scheme the checkpoint stores ``module`` under, before the family's unquantized_modules."""
        if self.quant is None:
            return None
        return self.quant.scheme_for_name(self.quant.name_map.to_checkpoint(module)[0])

    def target(self, module: str) -> tuple[str, int, int]:
        """``(fused module, part index, part count)``; a standalone linear is its own single-part target."""
        parent, _, leaf = module.rpartition(".")
        candidates = self.by_part.get(leaf)
        if not candidates:
            return module, 0, 1
        if len(candidates) > 1:
            # GDN: quantized checkpoints split qkv|z from the bf16 b|a; same test as gdn.py
            split = self.scheme(f"{parent}.in_proj_qkvz") is not None
            keep = {"in_proj_qkvz", "in_proj_ba"} if split else {"in_proj"}
            candidates = [c for c in candidates if c[0] in keep]
        fused, idx = candidates[0]
        return f"{parent}.{fused}", idx, len(self.groups[fused])

    def add(self, name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]] | None:
        """Take one tensor; the emitted ``[(name, tensor)]`` once its target module is complete, ``[]`` before, None if ``name`` is not a Linear's tensor."""
        module, _, suffix = name.rpartition(".")
        if module.rpartition(".")[2] not in _LINEAR_LEAVES:
            return None
        stored = self.stored(module)
        roles = {"weight": "weight"} if stored is None else {e.name: r for r, e in self.quant.storage(stored).items()}
        role = roles.get(suffix)
        if role is None:
            if suffix in _DROPPED_SUFFIXES:
                return []
            raise ValueError(
                f"{name}: the checkpoint's quant config declares {module} {stored or 'unquantized'}, stored as {sorted(roles)}"
            )
        if stored is None and tensor.dtype in _QUANT_DTYPES:
            raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} unquantized")
        if stored is not None and role == "weight" and tensor.dtype is not _ELEM_DTYPES[stored.weight.elem]:
            raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} {stored}")
        target, idx, count = self.target(module)
        _, parts, expected, _ = self.pending.setdefault(target, (count, {}, {}, stored))
        parts.setdefault(idx, {})[role] = tensor
        expected[idx] = set(roles.values())
        if len(parts) < count or any(set(parts[i]) != expected[i] for i in parts):
            return []
        del self.pending[target]
        return self._emit(target, [parts[i] for i in range(count)], stored)

    def missing(self) -> list[str]:
        """One line per incomplete module: the roles its parts still lack."""
        lines = []
        for target, (count, parts, expected, stored) in sorted(self.pending.items()):
            lacking = sorted(set().union(*(expected[i] - set(parts[i]) for i in parts)))
            note = ""
            if lacking == ["input_scale"]:
                fix = "declares W4A16_NVFP4 or sets with_input_scale false" if stored is not None and stored.kind is QuantKind.NVFP4 else "sets with_input_scale false"
                note = f" (an export without activation scales {fix})"
            if len(parts) < count:
                lacking.append(f"{count - len(parts)} of {count} fused parts")
            lines.append(f"{target}: missing {lacking}{note}")
        return lines

    def _part_leaf(self, target: str, idx: int) -> str:
        parts = self.groups.get(target)
        return parts[idx] if parts is not None else target.rpartition(".")[2]

    def _shard_part(self, target: str, idx: int, part: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """TP-shard one fused part's role tensors to the model's TP-local buffers:
        column-merged parts shard the output dim per head (KV heads replicate when
        num_kv < tp), row-parallel parts shard the input dim (per-row globals and the
        activation scale stay full), lm_head the vocab (div_ceil ranges)."""
        tp = get_tp_info()
        rank, world = tp.rank, tp.size
        if target.endswith(("gate", "shared_expert_gate")):
            return part  # router gates are LinearReplicated
        if target.endswith(("o_proj", "out_proj", "down_proj")):
            # 0-dim per-tensor scales stay as-is; _check expands them to the sharded rows
            return {
                role: _shard_tp(t, rank=rank, world_size=world, dim=1)
                if role in ("weight", "weight_scale_inv", "weight_scale") and t.ndim > 1 else t
                for role, t in part.items()
            }
        if target.endswith("lm_head"):
            return {role: _shard_vocab(t, rank, world) for role, t in part.items()}
        parent, _, _ = target.rpartition(".")
        base = f"{parent}.{self._part_leaf(target, idx)}"
        return {role: _shard_dense_part(base, t, self.config) if t.ndim > 0 else t
                for role, t in part.items()}

    def _emit(self, target: str, parts: list[dict[str, torch.Tensor]], stored: QuantScheme | None):
        if get_tp_info().size > 1:
            parts = [self._shard_part(target, idx, part) for idx, part in enumerate(parts)]
        if stored is not None:
            parts = [self._check(target, stored, part) for part in parts]
            if self.scheme(target) is None:
                parts = [{"weight": _dequant(stored, part)} for part in parts]
        out = []
        for role in parts[0]:
            tensors = [part[role] for part in parts]
            if role == "input_scale":
                # fused parts read the same activation, so ModelOpt calibrates one range for them: max is exact then and safe if they drift
                value = torch.stack(tensors).max()
            else:
                value = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
            out.append((f"{target}.{role}", value))
        return out

    def _check(self, target: str, scheme: QuantScheme, part: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Validate one part against ``scheme`` and put its scales in the layer's form."""
        part = {
            role: 1.0 / tensor.to(torch.float32) if self.quant.storage(scheme)[role].reciprocal else tensor
            for role, tensor in part.items()
        }
        weight = part["weight"]
        if weight.dtype is not _ELEM_DTYPES[scheme.weight.elem]:
            raise ValueError(f"{target}: weight is {weight.dtype} but the checkpoint's quant config declares {scheme}")
        rows, cols = weight.shape[0], weight.shape[1] * (2 if scheme.weight.elem == "e2m1" else 1)
        block_rows, block_cols = scheme.weight.group or (1, 1)
        scale_role = "weight_scale_inv" if "weight_scale_inv" in part else "weight_scale"
        out = dict(part)
        if block_cols < 0:
            out[scale_role] = _per_row_scale(part[scale_role], rows)
        else:
            if rows % block_rows or cols % block_cols:
                raise ValueError(f"{target}: {rows}x{cols} weight is not a multiple of the {block_rows}x{block_cols} scale block of {scheme}")
            expected = (rows // block_rows, cols // block_cols)
            if tuple(part[scale_role].shape) != expected:
                raise ValueError(f"{target}: {scale_role} is {tuple(part[scale_role].shape)}, expected {expected} for {scheme}")
            if scheme.weight.scale == "e4m3" and part[scale_role].dtype is not torch.float8_e4m3fn:
                raise ValueError(f"{target}: {scale_role} is {part[scale_role].dtype} but {scheme} stores e4m3 scales")
        if "weight_global" in part:
            g = part["weight_global"].reshape(-1).to(torch.float32)
            if g.numel() != 1:
                raise ValueError(f"{target}: weight_global has {g.numel()} elements, expected one per-tensor scale")
            out["weight_global"] = g.to(torch.float16).expand(rows).contiguous()
        if "input_scale" in part:
            out["input_scale"] = part["input_scale"].reshape(()).to(torch.float32)
        return out


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense weights fused to the model's buffers (TP-sharded under TP>1), and the routed experts only where a resident path takes them from here: bf16 stacked experts as stored, block-fp8 experts restacked per layer, NVFP4 experts as the fused backend's native banks.

    Per-expert NVFP4 experts otherwise come from the offload cache's expert reader.
    """
    hf_config = cached_load_hf_config(model_path)
    config = parse_config(hf_config)
    stacked = include_moe_experts and config.is_moe and config.expert_quant == "none"
    if include_non_moe or stacked:
        reader = _DenseReader(get_quant_config(), get_model_spec(hf_config.architectures[0]), config) if include_non_moe else None
        yield from _iter_shards(model_path, device, reader, stacked=stacked, include_vision=include_vision, config=config)
    if include_moe_experts and config.is_moe and config.expert_quant == "fp8_block":
        yield from _resident_fp8_experts(model_path, config)
    if include_moe_experts and config.is_moe and config.expert_quant == "nvfp4":
        # Resident (fused) NVFP4 experts: TP-sharded native banks (the offload path loads
        # these into the expert cache instead; include_moe_experts is False there).
        yield from _iter_resident_nvfp4_experts(model_path, config)


def _iter_shards(model_path: str, device: torch.device, reader: _DenseReader | None, *, stacked: bool, include_vision: bool, config):
    tp = get_tp_info()
    for file in tqdm(iter_weight_files(model_path), desc="Loading weights", disable=not tp.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is None or _EXPERT_RE.search(name):
                    continue
                if not include_vision and name.startswith(VISION_KEY_PREFIXES):
                    continue
                if _STACKED_EXPERT_RE.match(name):
                    if stacked:
                        yield name, _maybe_shard(name, f.get_tensor(raw_name))
                    continue
                if reader is None:
                    continue
                tensor = f.get_tensor(raw_name)
                emitted = reader.add(name, tensor)
                if emitted is not None:
                    yield from emitted
                elif _is_gemma_norm(name):
                    yield name, tensor + 1.0  # (1 + weight) baked into the stored norm weight
                else:
                    if name.endswith("conv1d.weight") and tp.size > 1:
                        # depthwise conv: shard the channels per sub-part (q | k | v) so each
                        # rank keeps whole heads (the GDN recurrence needs them).
                        g = _linear_group(config)
                        if g is not None:
                            key_dim = g.num_key_heads * g.key_head_dim
                            value_dim = g.num_value_heads * g.value_head_dim
                            tensor = _shard_tp_parts(tensor, (key_dim, key_dim, value_dim),
                                                     rank=tp.rank, world_size=tp.size)
                    yield name, _maybe_shard(name, tensor)
    if reader is not None and reader.pending:
        lines = reader.missing()
        shown = "\n  ".join(lines[:8]) + (f"\n  ... {len(lines) - 8} more" if len(lines) > 8 else "")
        raise ValueError(f"checkpoint is missing tensors the quant config declares for {len(lines)} modules:\n  {shown}")


def iter_vision_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The vision tower alone, named as iter_weights names it."""
    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is not None and name.startswith(VISION_KEY_PREFIXES):
                    yield name, f.get_tensor(raw_name)


def iter_weights_parallel(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    workers: int = 8,
    chunk: int = 8 << 20,
) -> Iterator[tuple[str, torch.Tensor]]:
    """experts-only parallel read via the common chunked multi-threaded O_DIRECT reader.
    Qwen3.5 stores experts pre-fused/pre-stacked per layer (already ``[E, ...]``), so no
    merge/stack -- just rename and yield; bank builder places by name as the serial path.
    The pieces stay full under TP>1; the kernel pack shards per rank."""
    assert include_moe_experts and not include_non_moe, (
        "qwen3_5_moe parallel reader is experts-only (used by the expert piece reader)"
    )
    from freetoken.models.weight import iter_expert_tensors_parallel

    def _is_expert(raw_name: str) -> bool:
        name = _rename(raw_name)
        return name is not None and _STACKED_EXPERT_RE.match(name) is not None

    for raw_name, tensor in iter_expert_tensors_parallel(
        model_path, _is_expert, workers=workers, chunk=chunk
    ):
        yield _rename(raw_name), tensor


# ======================================================================================
# Block-FP8 routed experts (Qwen3.5-35B-A3B-FP8): offload expert pieces and resident stacks.
# ======================================================================================

# Routed-expert checkpoint key (per-expert, un-fused). ``mtp.layers...`` is excluded by the
# ``model.language_model.`` anchor, so the parallel reader only sees the real experts.
_FP8_EXPERT_KEY_RE = (
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate|up|down)_proj\.(?P<kind>weight|{scale})$"
)


def _resident_fp8_experts(model_path, config):
    from freetoken.kernel.triton.fp8_block_linear import FP8

    B = 128
    L, E, H, I, dense = _moe_dims(config)
    tp = get_tp_info()
    # TP-sharded on the intermediate dim (matches Fp8BlockMoEMethod's local banks):
    # gate_up column-parallel (gate rows [r*i, (r+1)*i), up rows [i + r*i, ...)),
    # down row-parallel (input cols [r*i, (r+1)*i)).
    i = I // tp.size
    shapes = {
        "gate_up_proj": ((E, 2 * i, H), FP8),
        "gate_up_scale_inv": ((E, 2 * i // B, H // B), torch.bfloat16),
        "down_proj": ((E, H, i), FP8),
        "down_scale_inv": ((E, H // B, i // B), torch.bfloat16),
    }
    g0, g1 = tp.rank * i, (tp.rank + 1) * i
    is_ = i // B
    gs0, gs1 = tp.rank * is_, (tp.rank + 1) * is_
    layers: dict[int, dict[str, torch.Tensor]] = {}
    placed = [0] * L
    for li, e0, e1, piece in iter_expert_pieces(model_path, config, QuantKind.FP8_BLOCK, parallel=None):
        stack = layers.setdefault(li, {n: torch.empty(shape, dtype=dt) for n, (shape, dt) in shapes.items()})
        # local bank: gate rows [0:i) | up rows [i:2i); the piece slice is rank-dependent
        stack["gate_up_proj"][e0:e1, :i] = piece["gate"][:, g0:g1, :]
        stack["gate_up_proj"][e0:e1, i:2 * i] = piece["up"][:, g0:g1, :]
        stack["gate_up_scale_inv"][e0:e1, :is_] = piece["gate_scale"][:, gs0:gs1, :]
        stack["gate_up_scale_inv"][e0:e1, is_:2 * is_] = piece["up_scale"][:, gs0:gs1, :]
        stack["down_proj"][e0:e1] = piece["down"][:, :, g0:g1]
        stack["down_scale_inv"][e0:e1] = piece["down_scale"][:, :, gs0:gs1]
        placed[li] += e1 - e0
        if placed[li] == E:
            pre = f"model.layers.{dense + li}.mlp.experts"
            for name, tensor in layers.pop(li).items():
                yield f"{pre}.{name}", tensor
    assert not layers, f"incomplete resident fp8 experts for layers {sorted(layers)}"


def _iter_resident_nvfp4_experts(model_path: str, config) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the resident (fused) TP-sharded NVFP4 expert banks as per-layer state-dict views.

    Called by the dense pass when ``include_moe_experts`` is True (the fused backend). The
    offload path never calls this -- it loads the routed experts into the offload cache
    instead. The keys match the MoELayer's resident NVFP4 tensor attributes (``gate_up_packed``
    / ``gate_up_scale`` / ``gate_up_global`` / ``down_packed`` / ``down_scale`` /
    ``down_global``) under the ``...mlp.experts`` prefix; the engine moves each to GPU.

    TP-sharded on the intermediate dim (matches Nvfp4MoEMethod.create_weights): gate_up is
    column-parallel (gate rows [g0:g1) -> local [0:i), up rows [I+g0:I+g1) -> local [i:2i)),
    down is row-parallel (input cols, packed //2 / scale //16); the down global keeps the full
    H output rows."""
    from freetoken.layers.quantization.moe.base import global_rows
    from freetoken.models.nvfp4_banks import iter_nvfp4_expert_pieces

    tp = get_tp_info()
    L, E, H, I, dense = _moe_dims(config)
    i = I // tp.size
    assert i % 16 == 0, (
        f"per-rank intermediate {i} not a multiple of 16 (NVFP4 per-16 block scale must "
        f"not straddle a rank boundary)"
    )
    fp8 = torch.float8_e4m3fn
    specs = {
        "gate_up_packed": ((E, 2 * i, H // 2), torch.uint8),
        "gate_up_scale": ((E, 2 * i, H // 16), fp8),
        "gate_up_global": ((E, 2 * i), torch.float16),
        "down_packed": ((E, H, i // 2), torch.uint8),
        "down_scale": ((E, H, i // 16), fp8),
        "down_global": ((E, H), torch.float16),
    }
    g0, g1 = tp.rank * i, (tp.rank + 1) * i
    layers: dict[int, dict[str, torch.Tensor]] = {}
    placed = [0] * L
    for li, e0, e1, piece in iter_nvfp4_expert_pieces(
        model_path, config, nvfp4_expert_spec(model_path, config), parallel=None
    ):
        stack = layers.setdefault(li, {n: torch.empty(shape, dtype=dt) for n, (shape, dt) in specs.items()})
        # gate_up: column-parallel on the (unpacked) output rows
        stack["gate_up_packed"][e0:e1, :i] = piece["gate"][:, g0:g1]
        stack["gate_up_packed"][e0:e1, i:] = piece["up"][:, g0:g1]
        stack["gate_up_scale"][e0:e1, :i] = piece["gate_scale"][:, g0:g1]
        stack["gate_up_scale"][e0:e1, i:] = piece["up_scale"][:, g0:g1]
        stack["gate_up_global"][e0:e1, :i] = global_rows(piece["gate_global"], i)
        stack["gate_up_global"][e0:e1, i:] = global_rows(piece["up_global"], i)
        # down: row-parallel on the input cols (packed //2, per-16 scale //16); global full H
        stack["down_packed"][e0:e1] = piece["down"][:, :, g0 // 2:g1 // 2]
        stack["down_scale"][e0:e1] = piece["down_scale"][:, :, g0 // 16:g1 // 16]
        stack["down_global"][e0:e1] = global_rows(piece["down_global"], H)
        placed[li] += e1 - e0
        if placed[li] == E:
            pre = f"model.layers.{dense + li}.mlp.experts"
            for name, tensor in layers.pop(li).items():
                yield f"{pre}.{name}", tensor
    assert not layers, f"incomplete resident nvfp4 experts for layers {sorted(layers)}"


def _moe_dims(model_config):
    L = model_config.num_moe_layers
    return (
        L, model_config.num_experts, model_config.hidden_size,
        model_config.moe_intermediate_size, model_config.num_layers - L,  # dense prefix
    )


def iter_expert_pieces(model_path, config, kind: QuantKind, *, parallel: bool | None = False, workers: int = 8, chunk: int = 8 << 20):
    """Block-fp8 routed experts, one piece per expert: ``{gate, up, down}`` fp8 codes and their
    ``_scale`` (block scale) companions, named as the checkpoint's dialect stores them. Other expert kinds use the generic readers.

    The pieces stay full under TP>1; the kernel pack / resident restack shards per rank."""
    if kind is not QuantKind.FP8_BLOCK:
        return None
    from freetoken.models.weight import experts_scattered, iter_expert_tensors_parallel
    from freetoken.moe.expert_pieces import per_expert_pieces

    L, E, H, I, dense = _moe_dims(config)
    scale = get_quant_config().stored_tensors(QuantKind.FP8_BLOCK)["weight_scale_inv"].name
    key_re = re.compile(_FP8_EXPERT_KEY_RE.format(scale=re.escape(scale)))
    suffix = {"weight": "", scale: "_scale"}

    def locate(raw_name: str):
        m = key_re.match(raw_name)
        if m is None:
            return None
        li = int(m["layer"]) - dense
        if not 0 <= li < L:
            raise ValueError(f"unexpected routed-expert layer in {raw_name}")
        return li, int(m["expert"]), m["proj"] + suffix[m["kind"]]

    if parallel is None:
        parallel = experts_scattered(model_path)
    if parallel:
        tensors = iter_expert_tensors_parallel(
            model_path, lambda n: key_re.match(n) is not None, workers=workers, chunk=chunk
        )
        return per_expert_pieces(tensors, locate, tensors_per_expert=6)

    def _serial():
        reader = ShardReader(model_path, torch.device("cpu"))
        try:
            for li in tqdm(range(L), desc="Loading fp8 experts (serial)", disable=not get_tp_info().is_primary()):
                for e in range(E):
                    base = f"model.language_model.layers.{dense + li}.mlp.experts.{e}"
                    for proj in ("gate", "up", "down"):
                        for kind, suf in suffix.items():
                            name = f"{base}.{proj}_proj.{kind}"
                            yield name, reader.get_tensor(name)
        finally:
            reader.close()

    return per_expert_pieces(_serial(), locate, tensors_per_expert=6)


def nvfp4_expert_spec(model_path: str, config) -> Nvfp4ExpertSourceSpec:
    """The per-expert NVFP4 layout under the checkpoint's dialect names (ModelOpt or llm-compressor)."""
    quant = get_quant_config()
    stored = quant.stored_tensors(QuantKind.NVFP4)
    kind_map = {stored[role].name: kind for role, kind in _BANK_KINDS.items()}
    return Nvfp4ExpertSourceSpec(
        key_pattern=re.compile(_EXPERT_KEY_RE.format(kinds="|".join(map(re.escape, kind_map)))),
        proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
        layer_to_bank=lambda layer, config: layer,  # every layer is MoE
        desc=f"Qwen3.5 NVFP4 experts ({quant.dialect})",
        kind_map=kind_map,
        global_reciprocal=stored["weight_global"].reciprocal,
    )


__all__ = [
    "iter_weights",
    "iter_weights_parallel",
    "iter_expert_pieces",
    "nvfp4_expert_spec",
]
