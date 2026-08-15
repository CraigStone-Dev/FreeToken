from __future__ import annotations

from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.config import detect_compressed_tensors_nvfp4
from freetoken.models.loader import ct_nvfp4_fuse, iter_weight_files, nvfp4_parts_ct
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

# Vision stack of the multimodal wrapper -- served text-only, always dropped.
_VISION_PREFIXES = (
    "model.vision_tower.",
    "model.vision_adapter.",
    "model.vision_projection.",
    "vision_tower.",
    "vision_adapter.",
    "vision_projection.",
)

# Fused projections, concatenated on the output dim in this exact order to match the
# model's merged-linear splits. The attention gate rides the q/k/v fusion (it is computed
# from the same layer input); ``.self_attn.gate_proj`` and the SwiGLU ``.mlp.gate_proj``
# are disambiguated by the full suffix.
_FUSIONS: dict[str, tuple[str, ...]] = {
    ".self_attn.qkvg_proj": (
        ".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.gate_proj",
    ),
    ".mlp.gate_up_proj": (".mlp.gate_proj", ".mlp.up_proj"),
}

# compressed-tensors quant scales, consumed with their ``weight_packed`` (the input
# scales are for W4A4 activation quant, which FreeToken does not run).
_CT_SCALE_SUFFIXES = (
    ".weight_scale", ".weight_global_scale", ".input_global_scale", ".input_scale",
)


def _rename(raw_name: str) -> str | None:
    """HF key -> FreeToken state-dict key, or None to skip (vision stack)."""
    if raw_name.startswith(_VISION_PREFIXES):
        return None
    if raw_name.startswith("model.language_model."):
        return "model." + raw_name[len("model.language_model.") :]
    if raw_name.startswith("language_model."):
        return "model." + raw_name[len("language_model.") :]
    return raw_name  # lm_head.weight


def _try_fuse(
    name: str, tensor: torch.Tensor, buf: dict[str, dict[int, torch.Tensor]]
) -> tuple[str, torch.Tensor] | tuple[()] | None:
    """Buffer a bf16 fusion part; return the merged ``(name, tensor)`` once all parts
    arrive, ``()`` while incomplete, ``None`` if not a fusion part."""
    base = name[: -len(".weight")]
    for fused_suffix, parts in _FUSIONS.items():
        for idx, part in enumerate(parts):
            if base.endswith(part):
                key = base[: -len(part)] + fused_suffix + ".weight"
                slots = buf.setdefault(key, {})
                slots[idx] = tensor
                if len(slots) == len(parts):
                    del buf[key]
                    return key, torch.cat([slots[i] for i in range(len(parts))], dim=0)
                return ()
    return None


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    if not include_non_moe:
        return  # dense model: there is no experts-only pass
    if get_tp_info().size > 1:
        raise NotImplementedError("muse_glimmer weight loading currently supports TP=1 only")

    if detect_compressed_tensors_nvfp4(cached_load_hf_config(model_path)):
        yield from _iter_weights_compressed_tensors(model_path, device)
        return

    fuse_buf: dict[str, dict[int, torch.Tensor]] = {}
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is None:
                    continue
                tensor = f.get_tensor(raw_name)
                if name.endswith(".weight"):
                    fused = _try_fuse(name, tensor, fuse_buf)
                    if fused is not None:
                        if fused != ():  # () means buffered, not yet complete
                            yield fused
                        continue
                # All norms pass through raw: the decoder's centered (1+w) norms apply the
                # +1 at runtime (GemmaPlusOneRMSNorm), the final norm / qk norms need none.
                yield name, tensor

    assert not fuse_buf, f"Incomplete projection fusions: {list(fuse_buf.keys())}"


def _iter_weights_compressed_tensors(
    model_path: str, device: torch.device
) -> Iterator[tuple[str, torch.Tensor]]:
    """Dense pass for the compressed-tensors NVFP4 checkpoint.

    Every text Linear (q/k/v/o, the attention gate, the MLP) is kept native NVFP4
    (W4A16): ``.weight`` (uint8 packed) + ``.weight_scale`` (fp8 block) + ``.weight_global``
    (fp16 per-row, the reciprocal of the stored quant-side global). q/k/v/gate fuse into
    ``qkvg_proj`` and gate/up into ``gate_up_proj`` on the output dim, each part keeping
    its own scales, so the fused FP4 weights are exact. Embeddings, norms and lm_head are
    bf16 (the checkpoint's ignore list)."""
    nvfp4_buf: dict[str, dict[int, tuple]] = {}
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading compressed-tensors weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                if raw_name.endswith(_CT_SCALE_SUFFIXES):
                    continue  # consumed with their weight_packed
                name = _rename(raw_name)
                if name is None:
                    continue
                if raw_name.endswith(".weight_packed"):
                    base = name[: -len(".weight_packed")]
                    parts = nvfp4_parts_ct(f, raw_name[: -len(".weight_packed")])
                    emit = ct_nvfp4_fuse(base, parts, nvfp4_buf, _FUSIONS)
                    if emit is not None:
                        yield from emit
                    else:  # standalone: o_proj, down_proj
                        w, s, g = parts
                        yield base + ".weight", w
                        yield base + ".weight_scale", s
                        yield base + ".weight_global", g
                    continue
                yield name, f.get_tensor(raw_name)

    assert not nvfp4_buf, f"Incomplete NVFP4 fusions: {list(nvfp4_buf.keys())}"


__all__ = ["iter_weights"]
