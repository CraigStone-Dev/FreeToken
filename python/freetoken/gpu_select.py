"""--gpu for ft serve / bench bw / checkpoint: narrow CUDA_VISIBLE_DEVICES in the parent, rank i then binds cuda:i = entry i.

Stdlib only (torch is imported lazily); not under freetoken.utils, which imports transformers.
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence

UUID_PREFIX = "GPU-"


def is_gpu_uuid(spec: str) -> bool:
    return spec[: len(UUID_PREFIX)].upper() == UUID_PREFIX


def is_gpu_index(spec: str) -> bool:
    # not str.isdigit(): that also accepts superscripts and other Unicode digits
    return spec.isascii() and spec.isdecimal()


def _canonical(entry: str) -> str:
    """A UUID in the exact form the driver matches (upper-case GPU- prefix), an index as-is."""
    if not (is_gpu_uuid(entry) or is_gpu_index(entry)):
        raise ValueError(
            f"{entry!r} is neither a GPU UUID (GPU-xxxx..., as `nvidia-smi -L` prints) "
            f"nor an nvidia-smi index"
        )
    return UUID_PREFIX + entry[len(UUID_PREFIX):] if is_gpu_uuid(entry) else entry


def parse_gpu_spec(value: str) -> tuple[str, ...]:
    """Split a --gpu value; ValueError on a bad entry, an empty value, or a mix of UUIDs and indices."""
    entries = tuple(_canonical(e.strip()) for e in value.split(",") if e.strip())
    if not entries:
        raise ValueError("--gpu needs at least one GPU")
    if len({is_gpu_uuid(e) for e in entries}) > 1:
        # the driver parses CUDA_VISIBLE_DEVICES as all-UUID or all-index
        raise ValueError("--gpu entries must be all UUIDs or all indices")
    return entries


def gpu_arg(value: str) -> tuple[str, ...]:
    """argparse type for a --gpu list."""
    try:
        return parse_gpu_spec(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def single_gpu_arg(value: str) -> str:
    """argparse type for a single-GPU --gpu."""
    entries = gpu_arg(value)
    if len(entries) != 1:
        raise argparse.ArgumentTypeError("takes exactly one GPU")
    return entries[0]


def apply_gpu_selection(specs: Sequence[str]) -> None:
    """Write --gpu entries into CUDA_VISIBLE_DEVICES. No-op when empty.

    With a preset CUDA_VISIBLE_DEVICES the choice must stay inside it: an index counts within
    the preset list, a UUID must match one of its entries.
    A bare index means the nvidia-smi number: CUDA_DEVICE_ORDER is forced to PCI_BUS_ID (the
    default FASTEST_FIRST numbers cards differently on a mixed box).
    """
    specs = parse_gpu_spec(",".join(specs)) if any(s.strip() for s in specs) else ()
    if not specs:
        return
    preset_raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    # CUDA ordinal k is preset[k]
    preset = None if preset_raw is None else [e.strip() for e in preset_raw.split(",") if e.strip()]

    entries: list[str] = []
    for spec in specs:
        if preset is None:
            entries.append(spec)
        elif is_gpu_uuid(spec):
            entries.append(_preset_uuid(spec, preset, preset_raw))
        else:
            idx = int(spec)
            if idx >= len(preset):
                raise ValueError(
                    f"--gpu {spec}: only {len(preset)} GPU(s) are visible through "
                    f"CUDA_VISIBLE_DEVICES={preset_raw!r} (indices count within that list)"
                )
            entries.append(preset[idx])

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(entries)
    if preset is None and not is_gpu_uuid(entries[0]):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def _preset_uuid(spec: str, preset: list[str], preset_raw: str) -> str:
    """The preset entry `spec` names (prefix match either way), else ValueError."""
    if not all(is_gpu_uuid(p) for p in preset):
        raise ValueError(
            f"--gpu {spec}: CUDA_VISIBLE_DEVICES={preset_raw!r} lists GPUs by index; "
            f"give --gpu as an index into that list"
        )
    hits = [p for p in preset if p.upper().startswith(spec.upper()) or spec.upper().startswith(p.upper())]
    if len(hits) != 1:
        raise ValueError(
            f"--gpu {spec}: not one of the GPUs visible through CUDA_VISIBLE_DEVICES={preset_raw!r}"
        )
    return hits[0]


def validate_gpu_selection(expected: int) -> None:
    """Raise if the visible GPU count is not ``expected``. Skipped without NVML."""
    import torch

    # NVML only: device_count() falls back to the CUDA runtime and would init CUDA in this process
    count_nvml = getattr(torch.cuda, "_device_count_nvml", None)
    found = count_nvml() if count_nvml is not None else -1
    if found < 0:
        return
    if found == expected:
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    entries = [e.strip() for e in visible.split(",") if e.strip()]
    # CUDA keeps the valid prefix, so entries[found] is the bad one (or an ambiguous prefix)
    if found < len(entries):
        raise RuntimeError(
            f"GPU {entries[found]!r} not found or not a unique prefix "
            f"(CUDA_VISIBLE_DEVICES={visible!r}); run `nvidia-smi -L` to list GPUs"
        )
    raise RuntimeError(
        f"expected {expected} visible GPU(s) but found {found} "
        f"(CUDA_VISIBLE_DEVICES={visible!r}); run `nvidia-smi -L` to list GPUs"
    )


def format_gpu_uuid(raw) -> str | None:
    """nvidia-smi form GPU-<uuid> from a uuid.UUID."""
    return None if raw is None else f"{UUID_PREFIX}{raw}"


def gpu_identity(index: int) -> dict:
    """{index, name, uuid, total_bytes} of visible device ``index``."""
    import torch

    props = torch.cuda.get_device_properties(index)
    return {
        "index": index,
        "name": props.name,
        "uuid": format_gpu_uuid(getattr(props, "uuid", None)),
        "total_bytes": int(props.total_memory),
    }
