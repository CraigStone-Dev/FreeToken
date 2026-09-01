"""Disk tier: NVMe-backed MoE experts (VRAM <- RAM <- NVMe).

Lets the offload backend serve experts that do NOT fit in pinned RAM: the RAM
bank holds only the first ``ram_experts`` experts per layer (pinned), the rest
stay on disk in the original checkpoint. When the GPU slot cache misses a
disk-resident expert, :class:`DiskTier` fetches its rows with O_DIRECT preadv
into a small pinned staging buffer and H2D-copies them into the slot the LRU
kernel already assigned, then shrinks the miss list so the existing PCIe
``copy_missing`` path only moves the RAM-resident misses.

v0 scope (prototype):
* native NVFP4 layout only (the "triton" backend banks -- what sm_120 picks);
* ``decode_target == "gpu"`` (offload) only -- the CPU executor reads banks
  directly and would read released pages;
* synchronous fetch (the layer waits for its disk misses); no CUDA-graph
  capture (the miss-list D2H/H2D round trip is host-side and variable);
* prefill_overlap off (the double-buffer prefill path bypasses the slot cache).

The bank rows are read from the ORIGINAL safetensors shards: every expert
tensor is a contiguous per-expert tensor, so a bank row is one (or two, for
the gate|up-fused banks) aligned super-block preads. No FTW conversion needed.
"""

from __future__ import annotations

import ctypes
import json
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

from freetoken.moe.host_banks import HostBank

_ALIGN = 4096


@dataclass(frozen=True)
class DiskTierSpec:
    """Engine -> loader: how many experts per layer stay pinned in RAM.

    Experts ``[0, ram_experts)`` are pinned as usual; ``[ram_experts, E)`` keep
    their bank rows allocated but their pages are released after load and are
    served from disk by :class:`DiskTier`."""

    ram_experts: int


def release_bank_tails(banks_by_name: dict[str, list[HostBank]], num_experts: int,
                       ram_experts: int) -> None:
    """MADV_DONTNEED the unpinned tail rows of every bank layer (post-load)."""
    for layer_banks in banks_by_name.values():
        for bank in layer_banks:
            row_bytes = bank.nbytes // num_experts
            bank.release_range(ram_experts * row_bytes, bank.nbytes - ram_experts * row_bytes)

# Native NVFP4 bank order (== _BANK_SCHEMAS["nvfp4"]) and, per bank, the
# checkpoint segments that make up one expert row: (proj, kind, dst_row_start,
# dst_row_end). The gate|up-fused banks splice gate rows then up rows on the
# output-row axis; down banks are a single segment. Row ends are None = rest.
_NVP4_BANK_SEGS = (
    (("gate_proj", "weight", 0, None), ("up_proj", "weight", None, None)),
    (("gate_proj", "weight_scale", 0, None), ("up_proj", "weight_scale", None, None)),
    (("gate_proj", "weight_scale_2", 0, None), ("up_proj", "weight_scale_2", None, None)),
    (("down_proj", "weight", 0, None),),
    (("down_proj", "weight_scale", 0, None),),
    (("down_proj", "weight_scale_2", 0, None),),
)



def _preadv_error(tier, staging, shard_idx: int, off: int, a0: int, slen: int,
                  direct: bool) -> OSError:
    vma = "?"
    try:
        for line in open("/proc/self/maps"):
            lo, hi = line.split()[0].split("-")
            if int(lo, 16) <= staging.addr < int(hi, 16):
                vma = line.strip()[:120]
                break
    except OSError:
        pass
    return OSError(
        f"disk-tier preadv failed: shard={shard_idx} off={off} a0={a0} "
        f"slen={slen} direct={direct} buf={hex(staging.addr)} "
        f"staging_size={tier._staging_size} thread={threading.current_thread().name} "
        f"vma={vma}"
    )


def _read_safetensors_offsets(path: str) -> dict[str, tuple[int, int]]:
    """{tensor_name: (start, end)} from a shard's safetensors header, as ABSOLUTE
    file offsets (data_offsets are relative to the data section, i.e. after the
    8-byte length + header JSON)."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        meta = json.loads(f.read(hlen))
    base = 8 + hlen
    return {
        k: (v["data_offsets"][0] + base, v["data_offsets"][1] + base)
        for k, v in meta.items() if k != "__metadata__"
    }


class Nvfp4DiskIndex:
    """(bank, layer, expert) -> per-segment (shard_idx, offset, nbytes) locations.

    Built from the original checkpoint: the HF index json (name -> shard) plus
    each referenced shard's safetensors header (name -> byte range). Expert
    tensors are per-expert and contiguous, so a row is exactly one byte range
    per segment.
    """

    def __init__(self, model_dir: str, config, spec) -> None:
        from freetoken.models.nvfp4_banks import _num_moe_layers
        from freetoken.utils.hf import download_hf_weight

        model_dir = download_hf_weight(model_dir)  # hub id -> local cache dir; no-op if local
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

        num_layers = _num_moe_layers(config)
        # (bank_layer, expert, proj, kind) -> (tensor_name, shard)
        loc: dict[tuple[int, int, str, str], tuple[str, str]] = {}
        for name, shard in weight_map.items():
            m = spec.key_pattern.match(name)
            if m is None:
                continue
            bank_layer = spec.layer_to_bank(int(m.group("layer")), config)
            if bank_layer is None:
                continue
            loc[(bank_layer, int(m.group("expert")), m.group("proj"), m.group("kind"))] = (
                name, shard)

        shards = sorted(set(shard for _, shard in loc.values()))
        self.shard_paths = [os.path.join(model_dir, s) for s in shards]
        offsets = {s: _read_safetensors_offsets(os.path.join(model_dir, s)) for s in shards}
        shard_idx = {s: i for i, s in enumerate(shards)}

        E = config.num_experts
        seg_size = struct.calcsize("<iqq")  # (shard_idx, offset, nbytes)
        self.entries: list[list[bytes]] = []  # [bank][layer] -> packed segments per expert
        for bank_idx in range(len(_NVP4_BANK_SEGS)):
            per_layer = []
            for layer in range(num_layers):
                rows = bytearray()
                for e in range(E):
                    for proj, kind, _, _ in _NVP4_BANK_SEGS[bank_idx]:
                        key = (layer, e, proj, kind)
                        entry = loc.get(key)
                        if entry is None:
                            raise KeyError(
                                f"disk tier: no {proj}.{kind} tensor for layer {layer} expert {e} "
                                f"(bank {bank_idx}) in {index_path}"
                            )
                        name, shard = entry
                        start, end = offsets[shard][name]
                        rows += struct.pack("<iqq", shard_idx[shard], start, end - start)
                per_layer.append(bytes(rows))
            self.entries.append(per_layer)
        self._seg_size = seg_size

    def row_segments(self, bank_idx: int, layer: int, expert: int) -> list[tuple[int, int, int]]:
        """[(shard_idx, offset, nbytes)] for one expert row, in segment order."""
        base = expert * self._seg_size * len(_NVP4_BANK_SEGS[bank_idx])
        raw = self.entries[bank_idx][layer][base:base + self._seg_size * len(_NVP4_BANK_SEGS[bank_idx])]
        return [
            struct.unpack_from("<iqq", raw, i * self._seg_size)
            for i in range(len(_NVP4_BANK_SEGS[bank_idx]))
        ]


class DiskTier:
    """Runtime fetcher: disk-resident slot-cache misses -> staging -> GPU slot."""

    def __init__(self, index: Nvfp4DiskIndex, cache, ram_experts: int, workers: int = 8) -> None:
        self._index = index
        self._ram = ram_experts
        self._banks = list(cache.banks)  # [(per_layer_host, gpu_cache)] in schema order
        self._row_bytes = [
            b[0][0][0].numel() * b[0][0][0].element_size() for b in self._banks
        ]  # full expert-row bytes per bank (staging must hold the biggest one)
        # Per-bank destination row slices (gate|up split at the row midpoint).
        self._dst_slices: list[list[tuple[int, int]]] = []
        for bank_idx, (host_layer, _gpu) in enumerate(self._banks):
            row = host_layer[0][0]
            if len(_NVP4_BANK_SEGS[bank_idx]) == 2:
                mid = row.shape[0] // 2
                self._dst_slices.append([(0, mid), (mid, row.shape[0])])
            else:
                self._dst_slices.append([(0, row.shape[0])])
        max_row = max(self._row_bytes)
        self._staging_size = ((max_row + _ALIGN - 1) // _ALIGN + 2) * _ALIGN
        self._staging = threading.local()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="disk-tier")
        self._fd_lock = threading.Lock()
        self._fds: dict[int, tuple[int, bool]] = {}
        self._fetches = 0
        self._fetch_bytes = 0
        self._decode_verify_done = False

    # ------------------------------------------------------------------ fds
    def _fd(self, shard_idx: int) -> tuple[int, bool]:
        """(fd, o_direct) for a shard; O_DIRECT falls back to plain preadv where the
        filesystem refuses it (tmpfs/overlayfs -- tests)."""
        ent = self._fds.get(shard_idx)
        if ent is None:
            with self._fd_lock:
                ent = self._fds.get(shard_idx)
                if ent is None:
                    path = self._index.shard_paths[shard_idx]
                    try:
                        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                        direct = True
                    except OSError:
                        fd = os.open(path, os.O_RDONLY)
                        direct = False
                    ent = (fd, direct)
                    self._fds[shard_idx] = ent
        return ent

    # --------------------------------------------------------------- staging
    def _staging_buf(self) -> HostBank:
        buf = getattr(self._staging, "buf", None)
        if buf is None:
            buf = HostBank((self._staging_size,), torch.uint8)
            buf.pin()  # small; pin once per worker thread
            self._staging.buf = buf
        return buf

    # ---------------------------------------------------------------- fetch
    def _fetch_expert(self, layer: int, expert: int, slot: int) -> None:
        # The server runs under inference_mode; the fetch pool threads do not,
        # so the H2D writes into the (inference) slot cache need their own scope.
        with torch.inference_mode():
            self._fetch_expert_inner(layer, expert, slot)

    def _fetch_expert_inner(self, layer: int, expert: int, slot: int) -> None:
        staging = self._staging_buf()
        for bank_idx, (_host_layer, gpu_cache) in enumerate(self._banks):
            row = gpu_cache[slot]
            segs = self._index.row_segments(bank_idx, layer, expert)
            for (d0, d1), (shard_idx, off, nbytes) in zip(self._dst_slices[bank_idx], segs):
                fd, direct = self._fd(shard_idx)
                if direct:
                    a0 = off & ~(_ALIGN - 1)
                    slen = (off + nbytes - a0 + _ALIGN - 1) & ~(_ALIGN - 1)
                else:
                    a0, slen = off, nbytes
                mv = (ctypes.c_char * slen).from_address(staging.addr)
                try:
                    os.preadv(fd, [mv], a0)
                except OSError:
                    raise _preadv_error(self, staging, shard_idx, off, a0, slen, direct)
                row_off = off - a0
                if bank_idx in (2, 5):
                    # Global-scale banks: the checkpoint stores a per-expert fp32
                    # SCALAR (weight_scale_2); the bank row is that value as fp16
                    # broadcast across the row -- convert + fill, no byte copy.
                    val = staging.tensor[row_off:row_off + 4].view(torch.float32)[0].to(
                        torch.float16)
                    row[d0:d1].fill_(val)
                    continue
                src = staging.tensor[row_off:row_off + nbytes]
                dst = row[d0:d1]
                dst.copy_(src.view(dst.dtype).view(dst.shape), non_blocking=True)
        self._fetches += 1
        self._fetch_bytes += sum(self._row_bytes)

    def _sync_fetches(self) -> None:
        """Wait for the pool threads' async H2D copies to land.

        The copies are enqueued on the pool threads' default stream; f.result() only
        waits for them to be ENQUEUED. The GEMM's stream is not ordered with that
        stream, so sync the default stream before the GEMM reads the slots."""
        if not torch.cuda.is_available():
            return  # CPU-only tests: the copies are synchronous CPU->CPU
        torch.cuda.default_stream().synchronize()

    def _verify_slot(self, cache, layer: int, expert: int, slot: int | None = None) -> None:
        """One-shot debug: read back an expert's slot rows and compare against the
        checkpoint bytes (ground truth). Gated on FT_DISK_TIER_VERIFY."""
        import torch
        if slot is None:
            slot = expert  # identity mapping (prefill)
        for bank_idx, (_host_layer, gpu_cache) in enumerate(self._banks):
            slot_row = gpu_cache[slot].contiguous()
            flat = slot_row.view(torch.uint8).reshape(-1)
            segs = self._index.row_segments(bank_idx, layer, expert)
            dsl = self._dst_slices[bank_idx]
            # Reconstruct the reference row bytes from the checkpoint.
            row_bytes = flat.numel()
            ref = torch.zeros(row_bytes, dtype=torch.uint8)
            row_el = slot_row.element_size()
            row_leading = slot_row.numel() // slot_row.shape[0] if slot_row.dim() > 1 else 1
            for (d0, d1), (shard_idx, off, nbytes) in zip(dsl, segs):
                fd, direct = self._fd(shard_idx)
                a0 = off if not direct else (off & ~(_ALIGN - 1))
                slen = nbytes if not direct else (off + nbytes - a0 + _ALIGN - 1) & ~(_ALIGN - 1)
                buf = os.pread(fd, slen, a0)
                row_off = off - a0
                seg = buf[row_off:row_off + nbytes]
                if bank_idx in (2, 5):
                    val = int.from_bytes(seg, "little")
                    import struct as _st
                    f32 = _st.unpack("<f", seg[:4])[0]
                    import numpy as _np
                    f16 = _np.float16(f32).tobytes()
                    for r in range(d0, d1):
                        ref[r * row_el:(r + 1) * row_el] = torch.frombuffer(f16, dtype=torch.uint8)
                else:
                    dst_off = d0 * row_leading * row_el
                    ref[dst_off:dst_off + len(seg)] = torch.frombuffer(seg, dtype=torch.uint8)
            try:
                ref = ref.to(flat.device)
                match = bool(torch.equal(flat, ref))
                print(f"[verify] bank={bank_idx} expert={expert} match={match} "
                      f"slot_norm={slot_row.float().norm().item():.4f} "
                      f"ref_norm={ref.view(slot_row.dtype).view(slot_row.shape).float().norm().item():.4f} "
                      f"slot_head={flat[:8].tolist()} ref_head={ref[:8].tolist()}", flush=True)
            except Exception as exc:  # never crash the server in debug
                print(f"[verify] bank={bank_idx} expert={expert} ERROR {exc!r}", flush=True)

    def verify_ram(self, cache, layer: int) -> None:
        """One-shot debug: after the PCIe copy, check a RAM-resident expert's slot rows
        against the checkpoint reference. Gated on FT_DISK_TIER_VERIFY."""
        expert = min(10, self._ram - 1)  # a RAM-resident expert
        print(f"[verify-ram] layer={layer} expert={expert} (RAM prefix)", flush=True)
        self._verify_slot(cache, layer, expert)

    def materialize_layer(self, cache, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Disk-tier prefill: materialize the RAM-resident prefix into identity slots
        (the normal kernel restricted to K experts; the following ``copy_missing``
        streams it over PCIe), then fetch the routed disk-resident experts into
        THEIR identity slots. The identity mapping (position == expert id) is
        preserved, so the prefill GEMM is unchanged."""
        from freetoken.moe.offload_kernels import _materialize_layer_gpu

        _materialize_layer_gpu(cache, layer_id, materialize_count=self._ram)
        routed = expert_ids.reshape(-1)
        disk = torch.unique(routed[routed >= self._ram])
        if layer_id < 3:
            print(f"[disk-tier dbg] layer={layer_id} routed={routed.numel()} "
                  f"unique_disk={disk.numel()} disk={disk.tolist()[:12]}", flush=True)
        if disk.numel() == 0:
            return
        step = int(cache.step.item())  # already incremented by the kernel
        futures = [
            self._pool.submit(self._fetch_expert, layer_id, int(e), int(e))
            for e in disk.tolist()
        ]
        for f in futures:
            f.result()
        self._sync_fetches()
        if os.environ.get("FT_DISK_TIER_VERIFY") and layer_id in (0, 20) and disk.numel() > 0:
            for e in disk.tolist()[:6]:
                self._verify_slot(cache, layer_id, int(e))
        # Same bookkeeping the materialize kernel writes, per fetched expert.
        flat = layer_id * cache.num_experts + disk
        cache.slot_for_id[layer_id, disk] = disk
        cache.id_of_slot[disk] = flat
        cache.usage[disk] = step

    def fetch_pending(self, cache, layer_id: int) -> None:
        """Fetch this layer's disk-resident misses into their slots; shrink the miss
        list to the RAM-resident remainder for the existing PCIe copy path."""
        n = int(cache.num_indices.item())
        if n == 0:
            return
        src = cache.src_indices[:n].cpu()
        slots = cache.evict_slots[:n].cpu()
        disk = [i for i in range(n) if int(src[i]) >= self._ram]
        if not disk:
            return
        futures = [
            self._pool.submit(self._fetch_expert, layer_id, int(src[i]), int(slots[i]))
            for i in disk
        ]
        for f in futures:
            f.result()
        self._sync_fetches()
        if os.environ.get("FT_DISK_TIER_VERIFY") and not self._decode_verify_done:
            i0 = disk[0]
            self._decode_verify_done = True
            print(f"[verify-decode] layer={layer_id} expert={int(src[i0])} slot={int(slots[i0])} "
                  f"ndisk={len(disk)}", flush=True)
            self._verify_slot(cache, layer_id, int(src[i0]), int(slots[i0]))
        disk_set = set(disk)
        ram = [i for i in range(n) if i not in disk_set]
        if ram:
            sel = torch.tensor(ram, dtype=torch.long)
            cache.src_indices[:len(ram)].copy_(src[sel].to(cache.src_indices.dtype))
            cache.evict_slots[:len(ram)].copy_(slots[sel].to(cache.evict_slots.dtype))
        cache.num_indices.fill_(len(ram))

    def refresh(self, cache) -> None:
        """Rebind the slot-cache references after a runtime cache rebuild."""
        self._banks = list(cache.banks)

    def stats(self) -> dict:
        return {"experts_fetched": self._fetches, "bytes_fetched": self._fetch_bytes}
