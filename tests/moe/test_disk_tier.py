"""CPU tests for the NVMe disk tier (moe/disk_tier.py).

A synthetic NVFP4 MoE checkpoint (2 layers, 4 experts) is written as safetensors
shards with deterministic per-tensor content; the tests verify that
:class:`Nvfp4DiskIndex` resolves the right byte ranges and that
:class:`DiskTier` places the right bytes into slot-cache rows and rewrites the
miss list. No CUDA needed -- the "GPU" banks here are CPU tensors and the
staging pin is stubbed.
"""

import json
import re
import struct
import threading
import types

import pytest
import torch

from freetoken.moe.disk_tier import DiskTier, Nvfp4DiskIndex
from freetoken.moe.host_banks import HostBank
from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec

H, I, E, L = 16, 32, 4, 2
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")

SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=re.compile(
        r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\."
        r"(?P<kind>weight|weight_scale|weight_scale_2)$"
    ),
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,
    desc="disk-tier test",
)

# (proj, kind, shape, dtype) -- the native NVFP4 per-expert tensor layout.
TENSOR_SPECS = (
    ("gate_proj", "weight", (I, H // 2), torch.uint8),
    ("up_proj", "weight", (I, H // 2), torch.uint8),
    ("down_proj", "weight", (H, I // 2), torch.uint8),
    ("gate_proj", "weight_scale", (I, H // 16), torch.uint8),
    ("up_proj", "weight_scale", (I, H // 16), torch.uint8),
    ("down_proj", "weight_scale", (H, I // 16), torch.uint8),
    ("gate_proj", "weight_scale_2", (I,), torch.float16),
    ("up_proj", "weight_scale_2", (I,), torch.float16),
    ("down_proj", "weight_scale_2", (H,), torch.float16),
)

BANK_SHAPES = (
    (2 * I, H // 2),  # gate_up_packed
    (2 * I, H // 16),  # gate_up_scale
    (2 * I,),  # gate_up_global
    (H, I // 2),  # down_packed
    (H, I // 16),  # down_scale
    (H,),  # down_global
)
BANK_DTYPES = (torch.uint8, torch.uint8, torch.float16, torch.uint8, torch.uint8, torch.float16)


def _name(layer, expert, proj, kind):
    return f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.{kind}"


def _tensor_for(layer, expert, proj, kind):
    """Deterministic content: a base offset per (layer, expert, proj, kind) so any
    misplacement is visible."""
    proj_i = ("gate_proj", "up_proj", "down_proj").index(proj)
    kind_i = ("weight", "weight_scale", "weight_scale_2").index(kind)
    base = layer * 100000 + expert * 1000 + proj_i * 100 + kind_i * 10
    for p, k, shape, dtype in TENSOR_SPECS:
        if p == proj and k == kind:
            n = int(torch.tensor(shape).prod())
            if dtype == torch.uint8:
                return torch.arange(n, dtype=torch.uint8).add_(base % 251).view(shape)
            return (torch.arange(n, dtype=torch.float32) + base).to(dtype).view(shape)
    raise AssertionError((proj, kind))


@pytest.fixture()
def checkpoint(tmp_path):
    import safetensors.torch

    by_shard = {s: {} for s in SHARDS}
    weight_map = {}
    for layer in range(L):
        for expert in range(E):
            for proj, kind, _shape, _dtype in TENSOR_SPECS:
                shard = SHARDS[(layer * E + expert) % 2]
                name = _name(layer, expert, proj, kind)
                by_shard[shard][name] = _tensor_for(layer, expert, proj, kind)
                weight_map[name] = shard
    for shard, tensors in by_shard.items():
        safetensors.torch.save_file(tensors, str(tmp_path / shard), metadata={"format": "pt"})
    with open(tmp_path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump({"weight_map": weight_map, "metadata": None}, f)
    config = types.SimpleNamespace(num_experts=E, hidden_size=H, moe_intermediate_size=I,
                                   num_layers=L, first_k_dense_replace=0)
    return tmp_path, config


def _index(checkpoint):
    path, config = checkpoint
    return Nvfp4DiskIndex(str(path), config, SPEC)


def test_index_segments_match_file_bytes(checkpoint):
    import safetensors

    path, config = checkpoint
    index = _index(checkpoint)
    assert len(index.shard_paths) == 2
    # Every (bank, layer, expert) row segment must point at the exact tensor bytes.
    for bank_idx in range(6):
        for layer in range(L):
            for expert in range(E):
                segs = index.row_segments(bank_idx, layer, expert)
                expected = {
                    0: [("gate_proj", "weight"), ("up_proj", "weight")],
                    1: [("gate_proj", "weight_scale"), ("up_proj", "weight_scale")],
                    2: [("gate_proj", "weight_scale_2"), ("up_proj", "weight_scale_2")],
                    3: [("down_proj", "weight")],
                    4: [("down_proj", "weight_scale")],
                    5: [("down_proj", "weight_scale_2")],
                }[bank_idx]
                assert len(segs) == len(expected)
                for (shard_idx, off, nbytes), (proj, kind) in zip(segs, expected):
                    shard_path = index.shard_paths[shard_idx]
                    with open(shard_path, "rb") as f:
                        (hlen,) = struct.unpack("<Q", f.read(8))
                        meta = json.loads(f.read(hlen))
                        tstart, tend = meta[_name(layer, expert, proj, kind)]["data_offsets"]
                        base = 8 + hlen  # data_offsets are data-section-relative
                        assert (off, off + nbytes) == (tstart + base, tend + base), (
                            bank_idx, layer, expert, proj, kind)
                    with safetensors.safe_open(shard_path, framework="pt", device="cpu") as sf:
                        tensor = sf.get_tensor(_name(layer, expert, proj, kind))
                    assert nbytes == tensor.numel() * tensor.element_size()


def _fake_cache():
    """CPU stand-in for OffloadMoeCache: banks in schema order + miss-list tensors."""
    banks = [
        (
            [torch.zeros(E, *shape, dtype=dtype) for _ in range(L)],
            torch.full((8, *shape), 0xFF, dtype=dtype),
        )
        for shape, dtype in zip(BANK_SHAPES, BANK_DTYPES)
    ]
    cache = type("FakeCache", (), {})()
    cache.banks = banks
    cache.num_experts = E
    cache.num_layers = L
    cache.num_indices = torch.tensor([0], dtype=torch.int64)
    cache.src_indices = torch.zeros(64, dtype=torch.int32)
    cache.evict_slots = torch.zeros(64, dtype=torch.int32)
    return cache


def _tier(checkpoint, cache, ram_experts=2):
    index = _index(checkpoint)
    tier = DiskTier(index, cache, ram_experts=ram_experts, workers=2)
    # Staging must hold the largest bank row's O_DIRECT super-block (a size
    # regression here overflows the buffer -> EFAULT/segfault at fetch time).
    max_row = max(
        b[0][0][0].numel() * b[0][0][0].element_size() for b in cache.banks)
    assert tier._staging_size >= max_row + 2 * 4096, (tier._staging_size, max_row)
    # Stub the pinned staging (HostBank.pin needs CUDA) with the same per-thread
    # buffer semantics as production (threading.local).
    local = threading.local()

    def _staging_buf():
        buf = getattr(local, "buf", None)
        if buf is None:
            buf = HostBank((tier._staging_size,), torch.uint8)
            local.buf = buf
        return buf

    tier._staging_buf = _staging_buf
    return tier


def _expected_rows(layer, expert):
    """The 6 bank rows for one expert, as flat uint8, in schema order."""
    rows = []
    for bank_idx, (shape, dtype) in enumerate(zip(BANK_SHAPES, BANK_DTYPES)):
        if bank_idx < 3:
            gate = _tensor_for(layer, expert, "gate_proj", ("weight", "weight_scale", "weight_scale_2")[bank_idx])
            up = _tensor_for(layer, expert, "up_proj", ("weight", "weight_scale", "weight_scale_2")[bank_idx])
            row = torch.cat([gate.reshape(-1), up.reshape(-1)]).view(shape)
        else:
            kind = ("weight", "weight_scale", "weight_scale_2")[bank_idx - 3]
            row = _tensor_for(layer, expert, "down_proj", kind)
        rows.append(row.contiguous().view(torch.uint8).reshape(-1))
    return rows


def test_fetch_expert_places_all_banks(checkpoint):
    cache = _fake_cache()
    tier = _tier(checkpoint, cache)
    for layer in range(L):
        for expert in range(E):
            slot = (layer * E + expert) % 8
            tier._fetch_expert(layer, expert, slot)
            expected = _expected_rows(layer, expert)
            for bank_idx, (host_layer, gpu_cache) in enumerate(cache.banks):
                got = gpu_cache[slot].contiguous().view(torch.uint8).reshape(-1)
                assert torch.equal(got, expected[bank_idx]), (bank_idx, layer, expert)
    stats = tier.stats()
    assert stats["experts_fetched"] == L * E


def test_fetch_pending_filters_and_rewrites(checkpoint):
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2)  # experts 0,1 RAM; 2,3 disk
    layer = 1
    # Miss list: expert 0 (RAM), 2 (disk), 3 (disk) -> slots 5, 6, 7.
    cache.src_indices[:3] = torch.tensor([0, 2, 3], dtype=torch.int32)
    cache.evict_slots[:3] = torch.tensor([5, 6, 7], dtype=torch.int32)
    cache.num_indices.fill_(3)

    tier.fetch_pending(cache, layer)

    # Disk misses fetched into their slots...
    expected = _expected_rows(layer, 2)
    for bank_idx, (_host, gpu_cache) in enumerate(cache.banks):
        assert torch.equal(
            gpu_cache[6].contiguous().view(torch.uint8).reshape(-1), expected[bank_idx])
    expected = _expected_rows(layer, 3)
    for bank_idx, (_host, gpu_cache) in enumerate(cache.banks):
        assert torch.equal(
            gpu_cache[7].contiguous().view(torch.uint8).reshape(-1), expected[bank_idx])
    # ...and the miss list shrank to the RAM-resident remainder.
    assert cache.num_indices.item() == 1
    assert cache.src_indices[0].item() == 0
    assert cache.evict_slots[0].item() == 5


def test_fetch_pending_all_ram_is_noop(checkpoint):
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2)
    cache.src_indices[:2] = torch.tensor([0, 1], dtype=torch.int32)
    cache.evict_slots[:2] = torch.tensor([0, 1], dtype=torch.int32)
    cache.num_indices.fill_(2)
    tier.fetch_pending(cache, 0)
    assert cache.num_indices.item() == 2
    assert tier.stats()["experts_fetched"] == 0


def test_fetch_pending_all_disk_clears_list(checkpoint):
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2)
    cache.src_indices[:1] = torch.tensor([3], dtype=torch.int32)
    cache.evict_slots[:1] = torch.tensor([4], dtype=torch.int32)
    cache.num_indices.fill_(1)
    tier.fetch_pending(cache, 0)
    assert cache.num_indices.item() == 0
    expected = _expected_rows(0, 3)
    for bank_idx, (_host, gpu_cache) in enumerate(cache.banks):
        assert torch.equal(
            gpu_cache[4].contiguous().view(torch.uint8).reshape(-1), expected[bank_idx])


def test_release_range_frees_pages():
    """release_range must actually drop the resident pages (MAP_SHARED /dev/zero
    mappings ignore MADV_DONTNEED, so the in-place private remap is verified
    with mincore)."""
    import ctypes as ct

    size = 4 * 1024 * 1024
    bank = HostBank((size,), torch.uint8)
    bank.tensor.fill_(7)  # fault every page in
    libc = ct.CDLL("libc.so.6", use_errno=True)
    libc.mincore.argtypes = [ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_ubyte)]
    libc.mincore.restype = ct.c_int

    def resident_pages(addr, nbytes):
        vec = (ct.c_ubyte * ((nbytes + 4095) // 4096))()
        assert libc.mincore(addr, nbytes, vec) == 0
        return sum(1 for b in vec if b & 1)

    pages = size // 4096
    assert resident_pages(bank.addr, size) == pages
    bank.release_range(0, size)
    assert resident_pages(bank.addr, size) == 0
    # The mapping stays valid: the refaulted pages read back as zeros.
    assert bank.tensor[0] == 0
