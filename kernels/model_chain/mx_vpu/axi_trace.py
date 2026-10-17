#!/usr/bin/env python3
"""Validate and split addressed DRAM events into bounded, per-port AXI traces.

This is offline replay preparation. It does not feed HBM timing back to a
program. Input JSONL must identify its address space and observation boundary;
DMA accesses before the L2 cannot be relabeled as DRAM transactions.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gzip
import hashlib
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "soc"))
from scramble_gpu_elf import AddressHash

GPU_BASE = 0x100000000
GPU_SIZE = 0x80000000
HOST_BASE = 0x80000000
UNIT = 32
SLICES = 4
HASH = AddressHash(GPU_BASE, GPU_SIZE, UNIT, SLICES)
FIELDS = ("cycle", "sequence", "fragment", "axi_path", "write", "address",
          "beats", "size_log2", "source")


@dataclass(frozen=True)
class Event:
    cycle: int
    sequence: int
    source: str
    operation: str
    address: int
    size_bytes: int
    address_space: str
    boundary: str
    axi_path: int | None = None

    @classmethod
    def from_json(cls, item: dict) -> "Event":
        event = cls(**item)
        for name in ("cycle", "sequence", "address", "size_bytes"):
            if type(getattr(event, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not (0 <= event.cycle < 1 << 64 and 0 <= event.sequence < 1 << 64) or event.size_bytes <= 0:
            raise ValueError("cycle/sequence must fit uint64 and transfer must be positive")
        if event.operation not in ("read", "write"):
            raise ValueError("operation must be read or write")
        if event.boundary != "dram":
            raise ValueError("HBM replay requires DRAM-boundary events; pre-L2 DMA is not a miss trace")
        if not isinstance(event.source, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,64}", event.source):
            raise ValueError("source must be a short identifier")
        if event.address_space not in ("gpu_local", "system_unhashed", "system_physical"):
            raise ValueError("address_space must explicitly identify whether the GPU hash was applied")
        if event.address < 0 or event.address % 8 or event.size_bytes % 8:
            raise ValueError("this replay driver requires aligned, whole 64-bit AXI beats")
        if event.axi_path is not None and (type(event.axi_path) is not int or not 0 <= event.axi_path <= 4):
            raise ValueError("axi_path must identify one of the five region ports")
        return event


def physical(address: int, space: str) -> int:
    if space == "gpu_local":
        if not 0 <= address < GPU_SIZE:
            raise ValueError("GPU-local address exceeds the pinned 2 GiB region")
        return HASH.forward(GPU_BASE + address)
    if space not in ("system_unhashed", "system_physical"):
        raise ValueError("unknown address space")
    if not HOST_BASE <= address < GPU_BASE + GPU_SIZE:
        raise ValueError("address is outside the five pinned memory regions")
    return HASH.forward(address) if space == "system_unhashed" else address


def route(address: int) -> int:
    if HOST_BASE <= address < GPU_BASE:
        return 0
    if GPU_BASE <= address < GPU_BASE + GPU_SIZE:
        return 1 + (address - GPU_BASE) // (GPU_SIZE // SLICES)
    raise ValueError("physical address has no AXI region port")


def bursts(event: Event, ratio_num: int, ratio_den: int):
    if ratio_num <= 0 or ratio_den <= 0:
        raise ValueError("clock ratio must be positive")
    cycle = (event.cycle * ratio_num + ratio_den - 1) // ratio_den
    if cycle >= 1 << 64:
        raise ValueError("replay cycle overflows 64 bits")
    offset, fragment = 0, 0
    while offset < event.size_bytes:
        source = event.address + offset
        addr = physical(source, event.address_space)
        path = route(addr)
        if event.axi_path is not None and event.axi_path != path:
            raise ValueError(f"recorded AXI path {event.axi_path} disagrees with physical routing {path}")
        # A hashed unit cannot be assumed contiguous with its successor.
        limit = UNIT - source % UNIT if event.address_space != "system_physical" else 2048
        region_end = GPU_BASE if path == 0 else GPU_BASE + path * (GPU_SIZE // SLICES)
        count = min(event.size_bytes-offset, limit, 4096-addr % 4096, region_end-addr, 2048)
        if count <= 0 or count % 8:
            raise ValueError("transfer cannot be represented as whole AXI beats")
        yield dict(cycle=cycle, sequence=event.sequence, fragment=fragment,
                   axi_path=path, write=int(event.operation == "write"),
                   address=f"0x{addr:x}", beats=count//8, size_log2=3, source=event.source)
        offset += count
        fragment += 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def convert(source: Path, output: Path, ratio_num: int, ratio_den: int) -> dict:
    if output.exists():
        raise ValueError("output directory must be new, to avoid mixing replay runs")
    if ratio_num <= 0 or ratio_den <= 0:
        raise ValueError("explicit positive replay/source clock ratio is required")
    output.mkdir(parents=True)
    streams, writers, counts = {}, {}, [0]*5
    last_cycle, last_sequence, records, transferred = -1, -1, 0, 0
    opener = gzip.open if source.suffix == ".gz" else open
    try:
        with opener(source, "rt") as stream:
            for number, line in enumerate(iter(lambda: stream.readline(16385), ""), 1):
                if len(line) > 16384:
                    raise ValueError(f"input line {number}: oversized event")
                if not line.strip():
                    continue
                try:
                    event = Event.from_json(json.loads(line))
                    if event.cycle < last_cycle or event.sequence <= last_sequence:
                        raise ValueError("cycles must not decrease and global sequence must strictly increase")
                    last_cycle, last_sequence = event.cycle, event.sequence
                    for burst in bursts(event, ratio_num, ratio_den):
                        path = burst["axi_path"]
                        if path not in writers:
                            streams[path] = gzip.open(output/f"axi{path}.csv.gz", "wt", newline="")
                            writers[path] = csv.DictWriter(streams[path], fieldnames=FIELDS)
                            writers[path].writeheader()
                        writers[path].writerow(burst)
                        counts[path] += 1
                        transferred += burst["beats"] * 8
                    records += 1
                except (ValueError, TypeError) as error:
                    raise ValueError(f"input line {number}: {error}") from error
        if records == 0:
            raise ValueError("empty traces cannot pass a replay gate")
    finally:
        for stream in streams.values():
            stream.close()
    result = dict(schema="radiance-axi-replay-v1", input_sha256=sha256(source),
                  clock_ratio=dict(replay=ratio_num, source=ratio_den),
                  input_records=records, transferred_bytes=transferred, port_bursts=counts,
                  files={p.name: sha256(p) for p in sorted(output.glob("*.csv.gz"))},
                  timing_only=True, data_payload_captured=False,
                  whole_program_latency_valid=False)
    (output/"manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ratio-num", type=int, required=True, help="replay/source clock ratio numerator")
    parser.add_argument("--ratio-den", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.input, args.output, args.ratio_num, args.ratio_den), indent=2))
