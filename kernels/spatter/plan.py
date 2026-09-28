"""Composable address maps for the five Spatter transfer families.

This is the software contract for the Muon operations in spatter_ops.hpp.
Each transfer is one 64-bit read and one 64-bit write. A future multi-kernel
launcher can materialize the output of one transfer as the next one's input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AddressMap:
    kind: str  # dense, pattern, or nested_pattern
    table: str | None = None
    inner: str | None = None
    delta: int = 0

    def at(self, case: dict, iteration: int, j: int) -> int:
        if self.kind == "dense":
            return j + case["length"] * (iteration % case["wrap"])
        if self.kind == "pattern":
            base = case[self.table][j]
        elif self.kind == "nested_pattern":
            base = case[self.table][case[self.inner][j]]
        else:
            raise ValueError(f"unsupported address map: {self.kind}")
        return base + self.delta * iteration


@dataclass(frozen=True)
class TransferPlan:
    source_array: str
    destination_array: str
    read: AddressMap
    write: AddressMap
    schedule: str  # owner for dense gather output, task for other families

    def as_dict(self) -> dict:
        return asdict(self)


def plan_for(case: dict) -> TransferPlan:
    kind = case["kind"]
    dense = AddressMap("dense")
    if kind == "gather":
        return TransferPlan("sparse", "dense",
                            AddressMap("pattern", "pattern", delta=case["delta"]),
                            dense, "owner")
    if kind == "scatter":
        return TransferPlan("dense", "sparse", dense,
                            AddressMap("pattern", "pattern", delta=case["delta"]),
                            "task")
    if kind == "gs":
        return TransferPlan(
            "sparse_gather", "sparse_scatter",
            AddressMap("pattern", "gather", delta=case["delta_gather"]),
            AddressMap("pattern", "scatter", delta=case["delta_scatter"]),
            "task")
    if kind == "multigather":
        return TransferPlan(
            "sparse", "dense",
            AddressMap("nested_pattern", "pattern", "gather", case["delta"]),
            dense, "owner")
    if kind == "multiscatter":
        return TransferPlan(
            "dense", "sparse", dense,
            AddressMap("nested_pattern", "pattern", "scatter", case["delta"]),
            "task")
    raise ValueError(f"unsupported Spatter family: {kind}")


def execute_reference(case: dict, source: list[int]) -> list[int]:
    """Serial reference for one transfer; only deterministic writes are comparable."""
    plan = case["_plan"]
    if len(source) < case["src_length"]:
        raise ValueError("source is shorter than the address plan requires")
    output = [0] * case["dst_length"]
    for iteration in range(case["count"]):
        for j in range(case["length"]):
            output[plan.write.at(case, iteration, j)] = source[
                plan.read.at(case, iteration, j)]
    return output


def stitch_reference(stages: list[dict], source: list[int]) -> list[int]:
    """Compose transfer stages through dense intermediate arrays in software."""
    for position, case in enumerate(stages):
        if position and len(source) != case["src_length"]:
            raise ValueError("stage output length does not match the next input")
        source = execute_reference(case, source)
    return source
