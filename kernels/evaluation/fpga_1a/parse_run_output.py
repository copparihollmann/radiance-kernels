#!/usr/bin/env python3
"""Extract reportable metrics from one FPGA-1A job's console output.

Usage:  parse_run_output.py JOB_DIR [--stdout PATH] [--uartlog PATH]

Reads ``JOB_DIR/stdout.log`` (falling back to ``JOB_DIR/uartlog`` for the
device-side prints when stdout.log is absent or truncated) and writes a single
JSON object to stdout.  Nothing else is ever written to stdout; diagnostics go
to stderr.

This is the ``parser`` that ``campaign.py``'s per-target ``parser_argv`` hook
invokes.  The harness contract (campaign.py ~l.731) is:

    parsed = subprocess.run(expand(parser_argv, values), cwd=asset_root,
                            capture_output=True, text=True, check=False)
    ... stdout -> result["parsed"] when returncode == 0 and stdout is JSON

so three properties are load bearing and are enforced here:

  * stdout is ALWAYS a single JSON object, never anything else;
  * the exit status is ALWAYS 0, so a parse problem can never be mistaken
    for a parser failure and can never stop a campaign;
  * no file is written, nothing is mutated, and the output depends only on
    the bytes of the logs (deterministic).

Absent is absent: a field that the kernel did not print is simply missing from
the object.  No value is ever defaulted to zero or invented.  Non-finite floats
are never emitted, because ``json.dumps`` would render them as ``NaN``/
``Infinity``, which is not valid JSON and would poison ``result.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

SCHEMA = "radiance-fpga-1a-run-output-parse-v1"

# ---------------------------------------------------------------------------
# A. universal
# ---------------------------------------------------------------------------

# riscv-tests style termination written by the FireSim target.
PASSED_AFTER = re.compile(r"^\*\*\* PASSED \*\*\* after (\d+) cycles\s*$", re.M)
FAILED_CODE = re.compile(
    r"^\*\*\* FAILED \*\*\* \(code = (\d+)\) after (\d+) cycles\s*$", re.M)
FAILED_TOHOST = re.compile(r"^\*\*\* FAILED \*\*\* \(tohost = (\d+)\)\s*$", re.M)

# The device's own verdict (nightly host.cpp prints a bare PASS/FAIL line).
DEVICE_VERDICT = re.compile(r"^(PASS|FAIL)\s*$", re.M)

# nightly_host_report() (lib/include/nightly/host.h) -- also what campaign.py
# itself scrapes into result["core_kernel_cycles"] / result["kernel_cycles"].
CORE_CYCLES = re.compile(
    r"^cluster\s+(\d+)\s+core\s+(\d+):\s+start=(\d+)\s+end=(\d+)\s+cycles=(\d+)\s*$",
    re.M)
KERNEL_CYCLES = re.compile(r"\bkernel_cycles\s*[=:]\s*(\d+)")

# FireSim driver "Emulation Performance Summary".
WALLCLOCK = re.compile(r"^Wallclock Time Elapsed:\s*([0-9.]+)\s*(\S+)\s*$", re.M)
HOST_FREQ = re.compile(r"^Host Frequency:\s*([0-9.]+)\s*(\S+)\s*$", re.M)
TARGET_CYCLES = re.compile(r"^Target Cycles Emulated:\s*(\d+)\s*$", re.M)
EFF_FREQ = re.compile(
    r"^Effective Target Frequency:\s*([0-9.]+)\s*(\S+)\s*$", re.M)
FMR = re.compile(r"^FMR:\s*([0-9.eE+-]+)\s*$", re.M)

# Cyclotron (the functional "model" targets) prints a different shape.
SIM_FINISHED = re.compile(r"^simulation finished after (\d+) cycles\s*$", re.M)
CYCLOTRON_KERNEL = re.compile(r"nightly_kernel_cycles=(\d+)")
CYCLOTRON_CLUSTER = re.compile(
    r"nightly_cluster_cycles cluster=(\d+) cycles=(\d+) start=(\d+) end=(\d+)")
GMEM_CHECK = re.compile(
    r"^Cyclotron: gmem check (.*?): (PASS|FAIL)"
    r"(?: address=(0x[0-9a-fA-F]+))?(?: words=(\d+))?"
    r"(?: fnv64=([0-9a-fA-F]+))?(?: nonzero_u64=(\d+))?\s*$", re.M)

# Cheap extras a few kernels print.
MESH_UTIL_KERNEL = re.compile(r"\bmesh_util_kernel=([0-9.]+)%")
PROFILE_GUARD = re.compile(r"\bprofile_guard_cycles=(\d+)")

# ---------------------------------------------------------------------------
# B. llama_e2e (kernels/nightly/llama_e2e/host.cpp)
# ---------------------------------------------------------------------------

GEOMETRY = re.compile(
    r"^llama_e2e (?P<model>.*?): M=(?P<M>\d+) tokens at positions "
    r"(?P<p0>\d+)\.\.(?P<p1>\d+) \(KV cache (?P<kv>\d+), SK=(?P<sk>\d+)\) "
    r"layers=(?P<layers>\d+) D=(?P<D>\d+) F=(?P<F>\d+) "
    r"heads=(?P<hq>\d+)/(?P<hkv>\d+) hd=(?P<hd>\d+) vocab=(?P<vocab>\d+) "
    r"phases=(?P<phases>\d+) \(init (?P<init>\d+)\) clusters=(?P<clusters>\d+)\s*$",
    re.M)

PHASE_HEADER = "phase layer op cycles [MACs mesh_util]"
PER_OP_HEADER_RE = re.compile(
    r"^== per op \(all layers\): op count cycles share \[mesh_util\]\s*$")
PHASE_ROW = re.compile(
    r"^(\d+) (\d+|-) (\S+) (\d+)(?: (\d+) ([0-9.]+)%)?\s*$")
PER_OP_ROW = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*) (\d+) (\d+) ([0-9.]+)%(?: ([0-9.]+)%)?\s*$")

MODEL_LINE = re.compile(
    r"^model_cycles=(\d+) phase_cycles=(\d+) handshake_cycles=(\d+)\s*$", re.M)
DECODE_LINE = re.compile(r"^decode: (.*)$", re.M)
CYCLES_PER_LAYER_LINE = re.compile(r"^(cycles_per_layer=.*)$", re.M)
KEY_UINT = re.compile(r"\b([a-z_][a-z0-9_]*)=(\d+)\b")
GEMM_MACS_LINE = re.compile(
    r"^gemm_macs=(\d+) mesh_util_model=([0-9.]+)%\s*$", re.M)

# nh_print_cmp() -- bf16 tensor comparison.
BF16_CHECK = re.compile(
    r"^(?P<name>.+?): frob_rel_err=(?P<err>[0-9.]+)% exact=(?P<exact>\d+)/"
    r"(?P<n>\d+) nonfinite=(?P<nonfinite>\d+)\s*$", re.M)
# Byte-exact check (KV cache and friends).
BYTE_CHECK = re.compile(
    r"^(?P<name>.+?): mismatched_bytes=(?P<bad>\d+)/(?P<n>\d+)"
    r"(?: \(allowed (?P<allowed>\d+)\))?"
    r"(?: first_at=(?P<first>\d+) got=(?P<got>0x[0-9a-fA-F]+) "
    r"want=(?P<want>0x[0-9a-fA-F]+))?\s*$", re.M)
LOGITS_LINE = re.compile(r"^logits: nonfinite=(\d+)(?P<rest>.*)$", re.M)
TOP1_EMU = re.compile(r"top1_vs_emulation=(\d+)/(\d+)")
TOP1_HF = re.compile(r"top1_vs_HF=(\d+)/(\d+)")
TOP1_LABEL = re.compile(r"top1_vs_next_token=(\d+)/(\d+) \(HF fp32: (\d+)\)")

FREQ_TO_MHZ = {"hz": 1e-6, "khz": 1e-3, "mhz": 1.0, "ghz": 1e3}
TIME_TO_SECONDS = {"us": 1e-6, "ms": 1e-3, "s": 1.0, "sec": 1.0,
                   "secs": 1.0, "min": 60.0, "m": 60.0, "h": 3600.0}


def _float(text: str) -> float | None:
    """A float, or None when it is not finite (never emit NaN/Infinity)."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _int(text: str) -> int | None:
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _put(out: dict, key: str, value) -> None:
    """Record a field only when it actually has a value."""
    if value is not None:
        out[key] = value


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace")
    except OSError as error:
        print(f"parse_run_output: cannot read {path}: {error}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------


def parse_universal(text: str, out: dict) -> None:
    passed = PASSED_AFTER.findall(text)
    failed_code = FAILED_CODE.findall(text)
    failed_tohost = FAILED_TOHOST.findall(text)
    if passed:
        _put(out, "total_cycles", _int(passed[-1]))
        out["harness_result"] = "PASSED"
    elif failed_code:
        code, cycles = failed_code[-1]
        _put(out, "total_cycles", _int(cycles))
        _put(out, "harness_fail_code", _int(code))
        out["harness_result"] = "FAILED"
    elif failed_tohost:
        _put(out, "harness_fail_tohost", _int(failed_tohost[-1]))
        out["harness_result"] = "FAILED"

    verdicts = DEVICE_VERDICT.findall(text)
    if verdicts:
        out["device_verdict"] = verdicts[-1]

    kernel_cycles = KERNEL_CYCLES.findall(text)
    if kernel_cycles:
        # campaign.py records the LAST occurrence; match it exactly so the
        # side car can be cross-checked against result.json.
        _put(out, "kernel_cycles", _int(kernel_cycles[-1]))
        out["kernel_cycles_occurrences"] = len(kernel_cycles)

    cores = [
        {"cluster": int(c), "core": int(k), "start": int(s), "end": int(e),
         "cycles": int(d)}
        for c, k, s, e, d in CORE_CYCLES.findall(text)
    ]
    if cores:
        out["core_cycles"] = cores
        out["core_cycles_count"] = len(cores)
        out["core_cycles_min"] = min(row["cycles"] for row in cores)
        out["core_cycles_max"] = max(row["cycles"] for row in cores)
        out["core_cycles_span"] = (max(row["end"] for row in cores) -
                                   min(row["start"] for row in cores))


def parse_firesim(text: str, out: dict) -> None:
    summaries = text.count("Emulation Performance Summary")
    if summaries:
        out["firesim_summaries"] = summaries

    wall = WALLCLOCK.findall(text)
    if wall:
        value, unit = wall[-1]
        seconds = _float(value)
        scale = TIME_TO_SECONDS.get(unit.lower())
        if seconds is not None and scale is not None:
            _put(out, "wallclock_seconds", round(seconds * scale, 6))
        else:
            out["wallclock_raw"] = f"{value} {unit}"

    for regex, key in ((HOST_FREQ, "host_frequency_mhz"),
                       (EFF_FREQ, "effective_target_frequency_mhz")):
        found = regex.findall(text)
        if not found:
            continue
        value, unit = found[-1]
        number = _float(value)
        scale = FREQ_TO_MHZ.get(unit.lower())
        if number is not None and scale is not None:
            _put(out, key, round(number * scale, 9))
        else:
            out[key.replace("_mhz", "_raw")] = f"{value} {unit}"

    target = TARGET_CYCLES.findall(text)
    if target:
        _put(out, "target_cycles_emulated", _int(target[-1]))
    fmr = FMR.findall(text)
    if fmr:
        _put(out, "fmr", _float(fmr[-1]))


def parse_cyclotron(text: str, out: dict) -> None:
    finished = SIM_FINISHED.findall(text)
    if finished:
        _put(out, "sim_finished_cycles", _int(finished[-1]))
    kernel = CYCLOTRON_KERNEL.findall(text)
    if kernel:
        _put(out, "cyclotron_kernel_cycles", _int(kernel[-1]))
    clusters = [
        {"cluster": int(c), "cycles": int(cy), "start": int(s), "end": int(e)}
        for c, cy, s, e in CYCLOTRON_CLUSTER.findall(text)
    ]
    if clusters:
        out["cyclotron_cluster_cycles"] = clusters
    checks = []
    for name, status, address, words, fnv, nonzero in GMEM_CHECK.findall(text):
        row = {"name": name, "status": status}
        if address:
            row["address"] = address
        if words:
            row["words"] = int(words)
        if fnv:
            row["fnv64"] = fnv
        if nonzero:
            row["nonzero_u64"] = int(nonzero)
        checks.append(row)
    if checks:
        out["gmem_checks"] = checks
        out["gmem_checks_all_passed"] = all(
            row["status"] == "PASS" for row in checks)


def parse_extras(text: str, out: dict) -> None:
    mesh = MESH_UTIL_KERNEL.findall(text)
    if mesh:
        _put(out, "mesh_util_kernel_pct", _float(mesh[-1]))
    guard = PROFILE_GUARD.findall(text)
    if guard:
        _put(out, "profile_guard_cycles", _int(guard[-1]))


def parse_tables(text: str) -> tuple[list[dict], list[dict]]:
    """The llama_e2e per-phase table and the per-op roll-up.

    Both are scanned line by line inside their own section so that a driver
    line interleaved by FireSim cannot be mistaken for a data row, and so a
    missing header simply yields an empty table.
    """
    lines = text.splitlines()
    phases: list[dict] = []
    per_op: list[dict] = []
    section = None
    for line in lines:
        stripped = line.rstrip()
        if stripped == PHASE_HEADER:
            section = "phase"
            continue
        if PER_OP_HEADER_RE.match(stripped):
            section = "per_op"
            continue
        if stripped.startswith("== ") and not PER_OP_HEADER_RE.match(stripped):
            section = None
            continue
        if section == "phase":
            match = PHASE_ROW.match(stripped)
            if match:
                phase, layer, op, cycles, macs, util = match.groups()
                row = {"phase": int(phase),
                       "layer": None if layer == "-" else int(layer),
                       "op": op, "cycles": int(cycles)}
                if macs is not None:
                    row["macs"] = int(macs)
                    _put(row, "mesh_util_pct", _float(util))
                phases.append(row)
        elif section == "per_op":
            match = PER_OP_ROW.match(stripped)
            if match:
                op, count, cycles, share, util = match.groups()
                row = {"op": op, "count": int(count), "cycles": int(cycles)}
                _put(row, "share_pct", _float(share))
                if util is not None:
                    _put(row, "mesh_util_pct", _float(util))
                per_op.append(row)
            elif stripped:
                # The roll-up is a contiguous block; anything else ends it.
                section = None
    return phases, per_op


def parse_llama(text: str, out: dict) -> None:
    geometry = list(GEOMETRY.finditer(text))
    if geometry:
        g = geometry[-1].groupdict()
        out["model_geometry"] = {
            "model": g["model"], "M": int(g["M"]),
            "position_first": int(g["p0"]), "position_last": int(g["p1"]),
            "kv_cache": int(g["kv"]), "sk": int(g["sk"]),
            "layers": int(g["layers"]), "d": int(g["D"]), "f": int(g["F"]),
            "heads_q": int(g["hq"]), "heads_kv": int(g["hkv"]),
            "hd": int(g["hd"]), "vocab": int(g["vocab"]),
            "phases": int(g["phases"]), "init_phases": int(g["init"]),
            "clusters": int(g["clusters"]),
        }

    phases, per_op = parse_tables(text)
    if phases:
        out["phase_table"] = phases
        out["phase_table_rows"] = len(phases)
    if per_op:
        out["per_op"] = per_op
        out["per_op_rows"] = len(per_op)

    model = MODEL_LINE.findall(text)
    if model:
        total, busy, handshake = model[-1]
        _put(out, "model_cycles", _int(total))
        _put(out, "phase_cycles", _int(busy))
        _put(out, "handshake_cycles", _int(handshake))

    decode = DECODE_LINE.findall(text)
    if decode:
        fields = dict(KEY_UINT.findall(decode[-1]))
        out["decode"] = True
        for key in ("batch", "steps", "verify_tokens_per_step",
                    "cycles_per_step", "cycles_per_layer", "cycles_per_token"):
            if key in fields:
                prefix = "decode_" if key in ("batch", "steps",
                                              "verify_tokens_per_step") else ""
                _put(out, f"{prefix}{key}", _int(fields[key]))
    else:
        per_layer = CYCLES_PER_LAYER_LINE.findall(text)
        if per_layer:
            fields = dict(KEY_UINT.findall(per_layer[-1]))
            for key in ("cycles_per_layer", "cycles_per_token"):
                if key in fields:
                    _put(out, key, _int(fields[key]))

    gemm = GEMM_MACS_LINE.findall(text)
    if gemm:
        macs, util = gemm[-1]
        _put(out, "gemm_macs", _int(macs))
        _put(out, "mesh_util_model_pct", _float(util))


def parse_correctness(text: str, out: dict) -> None:
    bf16 = []
    for match in BF16_CHECK.finditer(text):
        g = match.groupdict()
        row = {"name": g["name"], "exact": int(g["exact"]), "n": int(g["n"]),
               "nonfinite": int(g["nonfinite"])}
        _put(row, "frob_rel_err_pct", _float(g["err"]))
        bf16.append(row)
    if bf16:
        out["bf16_checks"] = bf16
        out["bf16_checks_bit_exact"] = all(
            row["exact"] == row["n"] for row in bf16)
        out["bf16_checks_nonfinite_total"] = sum(
            row["nonfinite"] for row in bf16)

    byte_checks = []
    for match in BYTE_CHECK.finditer(text):
        g = match.groupdict()
        row = {"name": g["name"], "mismatched_bytes": int(g["bad"]),
               "n": int(g["n"])}
        for key, field in (("allowed", "allowed"), ("first", "first_at")):
            if g[key] is not None:
                row[field] = int(g[key])
        for key in ("got", "want"):
            if g[key] is not None:
                row[key] = g[key]
        byte_checks.append(row)
    if byte_checks:
        out["byte_checks"] = byte_checks
        out["byte_checks_mismatched_bytes_total"] = sum(
            row["mismatched_bytes"] for row in byte_checks)

    logits = list(LOGITS_LINE.finditer(text))
    if logits:
        match = logits[-1]
        _put(out, "logits_nonfinite", _int(match.group(1)))
        rest = match.group("rest")
        emu = TOP1_EMU.search(rest)
        if emu:
            out["logits_top1_vs_emulation"] = int(emu.group(1))
            out["logits_top1_vs_emulation_total"] = int(emu.group(2))
        hf = TOP1_HF.search(rest)
        if hf:
            out["logits_top1_vs_hf"] = int(hf.group(1))
            out["logits_top1_vs_hf_total"] = int(hf.group(2))
        label = TOP1_LABEL.search(rest)
        if label:
            out["logits_top1_vs_next_token"] = int(label.group(1))
            out["logits_top1_vs_next_token_total"] = int(label.group(2))
            out["logits_top1_vs_next_token_hf_fp32"] = int(label.group(3))


def parse_text(text: str) -> dict:
    """Every field this parser knows how to extract, from one blob of console
    output.  Fields the text does not contain are simply absent."""
    out: dict = {}
    parse_universal(text, out)
    parse_firesim(text, out)
    parse_cyclotron(text, out)
    parse_extras(text, out)
    parse_llama(text, out)
    parse_correctness(text, out)
    return out


def parse_job_dir(job_dir: Path, stdout_path: Path | None = None,
                  uartlog_path: Path | None = None) -> dict:
    """Parse one job/result directory.  Never raises for log-content reasons."""
    stdout_path = stdout_path or (job_dir / "stdout.log")
    uartlog_path = uartlog_path or (job_dir / "uartlog")

    result: dict = {"schema": SCHEMA, "sources": [], "parse_errors": []}

    stdout_text = read_text(stdout_path) if stdout_path.is_file() else None
    if stdout_text is not None:
        result["sources"].append(stdout_path.name)

    text = stdout_text or ""
    fields = parse_text(text) if text else {}

    # uartlog is the same console stream the FireSim driver mirrors to disk.
    # It is used only as a FALLBACK for the device-side prints, never merged
    # on top of stdout.log, because merging would double-count every per-core
    # row.  A field already found in stdout.log is kept.
    if uartlog_path.is_file():
        uart_text = read_text(uartlog_path)
        if uart_text:
            uart_fields = parse_text(uart_text)
            added = False
            for key, value in uart_fields.items():
                if key not in fields:
                    fields[key] = value
                    added = True
            if added or not stdout_text:
                result["sources"].append(uartlog_path.name)

    if not result["sources"]:
        result["parse_errors"].append(
            f"no readable console log in {job_dir}")

    result.update(fields)
    if not result["parse_errors"]:
        del result["parse_errors"]
    return result


def main(argv: list[str] | None = None) -> int:
    out: dict
    try:
        parser = argparse.ArgumentParser(
            description="Extract reportable metrics from one job's logs.")
        parser.add_argument("job_dir", help="job/result directory")
        parser.add_argument("--stdout", dest="stdout_path",
                            help="override the stdout.log path")
        parser.add_argument("--uartlog", dest="uartlog_path",
                            help="override the uartlog path")
        args = parser.parse_args(argv)
        out = parse_job_dir(
            Path(args.job_dir),
            Path(args.stdout_path) if args.stdout_path else None,
            Path(args.uartlog_path) if args.uartlog_path else None,
        )
    except SystemExit:
        # argparse already explained itself on stderr; still emit valid JSON so
        # campaign.py records a usable (if empty) "parsed" object.
        out = {"schema": SCHEMA, "parse_errors": ["bad parser arguments"]}
    except BaseException as error:  # never fail the job -- see module docstring
        out = {"schema": SCHEMA,
               "parse_errors": [f"{type(error).__name__}: {error}"]}
    try:
        sys.stdout.write(json.dumps(out, sort_keys=True) + "\n")
    except (TypeError, ValueError) as error:
        sys.stdout.write(json.dumps(
            {"schema": SCHEMA,
             "parse_errors": [f"unserializable: {error}"]}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
