"""Unit tests for the FPGA-1A run-output parser.

The fixtures under testdata/parse_run_output/ are:

  hpc-firesim-stdout.log        captured verbatim from
                                runs/fpga-1a-stream-scale-2sm-hbm-v1
  cyclotron-model-stdout.log    captured verbatim from a model-kind job
  llama-e2e-decode-stdout.log   built from the print statements in
  llama-e2e-prefill-stdout.log  kernels/nightly/llama_e2e/host.cpp (which is
                                authoritative for our ELFs) and cross-checked
                                against the hardware owner's archived
                                llama_e2e logs; no llama_e2e hardware output
                                exists in our corpus yet.
  failed-tohost-stdout.log      a synthetic failure shape
  quiet-stdout.log              a kernel that prints none of the fields

  compute-btree-stdout.log      captured verbatim from the completed jobs of
  compute-dot-1m-stdout.log     runs/hpc-v2-compute-instrumented-v1 (the
  compute-gemm-bf16-512-        2SM_DRAM_v2 half), the first campaign to carry
    stdout.log                  the a8b6bcd device-reported compute counters.
  compute-hotspot3d-stdout.log  btree has executed_ops == useful_ops with
                                executed_ops_distinguished=0; dot-1m is one of
                                the three kernels that really can tell the two
                                apart; gemm is the only kernel on the
                                MX/Gemmini datapath (512 MAC/cycle, not 64
                                ops/cycle) and the only one that publishes
                                phase stamps; hotspot3d is the four-launch
                                case.
"""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parse_run_output import SCHEMA, main, parse_job_dir, parse_text

HERE = Path(__file__).resolve().parent
DATA = HERE / "testdata" / "parse_run_output"
PARSER = HERE / "parse_run_output.py"


def fixture(name: str) -> str:
    return (DATA / name).read_text()


def parsed(name: str) -> dict:
    return parse_text(fixture(name))


def job_dir(stdout: str | None = None, uartlog: str | None = None,
            stack: list | None = None) -> Path:
    tmp = tempfile.TemporaryDirectory()
    if stack is not None:
        stack.append(tmp)
    root = Path(tmp.name)
    if stdout is not None:
        (root / "stdout.log").write_text(stdout)
    if uartlog is not None:
        (root / "uartlog").write_text(uartlog)
    return root


class UniversalFields(unittest.TestCase):
    def test_hpc_firesim_log(self):
        out = parsed("hpc-firesim-stdout.log")
        self.assertEqual(out["total_cycles"], 3450200567)
        self.assertEqual(out["harness_result"], "PASSED")
        self.assertEqual(out["device_verdict"], "PASS")
        self.assertEqual(out["kernel_cycles"], 16789)
        self.assertEqual(out["kernel_cycles_occurrences"], 1)
        self.assertEqual(out["wallclock_seconds"], 12.0)
        self.assertEqual(out["host_frequency_mhz"], 10.0)
        self.assertEqual(out["target_cycles_emulated"], 3450200567)
        self.assertEqual(out["effective_target_frequency_mhz"], 286.429)
        self.assertEqual(out["fmr"], 0.03)

    def test_hpc_per_core_rows(self):
        out = parsed("hpc-firesim-stdout.log")
        self.assertEqual(out["core_cycles_count"], 4)
        self.assertEqual(out["core_cycles"][0],
                         {"cluster": 0, "core": 0, "start": 1468,
                          "end": 17925, "cycles": 16457})
        self.assertEqual(out["core_cycles_max"], 16788)
        self.assertEqual(out["core_cycles_min"], 16455)
        # latest end - earliest start.  This equals the scraped kernel_cycles
        # for kernels that report via nightly_host_report(), but it is a
        # derived convenience, not an invariant: some kernels print their own
        # aggregate kernel_cycles=, and nightly_host_report()'s t1 - t0 is
        # uint32 and can wrap.  Both numbers are reported, neither is
        # normalised away.
        self.assertEqual(out["core_cycles_span"], out["kernel_cycles"])

    def test_failed_tohost_has_no_total_cycles(self):
        out = parsed("failed-tohost-stdout.log")
        self.assertEqual(out["harness_result"], "FAILED")
        self.assertEqual(out["harness_fail_tohost"], 3)
        self.assertNotIn("total_cycles", out)
        self.assertNotIn("harness_fail_code", out)

    def test_failed_with_code_keeps_cycles(self):
        out = parse_text("*** FAILED *** (code = 7) after 991 cycles\n")
        self.assertEqual(out["harness_fail_code"], 7)
        self.assertEqual(out["total_cycles"], 991)
        self.assertEqual(out["harness_result"], "FAILED")

    def test_frequency_units_are_normalised_not_assumed(self):
        out = parse_text("Host Frequency: 2.500 GHz\n"
                         "Effective Target Frequency: 125.000 KHz\n")
        self.assertEqual(out["host_frequency_mhz"], 2500.0)
        self.assertEqual(out["effective_target_frequency_mhz"], 0.125)

    def test_unknown_units_are_kept_raw_not_invented(self):
        out = parse_text("Host Frequency: 3.000 furlongs\n")
        self.assertNotIn("host_frequency_mhz", out)
        self.assertEqual(out["host_frequency_raw"], "3.000 furlongs")

    def test_last_occurrence_wins_like_campaign_py(self):
        out = parse_text("kernel_cycles=10\nkernel_cycles=20\n")
        self.assertEqual(out["kernel_cycles"], 20)
        self.assertEqual(out["kernel_cycles_occurrences"], 2)

    def test_nightly_kernel_cycles_is_not_kernel_cycles(self):
        # campaign.py's own CYCLES regex is word-anchored the same way.
        out = parse_text("Cyclotron: nightly_kernel_cycles=182607\n")
        self.assertNotIn("kernel_cycles", out)
        self.assertEqual(out["cyclotron_kernel_cycles"], 182607)


class CyclotronFields(unittest.TestCase):
    def test_model_log(self):
        out = parsed("cyclotron-model-stdout.log")
        self.assertEqual(out["sim_finished_cycles"], 182686)
        self.assertEqual(out["cyclotron_kernel_cycles"], 182607)
        self.assertEqual(len(out["cyclotron_cluster_cycles"]), 2)
        self.assertEqual(out["cyclotron_cluster_cycles"][1],
                         {"cluster": 1, "cycles": 182607, "start": 53,
                          "end": 182660})
        self.assertTrue(out["gmem_checks_all_passed"])
        self.assertEqual(out["gmem_checks"][0]["fnv64"], "b6c36c1bfdb82e5e")
        self.assertEqual(out["gmem_checks"][0]["words"], 128)
        # a model job has no FireSim driver summary at all
        self.assertNotIn("fmr", out)
        self.assertNotIn("total_cycles", out)


class LlamaDecodeFields(unittest.TestCase):
    def setUp(self):
        self.out = parsed("llama-e2e-decode-stdout.log")

    def test_model_geometry(self):
        self.assertEqual(self.out["model_geometry"], {
            "model": "llama7b decode", "M": 128, "position_first": 1024,
            "position_last": 1151, "kv_cache": 1024, "sk": 1280, "layers": 1,
            "d": 4096, "f": 11008, "heads_q": 32, "heads_kv": 32, "hd": 128,
            "vocab": 32000, "phases": 10, "init_phases": 4, "clusters": 2})

    def test_model_roll_up(self):
        self.assertEqual(self.out["model_cycles"], 44104761)
        self.assertEqual(self.out["phase_cycles"], 44825323)
        self.assertEqual(self.out["handshake_cycles"], 0)
        self.assertEqual(self.out["gemm_macs"], 17777216000)
        self.assertEqual(self.out["mesh_util_model_pct"], 78.65)

    def test_decode_line(self):
        self.assertTrue(self.out["decode"])
        self.assertEqual(self.out["decode_batch"], 16)
        self.assertEqual(self.out["decode_steps"], 2)
        self.assertEqual(self.out["decode_verify_tokens_per_step"], 4)
        self.assertEqual(self.out["cycles_per_step"], 22052380)
        self.assertEqual(self.out["cycles_per_layer"], 12663980)
        self.assertEqual(self.out["cycles_per_token"], 344568)

    def test_phase_table(self):
        rows = self.out["phase_table"]
        self.assertEqual(self.out["phase_table_rows"], 10)
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0], {"phase": 0, "layer": None, "op": "init",
                                   "cycles": 19771587})
        self.assertEqual(rows[5], {"phase": 5, "layer": 0, "op": "qkv",
                                   "cycles": 6757246, "macs": 3221225472,
                                   "mesh_util_pct": 93.10})
        # a row without MACs must not acquire a zero
        self.assertNotIn("macs", rows[4])
        self.assertNotIn("mesh_util_pct", rows[4])

    def test_per_op_roll_up(self):
        rows = {row["op"]: row for row in self.out["per_op"]}
        self.assertEqual(self.out["per_op_rows"], 6)
        self.assertEqual(rows["qk"], {"op": "qk", "count": 1,
                                      "cycles": 12530366, "share_pct": 28.41,
                                      "mesh_util_pct": 35.56})
        self.assertEqual(rows["final"], {"op": "final", "count": 1,
                                         "cycles": 1067851, "share_pct": 2.42})
        self.assertNotIn("mesh_util_pct", rows["final"])

    def test_kernel_cycles_alias_is_the_last_one(self):
        # host.cpp prints nightly_host_report()'s kernel_cycles first and our
        # model_cycles alias (commit bd0d178) second; campaign.py keeps the
        # last, so the parser must agree.
        self.assertEqual(self.out["kernel_cycles_occurrences"], 2)
        self.assertEqual(self.out["kernel_cycles"], self.out["model_cycles"])
        self.assertEqual(self.out["core_cycles_span"], 364864956)

    def test_correctness_fields(self):
        self.assertEqual(self.out["logits_nonfinite"], 0)
        self.assertEqual(self.out["logits_top1_vs_emulation"], 128)
        self.assertEqual(self.out["logits_top1_vs_emulation_total"], 128)
        self.assertEqual(self.out["logits_top1_vs_hf"], 120)
        self.assertEqual(self.out["logits_top1_vs_next_token"], 96)
        self.assertEqual(self.out["logits_top1_vs_next_token_hf_fp32"], 99)

    def test_tensor_checks(self):
        bf16 = {row["name"]: row for row in self.out["bf16_checks"]}
        self.assertEqual(bf16["o_proj vs emulation"]["frob_rel_err_pct"], 0.12)
        self.assertEqual(bf16["o_proj vs emulation"]["exact"], 4095)
        self.assertEqual(bf16["o_proj vs emulation"]["n"], 4096)
        self.assertFalse(self.out["bf16_checks_bit_exact"])
        self.assertEqual(self.out["bf16_checks_nonfinite_total"], 0)

    def test_byte_checks(self):
        rows = {row["name"]: row for row in self.out["byte_checks"]}
        self.assertEqual(rows["k_cache"],
                         {"name": "k_cache", "mismatched_bytes": 0,
                          "n": 655360})
        self.assertEqual(rows["v_cache"]["mismatched_bytes"], 3)
        self.assertEqual(rows["v_cache"]["allowed"], 8)
        self.assertEqual(rows["v_cache"]["first_at"], 17)
        self.assertEqual(rows["v_cache"]["got"], "0x3f")
        self.assertEqual(rows["v_cache"]["want"], "0x40")
        self.assertEqual(self.out["byte_checks_mismatched_bytes_total"], 3)


class LlamaPrefillFields(unittest.TestCase):
    def test_non_decode_cycles_per_layer_line(self):
        out = parsed("llama-e2e-prefill-stdout.log")
        self.assertNotIn("decode", out)
        self.assertNotIn("cycles_per_step", out)
        self.assertEqual(out["cycles_per_layer"], 4560277)
        self.assertEqual(out["cycles_per_token"], 304901)
        self.assertEqual(out["model_geometry"]["model"], "llama7b")
        self.assertEqual(out["per_op_rows"], 3)
        self.assertEqual(out["phase_table_rows"], 6)


class ComputeCounters(unittest.TestCase):
    """The device-reported COMPUTE axis (nightly/host.h, added by a8b6bcd).

    Every number asserted here was printed by the hardware and is copied out of
    the captured fixture; none of it is recomputed by the parser.
    """

    def test_btree_compute_block(self):
        out = parsed("compute-btree-stdout.log")
        self.assertEqual(out["compute"], {
            "op_unit": "integer-compare-add",
            "datapath": "scalar-simt",
            "useful_ops": 193986560,
            "executed_ops": 193986560,
            "executed_ops_distinguished": 0,
            "peak_ops_per_cycle": 64,
            "util_bp": 164,
            "util_pct": 1.64,
            "executed_util_bp": 164,
            "executed_util_pct": 1.64,
            "window_cycles": 184187406,
        })

    def test_executed_equals_useful_when_undistinguished(self):
        compute = parsed("compute-btree-stdout.log")["compute"]
        self.assertEqual(compute["executed_ops_distinguished"], 0)
        self.assertEqual(compute["executed_ops"], compute["useful_ops"])
        self.assertEqual(compute["executed_util_bp"], compute["util_bp"])

    def test_compute_window_is_the_scraped_kernel_cycles(self):
        for name in ("compute-btree-stdout.log", "compute-dot-1m-stdout.log",
                     "compute-gemm-bf16-512-stdout.log"):
            with self.subTest(fixture=name):
                out = parsed(name)
                self.assertEqual(out["compute"]["window_cycles"],
                                 out["kernel_cycles"])

    def test_btree_detail(self):
        out = parsed("compute-btree-stdout.log")
        self.assertEqual(out["compute_detail"],
                         {"btree_lookups": 524288,
                          "btree_comparisons_per_lookup": 368})
        # 524288 * 368 = 192937984 comparisons, plus the adds the op unit
        # folds in; the parser does not recompute it, it only reports both.
        self.assertNotIn("compute_launches", out)
        self.assertNotIn("phase_stamps", out)

    def test_dot_distinguishes_executed_from_useful(self):
        out = parsed("compute-dot-1m-stdout.log")
        compute = out["compute"]
        self.assertEqual(compute["op_unit"], "flop")
        self.assertEqual(compute["executed_ops_distinguished"], 1)
        self.assertEqual(compute["useful_ops"], 2097151)
        self.assertEqual(compute["executed_ops"], 2097662)
        self.assertGreater(compute["executed_ops"], compute["useful_ops"])
        detail = out["compute_detail"]
        self.assertEqual(detail["dot_device_fp32_multiplies"], 1048576)
        self.assertEqual(detail["dot_device_fp32_adds"], 1049086)
        self.assertEqual(detail["dot_threads_per_threadblock"], 256)
        self.assertEqual(detail["dot_host_software_float_adds"], 1)
        # the device's own two halves add up to the executed total it printed
        self.assertEqual(detail["dot_device_fp32_multiplies"] +
                         detail["dot_device_fp32_adds"],
                         compute["executed_ops"])

    def test_gemm_is_the_mx_gemmini_datapath(self):
        out = parsed("compute-gemm-bf16-512-stdout.log")
        compute = out["compute"]
        self.assertEqual(compute["op_unit"], "mac")
        self.assertEqual(compute["datapath"], "mx-gemmini")
        # 256 MAC/cycle/cluster x 2 clusters, not the 64 scalar ops/cycle the
        # other twenty report.
        self.assertEqual(compute["peak_ops_per_cycle"], 512)
        self.assertEqual(compute["useful_ops"], 134217728)
        self.assertEqual(compute["util_bp"], 7971)
        self.assertEqual(compute["util_pct"], 79.71)
        detail = out["compute_detail"]
        self.assertEqual(detail["useful_flop_ops"], 2 * compute["useful_ops"])
        self.assertEqual(detail["peak_flop_per_cycle"],
                         2 * compute["peak_ops_per_cycle"])
        self.assertEqual(detail["gemm_c0_fill_cycles"], 36693)
        self.assertEqual(detail["gemm_c0_compute_cycles"], 288583)
        self.assertEqual(detail["gemm_c0_drain_cycles"], 3449)
        self.assertEqual(detail["gemm_c0_compute_util_bp"], 9083)

    def test_gemm_phase_stamps(self):
        out = parsed("compute-gemm-bf16-512-stdout.log")
        self.assertEqual(out["phase_stamps_count"], 4)
        self.assertEqual(out["phase_stamps"][:2], [
            {"cluster": 0, "index": 0, "cycles": 38114},
            {"cluster": 0, "index": 1, "cycles": 326697},
        ])
        # cluster 1 published zeros; they are reported as printed, not dropped
        # and not turned into an absence.
        self.assertEqual([row["cycles"] for row in out["phase_stamps"][2:]],
                         [0, 0])

    def test_hotspot3d_launches_are_a_list_not_n_flat_keys(self):
        out = parsed("compute-hotspot3d-stdout.log")
        self.assertEqual(out["compute_launches_count"], 4)
        self.assertEqual([row["index"] for row in out["compute_launches"]],
                         [0, 1, 2, 3])
        for row in out["compute_launches"]:
            self.assertEqual(sorted(row),
                             ["compute_util_bp", "cycles", "executed_ops",
                              "index", "useful_ops"])
        for key in out:
            self.assertFalse(key.startswith("launch_"), key)
        self.assertEqual(out["compute_detail"]["hotspot3d_launches"], 4)
        self.assertEqual(out["compute_detail"]["hotspot3d_ops_per_update"], 17)

    def test_hotspot3d_launch_totals_reconstruct_the_aggregate(self):
        out = parsed("compute-hotspot3d-stdout.log")
        compute = out["compute"]
        # This is the field that fixes hotspot3d's roofline: the aggregate
        # window fuses four launches, and the per-launch split recovers them.
        self.assertEqual(out["compute_launches_cycles_total"],
                         compute["window_cycles"])
        self.assertEqual(out["compute_launches_useful_ops_total"],
                         compute["useful_ops"])
        self.assertEqual(out["compute_launches_executed_ops_total"],
                         compute["executed_ops"])
        per_launch = {row["useful_ops"] for row in out["compute_launches"]}
        self.assertEqual(len(per_launch), 1)
        self.assertEqual(per_launch.pop() * 4, compute["useful_ops"])

    def test_pre_instrumentation_log_has_no_compute_fields(self):
        out = parsed("hpc-firesim-stdout.log")
        for key in ("compute", "compute_detail", "compute_launches",
                    "compute_launches_count", "phase_stamps"):
            self.assertNotIn(key, out)

    def test_counters_must_own_their_whole_line(self):
        # The FireSim driver dumps a plusargs line full of key=value pairs and
        # the memory model prints its own; nothing embedded in a longer line
        # may be mistaken for a counter.
        out = parse_text(
            "+autocounter-readrate=0 +netbw0=200 useful_ops=5\n"
            "note: stream_fp_ops=7 here\n"
            "[FASED] configuration register dramTimings_tCAS=14\n"
            "launch_0_cycles=11 trailing words\n")
        self.assertNotIn("compute", out)
        self.assertNotIn("compute_detail", out)
        self.assertNotIn("compute_launches", out)

    def test_partial_block_reports_only_what_was_printed(self):
        out = parse_text("useful_ops=10\ncompute_window_cycles=100\n")
        self.assertEqual(out["compute"],
                         {"useful_ops": 10, "window_cycles": 100})
        self.assertNotIn("executed_ops", out["compute"])
        self.assertNotIn("util_pct", out["compute"])

    def test_launch_totals_need_every_launch_to_report_the_field(self):
        out = parse_text("launch_0_cycles=10\nlaunch_0_useful_ops=4\n"
                         "launch_1_cycles=20\n")
        self.assertEqual(out["compute_launches_count"], 2)
        self.assertEqual(out["compute_launches_cycles_total"], 30)
        self.assertNotIn("compute_launches_useful_ops_total", out)

    def test_last_occurrence_wins(self):
        out = parse_text("useful_ops=1\ncompute_op_unit=flop\n"
                         "useful_ops=2\ncompute_op_unit=mac\n")
        self.assertEqual(out["compute"]["useful_ops"], 2)
        self.assertEqual(out["compute"]["op_unit"], "mac")

    def test_carriage_returns_from_the_uart_do_not_hide_a_counter(self):
        out = parse_text("useful_ops=5\r\ncompute_op_unit=flop\r\n"
                         "btree_lookups=2\r\nlaunch_0_cycles=3\r\n"
                         "phase_stamp_c0_0=9\r\n")
        self.assertEqual(out["compute"], {"useful_ops": 5, "op_unit": "flop"})
        self.assertEqual(out["compute_detail"], {"btree_lookups": 2})
        self.assertEqual(out["compute_launches"], [{"index": 0, "cycles": 3}])
        self.assertEqual(out["phase_stamps_count"], 1)

    def test_launch_record_index_cannot_be_overwritten(self):
        out = parse_text("launch_2_cycles=10\nlaunch_2_index=99\n")
        self.assertEqual(out["compute_launches"],
                         [{"index": 2, "cycles": 10}])

    def test_detail_object_is_bounded(self):
        text = "".join(f"stream_k{i}={i}\n" for i in range(4096))
        out = parse_text(text)
        self.assertEqual(len(out["compute_detail"]), 256)


class Totality(unittest.TestCase):
    """A parser failure must never be able to fail a job."""

    def setUp(self):
        self.stack = []

    def tearDown(self):
        for tmp in self.stack:
            tmp.cleanup()

    def test_quiet_kernel_yields_an_object_without_the_fields(self):
        out = parsed("quiet-stdout.log")
        for key in ("total_cycles", "kernel_cycles", "fmr", "model_cycles",
                    "core_cycles", "logits_nonfinite"):
            self.assertNotIn(key, out)

    def test_missing_directory(self):
        out = parse_job_dir(Path("/nonexistent/job/dir"))
        self.assertEqual(out["schema"], SCHEMA)
        self.assertEqual(out["sources"], [])
        self.assertTrue(out["parse_errors"])

    def test_empty_log(self):
        out = parse_job_dir(job_dir(stdout="", stack=self.stack))
        self.assertEqual(out["schema"], SCHEMA)

    def test_binary_garbage_does_not_crash(self):
        root = job_dir(stack=self.stack)
        (root / "stdout.log").write_bytes(bytes(range(256)) * 64)
        out = parse_job_dir(root)
        self.assertEqual(out["schema"], SCHEMA)

    def test_uartlog_is_a_fallback_and_never_double_counts(self):
        text = fixture("hpc-firesim-stdout.log")
        both = parse_job_dir(job_dir(stdout=text, uartlog=text,
                                     stack=self.stack))
        self.assertEqual(both["core_cycles_count"], 4)
        self.assertEqual(both["kernel_cycles_occurrences"], 1)
        self.assertEqual(both["sources"], ["stdout.log"])

    def test_uartlog_used_when_stdout_is_absent(self):
        out = parse_job_dir(job_dir(uartlog=fixture("hpc-firesim-stdout.log"),
                                    stack=self.stack))
        self.assertEqual(out["sources"], ["uartlog"])
        self.assertEqual(out["kernel_cycles"], 16789)

    def test_determinism(self):
        root = job_dir(stdout=fixture("llama-e2e-decode-stdout.log"),
                       stack=self.stack)
        first = json.dumps(parse_job_dir(root), sort_keys=True)
        second = json.dumps(parse_job_dir(root), sort_keys=True)
        self.assertEqual(first, second)

    def test_main_always_exits_zero_and_prints_one_json_object(self):
        root = job_dir(stdout=fixture("hpc-firesim-stdout.log"),
                       stack=self.stack)
        for argv in ([str(root)], [], ["/nonexistent"],
                     [str(root), "--stdout", "/nonexistent"]):
            with self.subTest(argv=argv):
                completed = subprocess.run(
                    [sys.executable, "-I", str(PARSER), *argv],
                    capture_output=True, text=True, check=False)
                self.assertEqual(completed.returncode, 0)
                value = json.loads(completed.stdout)
                self.assertIsInstance(value, dict)
                self.assertEqual(value["schema"], SCHEMA)

    def test_output_is_strict_json_without_nan_or_infinity(self):
        for name in sorted(p.name for p in DATA.iterdir()):
            with self.subTest(fixture=name):
                text = json.dumps(parse_text(fixture(name)), sort_keys=True)
                self.assertNotIn("NaN", text)
                self.assertNotIn("Infinity", text)
                json.loads(text, parse_constant=_reject_constant)

    def test_no_side_effects(self):
        root = job_dir(stdout=fixture("hpc-firesim-stdout.log"),
                       stack=self.stack)
        before = sorted(p.name for p in root.iterdir())
        parse_job_dir(root)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(main([str(root)]), 0)
        self.assertEqual(json.loads(buffer.getvalue())["schema"], SCHEMA)
        self.assertEqual(sorted(p.name for p in root.iterdir()), before)


class ParserArgvContract(unittest.TestCase):
    """Pin campaign.py's parser_argv contract (campaign.py ~l.731).

    Entirely host side: the target is kind="model", so no FireSim code path,
    no host lock, no PCIe guard and no hardware is touched.  The "runner" is a
    python script that prints a canned log.
    """

    def build_campaign(self, root: Path, parser_argv, parser_name="parser.py",
                       parser_body=None):
        import campaign

        assets = root / "assets"
        assets.mkdir()
        (assets / "model.yaml").write_text("model: test\n")
        (assets / "runner.py").write_text(
            "print('cluster 0 core 0: start=1 end=1201 cycles=1200')\n"
            "print('cluster 0 core 1: start=2 end=1235 cycles=1233')\n"
            "print('kernel_cycles=1234')\n"
            "print('PASS')\n"
            "print('*** PASSED *** after 99999 cycles')\n"
        )
        (assets / parser_name).write_text(
            parser_body if parser_body is not None
            else PARSER.read_text())
        elf = root / "kernel.soc.elf"
        elf.write_bytes(b"elf")
        build_path = root / "build.json"
        build_path.write_text(json.dumps({
            "hardware_rebuilt": False, "no_vpu": True,
            "workloads": [{
                "name": "stream--copy--1sm", "family": "stream",
                "operation": "copy", "sm_count": 1, "loader": "logical",
                "elf": str(elf), "elf_sha256": campaign.sha256(elf),
                "status": "built"}]}))
        target = {
            "id": "one", "kind": "model", "sm_count": 1, "memory": "hbm",
            "loader": "logical", "model_config": "model.yaml",
            "runner": "runner.py", "parser": parser_name,
            "run_argv": [sys.executable, "-I", "{runner}", "{elf}",
                         "{model_config}"],
            "parser_argv": parser_argv,
        }
        asset_path = root / "assets.json"
        asset_path.write_text(json.dumps({
            "schema": "radiance-fpga-1a-assets-v1", "targets": [target]}))
        run_root = root / "runs"
        plan = campaign.make_plan(build_path, asset_path, assets, run_root,
                                  warmups=0, repetitions=1,
                                  allow_incomplete=True)
        return campaign, run_root / "plan.json", Path(plan["jobs"][0]["result_dir"])

    def run_one(self, parser_argv, **kwargs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        campaign, plan_path, result_dir = self.build_campaign(
            root, parser_argv, **kwargs)
        campaign.execute(plan_path, rerun_failed=False)
        return json.loads((result_dir / "result.json").read_text()), result_dir

    def test_parsed_stdout_is_merged_into_result(self):
        result, result_dir = self.run_one(
            [sys.executable, "-I", "{parser}", "{out}"])
        self.assertEqual(result["status"], "passed")
        self.assertNotIn("parser_error", result)
        self.assertEqual(result["parsed"]["schema"], SCHEMA)
        self.assertEqual(result["parsed"]["kernel_cycles"], 1234)
        self.assertEqual(result["parsed"]["total_cycles"], 99999)
        # and the harness keeps the raw parser streams beside the result
        self.assertTrue((result_dir / "parser.stdout").is_file())
        self.assertTrue((result_dir / "parser.stderr").is_file())

    def test_nonzero_parser_exit_records_an_error_but_passes_the_job(self):
        result, _ = self.run_one(
            [sys.executable, "-I", "{parser}", "{out}"],
            parser_body="import sys; sys.exit(3)\n")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["parser_error"], "parser exited 3")
        self.assertNotIn("parsed", result)

    def test_non_json_stdout_records_an_error_but_passes_the_job(self):
        result, _ = self.run_one(
            [sys.executable, "-I", "{parser}", "{out}"],
            parser_body="print('not json')\n")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["parser_error"], "parser stdout is not JSON")
        self.assertNotIn("parsed", result)

    def test_unlaunchable_parser_aborts_the_campaign_before_result_json(self):
        # THE risk: subprocess.run() is not guarded, and result.json is written
        # after the parser block, so an argv that cannot be launched loses the
        # job's result entirely and stops the whole campaign.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        campaign, plan_path, result_dir = self.build_campaign(
            root, ["definitely-not-on-path-9f3a", "{parser}"])
        with self.assertRaises(FileNotFoundError):
            campaign.execute(plan_path, rerun_failed=False)
        self.assertFalse((result_dir / "result.json").exists())
        self.assertTrue((result_dir / "stdout.log").is_file())

    def test_unknown_placeholder_also_aborts_before_result_json(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        campaign, plan_path, result_dir = self.build_campaign(
            root, [sys.executable, "{parser}", "{not_a_placeholder}"])
        with self.assertRaisesRegex(ValueError, "unknown run_argv placeholder"):
            campaign.execute(plan_path, rerun_failed=False)
        self.assertFalse((result_dir / "result.json").exists())

    def test_parser_argv_without_a_copied_parser_is_rejected_at_plan_time(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        import campaign
        assets = root / "assets"
        assets.mkdir()
        (assets / "model.yaml").write_text("model: test\n")
        asset_path = root / "assets.json"
        asset_path.write_text(json.dumps({
            "schema": "radiance-fpga-1a-assets-v1",
            "targets": [{"id": "one", "kind": "model", "sm_count": 1,
                         "memory": "hbm", "loader": "logical",
                         "model_config": "model.yaml",
                         "run_argv": ["true", "{elf}", "{model_config}"],
                         "parser_argv": ["true"]}]}))
        with self.assertRaisesRegex(ValueError,
                                    "parser_argv needs a copied parser"):
            campaign.load_assets(asset_path, assets)


def _reject_constant(name):  # pragma: no cover - only runs on a bad parse
    raise AssertionError(f"non-finite JSON constant emitted: {name}")


if __name__ == "__main__":
    unittest.main()
