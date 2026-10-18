"""Complete-output equivalent of the original nightly GEMM BF16 acceptance."""
import array
import math
from pathlib import Path
import re
import struct
import sys


def bf16(value):
    return struct.unpack('<f', struct.pack('<I', value << 16))[0]


def check(output: Path, golden: Path, log: str, clusters: int):
    result = {'tohost_pass': log.count('isa-test passed with tohost=0') == clusters,
              'host_handshake_pass': 'nightly host released verify GO after all END stamps' in log}
    if not output.exists():
        return dict(result, full_output_pass=False, reason='output dump missing')
    got = array.array('H'); got.frombytes(output.read_bytes())
    ref = array.array('H'); ref.frombytes(golden.read_bytes())
    if sys.byteorder != 'little':
        got.byteswap(); ref.byteswap()
    if len(got) != 512 * 512 or len(got) != len(ref):
        return dict(result, full_output_pass=False, reason='wrong output size')
    exact = bad = poison = nonfinite = 0
    se = sr = 0.0
    for value, expected in zip(got, ref):
        exact += value == expected
        poison += value == 0xeeee
        nonfinite += ((value >> 7) & 255) == 255
        if value != expected:
            same_sign = (value >> 15) == (expected >> 15)
            one_step = same_sign and abs((value & 0x7fff) - (expected & 0x7fff)) == 1
            tiny = ((value | expected) & 0x7fff) <= 0x0080
            bad += not (one_step or tiny)
        actual_f, ref_f = bf16(value), bf16(expected)
        se += (actual_f - ref_f) ** 2
        sr += ref_f ** 2
    error = math.sqrt(se / sr) if sr else (math.inf if se else 0.0)
    # Original host.cpp: max_err_bp=100 (1%), max_bad=n/100; zero poison/nonfinite.
    result.update(elements=len(ref), exact=exact, bad=bad, poison=poison,
                  nonfinite=nonfinite, relative_frobenius_error=error,
                  full_output_pass=poison == 0 and nonfinite == 0 and error <= 0.01 and bad <= len(ref) // 100)
    reports = []
    for cluster, words in re.findall(r'cluster (\d+) smem 0x80080: ([0-9a-f ]+)', log):
        values = [int(x, 16) for x in words.split()][::2]
        if len(values) == 8:
            reports.append(dict(cluster=int(cluster), done=values[0], n=values[1], exact=values[2],
                                ulp1=values[3], bad=values[4], poison=values[5], nonfinite=values[6], err_bp=values[7]))
    result['gpu_reports'] = reports
    result['gpu_verification_pass'] = len(reports) == clusters and all(
        v['done'] == 0x5ef1d0e5 and v['n'] == len(ref) and v['poison'] == 0 and
        v['nonfinite'] == 0 and v['bad'] <= len(ref) // 100 and v['err_bp'] <= 100 for v in reports)
    result['gpu_reports_agree_pass'] = bool(reports) and all(
        (v['exact'], v['bad'], v['err_bp']) == (reports[0]['exact'], reports[0]['bad'], reports[0]['err_bp']) for v in reports)
    return result
