#!/usr/bin/env python3
"""Generate/check full decoder variants serially using existing compact images.

Config JSON maps each model to checkpoint_dir, weight_image and token_ids.
Each phase preserves its command, log and failure. Numerical reference checks
run before optional ELF builds. This does not run a second heavy ISA simulator.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
sys.path.insert(0, str(CHAIN))
from compile_decoder import MODELS
sys.path.insert(0, str(CHAIN.parents[1]/"kernels/evaluation/llm"))
from stitch import model_specs


def digest(path):
    result=hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda:stream.read(1 << 20),b""):
            result.update(data)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--out-root",type=Path,required=True)
    parser.add_argument("--models",nargs="+",choices=MODELS[:3],default=list(MODELS[:3]))
    parser.add_argument("--mx-linear-policy",choices=("all","q_proj"),default="q_proj")
    parser.add_argument("--phase-timeout",type=int,default=1800)
    parser.add_argument("--build-device",action="store_true")
    args = parser.parse_args()
    if args.phase_timeout <= 0: parser.error("timeout must be positive")
    config = json.loads(args.config.read_text())
    output = args.out_root.resolve()
    output.mkdir(parents=True,exist_ok=True)
    specs = model_specs()
    records = dict(schema="radiance-rakanic-mx-vpu-full-decoder-v1",models={},
                   whole_program_latency_valid=False,config_sha256=digest(args.config))
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
               RADIANCE_BUILD_JOBS="1",PYTHONDONTWRITEBYTECODE="1")
    for model in args.models:
        item=config[model]
        records["models"][model]=record=dict(phases=[],accepted=False)
        def phase(name,command):
            log=output/f"{model}-{name}.log"
            start=time.monotonic()
            print(f"{model}: {name}",flush=True)
            with log.open("w") as stream:
                try:
                    result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=args.phase_timeout)
                    code=result.returncode
                except subprocess.TimeoutExpired:
                    code=124
            record["phases"].append(dict(name=name,command=command,returncode=code,
                elapsed_seconds=time.monotonic()-start,log=str(log),log_sha256=digest(log)))
            (output/"results.json").write_text(json.dumps(records,indent=2)+"\n")
            return code==0
        target=output/model
        command=[sys.executable,str(CHAIN/"compile_decoder.py"),"--model",model,
            "--layers",str(specs[model]["num_hidden_layers"]),"--prefill",str(len(item["token_ids"])),
            "--decode-steps","2","--generation","greedy","--token-ids",*map(str,item["token_ids"]),
            "--execution-backend","rakanic_mx_vpu","--mx-linear-policy",args.mx_linear_policy,
            "--checkpoint-dir",item["checkpoint_dir"],"--weight-image",item["weight_image"],
            "--device-check-all-stages","--stages-per-object","32","--out-root",str(output)]
        if not phase("generate",command):continue
        result=output/f"{model}-native-result.json"
        if not phase("native",[sys.executable,str(CHAIN/"run_checkpoint_native.py"),
            "--model",model,"--generated-root",str(output),"--jobs","1","--threads","8",
            "--source-only","--out",str(result)]):continue
        manifest=json.loads((target/"manifest.json").read_text())
        record["original_checkpoint_greedy_exact"]=manifest["format_comparison"].get("original_checkpoint_greedy_exact")
        record["format_reference_passed"]=True
        record["accepted"]=record["original_checkpoint_greedy_exact"] is True
        record["native_result_sha256"]=digest(result)
        if args.build_device:
            # Build the already-generated target; do not regenerate/requantize.
            code="import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from compile_decoder import build_device; build_device(Path(sys.argv[2]))"
            phase("build",[sys.executable,"-c",code,str(CHAIN),str(target)])
        (output/"results.json").write_text(json.dumps(records,indent=2)+"\n")
    print(json.dumps({m:dict(format_reference_passed=r.get("format_reference_passed",False),
        original_checkpoint_greedy_exact=r.get("original_checkpoint_greedy_exact"),accepted=r["accepted"])
        for m,r in records["models"].items()}),flush=True)


if __name__=="__main__":main()
