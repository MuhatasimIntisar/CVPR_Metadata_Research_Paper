#!/usr/bin/env python3
"""Make vLLM accept MIG device UUIDs in CUDA_VISIBLE_DEVICES (SLURM on Kelvin2 sets e.g. 'MIG-637c...').

vLLM's Platform.device_id_to_physical_device_id() does int() on the CUDA_VISIBLE_DEVICES entry and crashes on a
MIG UUID while importing. The patch keeps CUDA_VISIBLE_DEVICES as SLURM set it (so CUDA still uses the allocated
MIG slice) and, for a non-numeric entry, returns the logical index for the NVML capability query (all MIG
slices of an A100 report the same compute capability). Idempotent; run with the vmp-vllm env's python.
"""
import importlib.util
import pathlib
import re
import sys

MARK = "# vmp: MIG UUID patch"
spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    sys.exit("vllm is not installed in this python")
p = pathlib.Path(spec.origin).parent / "platforms" / "interface.py"
src = p.read_text(encoding="utf-8")
if MARK in src:
    print(f"already patched: {p}")
    sys.exit(0)
new, n = re.subn(r"^(\s*)return int\(physical_device_id\)\s*$",
                 r"\1return int(physical_device_id) if str(physical_device_id).strip().isdigit() else device_id  " + MARK,
                 src, count=1, flags=re.M)
if n != 1:
    sys.exit(f"pattern not found in {p}; this vLLM version needs a different fix")
p.write_text(new, encoding="utf-8")
print(f"patched: {p}")
