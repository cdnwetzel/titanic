"""Capture machine and library context for cross-hardware comparison.

Run on each workstation before a repeat run, and commit the output:

    python scripts/env_info.py --out results/env_myhost.json

Comparing results/ artifacts across machines: gate scores should match to
within BLAS noise (watch the third decimal); wall-clock differences per
evaluation are visible in the runner.log timestamps.
"""

import argparse
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone


def gather():
    info = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_logical_cores": None,
        "cpu_model": None,
        "ram_gb": None,
        "gpus": [],
        "packages": {},
    }
    try:
        import os
        info["cpu_logical_cores"] = os.cpu_count()
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            info["gpus"] = [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]
    except Exception:
        pass
    for pkg in ["numpy", "pandas", "scikit-learn", "scipy", "xgboost", "lightgbm", "catboost", "optuna"]:
        try:
            mod = __import__(pkg)
            info["packages"][pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            info["packages"][pkg] = "not installed"
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="write JSON to this path instead of stdout")
    args = parser.parse_args()
    data = gather()
    text = json.dumps(data, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
