#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export HEAD_RETURN_BASE_SOURCE="$task_root/tools/test_baselines/full37"
export TORSO_RETRY_BASE_SOURCE="$task_root/tools/test_baselines/full41"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ROS_DOMAIN_ID="${ERC_TEST_ROS_DOMAIN_ID:-136}"
export GZ_PARTITION="erc_solution_tests_$$"
export PYTHONPATH="$task_root/src/erc_phase1_solution:$task_root/src/erc_phase1_solution/test${PYTHONPATH:+:$PYTHONPATH}"
python3 - "$task_root/tools/test_baselines" <<'PY'
from pathlib import Path
import hashlib, json, sys
root = Path(sys.argv[1])
for name, expected in json.loads((root/'manifest.json').read_bytes()).items():
    actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit('Historical test fixture changed: ' + name)
import rclpy
PY
cd -- "$task_root/src/erc_phase1_solution"
exec python3 -m pytest test -q "$@"
