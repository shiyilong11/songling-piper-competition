#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES='' MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
if [[ -d "$ROOT/.local/osmesa/root/usr/lib/x86_64-linux-gnu" ]]; then
  export LD_LIBRARY_PATH="$ROOT/.local/osmesa/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "${DEMO_PYTHON:-$ROOT/.venv/bin/python}" scripts/interactive_demo.py "$@"
