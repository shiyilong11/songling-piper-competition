#!/usr/bin/env bash
set -euo pipefail
field_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$field_root"
field_python="${PIPER_FIELD_PYTHON:-/usr/bin/python3}"
"$field_python" -c 'import sys, platform; assert sys.version_info[:2] == (3, 10) and platform.machine() == "x86_64", "Use Ubuntu 22.04 x86_64 with system Python 3.10"'
if [[ ! -x .venv/bin/python ]]; then
  "$field_python" -m venv --without-pip --system-site-packages .venv
fi
pip_wheels=("$field_root"/wheelhouse/pip-*.whl)
if [[ ${#pip_wheels[@]} != 1 || ! -f ${pip_wheels[0]} ]]; then
  echo "The offline pip wheel is missing from the package" >&2
  exit 2
fi
PYTHONPATH="${pip_wheels[0]}" .venv/bin/python -m pip --disable-pip-version-check install \
  --no-index --only-binary=:all: --find-links "$field_root/wheelhouse" -r requirements.txt
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$field_root/src:$field_root/vendor" .venv/bin/python -c 'import cv2, mujoco, scipy, sam2, torch; assert not torch.version.cuda; print("CPU runtime ready", torch.__version__)'
