#!/usr/bin/env bash
set -euo pipefail
field_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$field_root"
field_python="${PIPER_FIELD_PYTHON:-/usr/bin/python3}"
"$field_python" -c 'import sys, platform; assert sys.version_info[:2] == (3, 10) and platform.machine() == "x86_64", "Use Ubuntu 22.04 x86_64 with system Python 3.10"'
pip_wheels=("$field_root"/wheelhouse/pip-*.whl)
if [[ ${#pip_wheels[@]} != 1 || ! -f ${pip_wheels[0]} ]]; then
  echo "The offline pip wheel is missing from the deployment directory" >&2
  exit 2
fi
if [[ ! -x .local/qwen-runtime/bin/python ]]; then
  "$field_python" -m venv --without-pip .local/qwen-runtime
fi
PYTHONPATH="${pip_wheels[0]}" .local/qwen-runtime/bin/python -m pip --disable-pip-version-check install \
  --no-index --only-binary=:all: --find-links "$field_root/wheelhouse-qwen" -r requirements-qwen.txt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .local/qwen-runtime/bin/python -c \
  'import torch; from transformers import Qwen3VLForConditionalGeneration; print("Qwen runtime ready", torch.__version__, "CUDA", torch.version.cuda)'
