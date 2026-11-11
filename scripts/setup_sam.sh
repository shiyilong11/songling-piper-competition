#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
sam_commit=2b90b9f5ceec907a1c18123530e92e794ad901a4
sam_sha=6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38
sam_weights=models/sam2/sam2.1_hiera_small.pt
mkdir -p .local/sam2 models/sam2
if [[ ! -d .local/sam2/source/.git ]]; then
  git clone https://github.com/facebookresearch/sam2.git .local/sam2/source
fi
git -C .local/sam2/source checkout --detach "$sam_commit"
python3.12 -m venv .local/sam2-runtime
sam_python=.local/sam2-runtime/bin/python
"$sam_python" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
"$sam_python" -m pip install scipy==1.18.1 setuptools wheel
SAM2_BUILD_CUDA=0 "$sam_python" -m pip install --no-build-isolation .local/sam2/source
if [[ ! -f "$sam_weights" ]]; then
  curl --fail --location --retry 3 --output "$sam_weights.part" \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
  printf '%s  %s\n' "$sam_sha" "$sam_weights.part" | sha256sum -c -
  mv "$sam_weights.part" "$sam_weights"
fi
printf '%s  %s\n' "$sam_sha" "$sam_weights" | sha256sum -c -
