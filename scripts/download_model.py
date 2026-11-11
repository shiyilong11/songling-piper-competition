"""Explicit, optional download. Not called by the demo server."""

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
snapshot_download(
    "Qwen/Qwen3-VL-4B-Instruct",
    revision="ebb281ec70b05090aa6165b016eac8ec08e71b17",
    local_dir=ROOT / "models/Qwen3-VL-4B-Instruct",
)
