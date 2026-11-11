"""Detect L1 RGB proposals in one saved image. No camera, robot, GPU or server."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import numpy as np
import yaml

from piper_sim.calibration import ROOT
from piper_sim.l1_rgb import L1RGBDetector, annotate_rgb, load_l1_rgb_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/competition/l1_rgb.yaml")
    parser.add_argument("--frame-id", default="offline_image")
    parser.add_argument(
        "--timestamp-s",
        type=float,
        default=0.0,
        help="Sensor timestamp; default 0 is an offline placeholder",
    )
    args = parser.parse_args()
    cv2.setNumThreads(1)
    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        parser.error(f"Cannot read image: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cfg = load_l1_rgb_config(args.config)
    result = L1RGBDetector(cfg).detect(rgb, frame_id=args.frame_id, timestamp_s=args.timestamp_s)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = result.as_dict()
    metadata["source"] = {
        "image": str(args.image.resolve()),
        "sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "mode": "saved_image_offline",
        "live_sensor_verified": False,
    }
    masks = {}
    for i, proposal in enumerate(result.proposals):
        key = f"mask_{i:03d}"
        masks[key] = proposal.mask
        metadata["proposals"][i]["mask"] = {
            "file": "masks.npz",
            "key": key,
            "encoding": "boolean_HW",
        }
    np.savez_compressed(args.output / "masks.npz", **masks)
    (args.output / "detections.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    (args.output / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    preview = annotate_rgb(rgb, result)
    if not cv2.imwrite(str(args.output / "preview.png"), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)):
        raise RuntimeError("Could not write preview")
    print(
        json.dumps(
            {
                "rgb_candidate_counts": metadata["rgb_candidate_counts"],
                "review_components": sum(bool(p.rejection_reasons) for p in result.proposals),
                "grasp_ready": False,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
