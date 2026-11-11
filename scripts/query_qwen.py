"""Prepare/check/query visual selection; never sends robot commands."""

import argparse
import json
import time
from pathlib import Path

import yaml

from piper_sim.qwen_adapter import (
    infer_http,
    infer_local,
    load_image,
    parse_prediction,
    prepare_local_inputs,
    request_metadata,
)

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene", type=Path, help="Export directory with camera_rgb.png/task.json")
    source.add_argument("--image", type=Path, help="Real or simulated RGB image")
    instruction = parser.add_mutually_exclusive_group(required=True)
    instruction.add_argument(
        "--instruction", help="Public task.json instruction ID; requires --scene"
    )
    instruction.add_argument("--instruction-text")
    parser.add_argument("--config", type=Path, default=ROOT / "config/vlm/qwen.yaml")
    parser.add_argument("--mode", choices=["prepare", "check", "infer"], default="prepare")
    parser.add_argument("--backend", choices=["local", "http", "mock"])
    parser.add_argument("--mock-response", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.instruction and not args.scene:
        parser.error("--instruction requires --scene")
    cfg = yaml.safe_load(args.config.read_text())
    backend = args.backend or cfg["backend"]
    if args.mode == "check" and backend != "local":
        parser.error("CPU processor check requires local backend")
    if args.mock_response and backend != "mock":
        parser.error("--mock-response requires --backend mock")
    if cfg.get("response_coordinates") != "normalized_1000":
        parser.error("Only normalized_1000 model coordinates are supported")
    instruction_text = args.instruction_text
    if args.instruction:
        task = json.loads((args.scene / "task.json").read_text())
        matches = [x["text"] for x in task["instructions"] if x["id"] == args.instruction]
        if len(matches) != 1:
            parser.error("Instruction ID must match exactly one public task entry")
        instruction_text = matches[0]
    if not instruction_text.strip():
        parser.error("Instruction must not be empty")
    image = load_image(args.image or args.scene / "camera_rgb.png")
    output = args.output or ROOT / "artifacts/vlm" / str(time.time_ns())
    output.mkdir(parents=True, exist_ok=False)
    image.save(output / "input.png")
    write_json(output / "request.json", request_metadata(image, instruction_text))
    report = {
        "mode": args.mode,
        "backend": backend,
        "inference_executed": False,
        "robot_control_enabled": False,
        "model_revision": cfg["model_revision"],
    }
    start = time.monotonic()
    exit_code = 0
    try:
        model_path = ROOT / cfg["model_path"]
        if args.mode == "check":
            import torch
            from transformers import Qwen3VLForConditionalGeneration

            _, inputs = prepare_local_inputs(model_path, image, instruction_text, cfg["local"])
            report.update(
                input_shapes={k: list(v.shape) for k, v in inputs.items()},
                model_class=Qwen3VLForConditionalGeneration.__name__,
                cuda_initialized=torch.cuda.is_initialized(),
            )
            if report["cuda_initialized"]:
                raise RuntimeError("CPU check unexpectedly initialized CUDA")
        elif args.mode == "infer":
            if backend == "mock":
                raw = (
                    args.mock_response.read_text()
                    if args.mock_response
                    else json.dumps(
                        {
                            "status": "ambiguous",
                            "target_point_1000": None,
                            "destination_color": None,
                        }
                    )
                )
                details = {"mock_only": True}
            else:
                report["inference_attempted"] = True
                if backend == "local":
                    raw, details = infer_local(model_path, image, instruction_text, cfg["local"])
                elif backend == "http":
                    raw, details = infer_http(image, instruction_text, cfg["http"])
                else:
                    raise ValueError("Unknown backend")
                report["inference_executed"] = True
            report.update(details)
            (output / "raw_response.txt").write_text(raw)
            prediction = parse_prediction(raw, image.size)
            write_json(output / "prediction.json", prediction)
            report["prediction_status"] = prediction["status"]
        report["success"] = True
    except Exception as exc:  # noqa: BLE001 -- CLI boundary: persist failure, return nonzero.
        report.update(success=False, error_type=type(exc).__name__, error=str(exc))
        exit_code = 1
    report["elapsed_s"] = round(time.monotonic() - start, 3)
    write_json(output / "report.json", report)
    print(json.dumps({"output": str(output.resolve()), **report}, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
