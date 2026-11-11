"""Private JSON-lines worker. Receives only RGB paths and instruction text."""

import contextlib
import json
import sys
from pathlib import Path

import yaml

from piper_sim.qwen_adapter import LocalQwenSession, load_image, parse_prediction, request_metadata


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    session = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if session is None:
                    session = LocalQwenSession(root / cfg["model_path"], cfg["local"])
                image = load_image(request["image"])
                raw, metrics = session.predict(image, request["instruction"])
            result = {
                "raw": raw,
                "metrics": metrics,
                "request": request_metadata(image, request["instruction"]),
                "inference_executed": True,
                "robot_control_enabled": False,
            }
            try:
                result["prediction"] = parse_prediction(raw, image.size)
            except ValueError as exc:
                result["validation_error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 -- worker must return errors to UI.
            result = {"error": str(exc), "inference_executed": False}
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
