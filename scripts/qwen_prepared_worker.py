"""Isolated on-demand Qwen worker for prepared wrist multi-frame requests."""

import contextlib
import json
import sys
from pathlib import Path

import yaml
from PIL import Image

from piper_sim.calibration import ROOT
from piper_sim.multiframe_task import make_multiframe_messages
from piper_sim.qwen_adapter import LocalQwenSession
from piper_sim.scene_memory import load_memory_config


def main():
    session = None
    cfg = load_memory_config()
    model_cfg = yaml.safe_load((ROOT / "config/vlm/qwen.yaml").read_text())
    settings = {
        **model_cfg["local"],
        "min_pixels": cfg["min_image_pixels"],
        "max_pixels": cfg["max_total_image_pixels"] // cfg["max_keyframes"],
        "max_new_tokens": cfg["max_new_tokens"],
    }
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                request = {}
                raise TypeError("JSON object required")
            with contextlib.redirect_stdout(sys.stderr):
                path = Path(request["request_file"]).resolve()
                payload = json.loads(path.read_text())
                names = payload["annotated_image_files"]
                if not 1 <= len(names) <= cfg["max_keyframes"]:
                    raise ValueError("Keyframe budget exceeded")
                images = []
                for name in names:
                    image_path = (path.parent / name).resolve()
                    if not image_path.is_relative_to(path.parent):
                        raise ValueError("Image must be inside prepared request")
                    with Image.open(image_path) as image:
                        images.append(image.convert("RGB"))
                if payload.get("request_kind") == "tracking_identity":
                    messages = payload["messages"]
                    if not isinstance(messages, list) or len(messages) != 2:
                        raise ValueError("Invalid tracking identity messages")
                else:
                    messages = make_multiframe_messages(
                        payload["instruction"], images, payload["snapshot"]
                    )
                followups = request.get("followup_messages", [])
                if (
                    not isinstance(followups, list)
                    or len(followups) > 2
                    or any(
                        not isinstance(message, dict)
                        or set(message) != {"role", "content"}
                        or message["role"] not in ("assistant", "user")
                        or not isinstance(message["content"], str)
                        or len(message["content"]) > 20000
                        for message in followups
                    )
                ):
                    raise ValueError("Invalid output-repair messages")
                messages.extend(followups)
                if session is None:
                    session = LocalQwenSession(ROOT / model_cfg["model_path"], settings)
                raw, metrics = session.predict_messages(images, messages)
                result = {"raw": raw, "metrics": metrics, "inference_executed": True}
        except Exception as exc:  # noqa: BLE001 -- keep the UI error boundary responsive.
            result = {"error": str(exc), "inference_executed": False}
        print(
            json.dumps(
                {**result, "request_id": request.get("request_id"), "robot_control_enabled": False},
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
