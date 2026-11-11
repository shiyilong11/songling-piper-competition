"""Dedicated 2D language grounding worker; no destination or robot control."""

import contextlib
import json
import sys

import yaml

from piper_sim.calibration import ROOT
from piper_sim.l2_qwen_categories import L2_CATEGORY_PROMPT, parse_l2_categories
from piper_sim.qwen_adapter import InvalidPrediction, LocalQwenSession, load_image
from piper_sim.segmentation_grounding import GROUNDING_PROMPT, parse_grounding


def main():
    config = yaml.safe_load((ROOT / "config/vlm/qwen.yaml").read_text())
    settings = {**config["local"], "max_new_tokens": 640}
    session = None
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if session is None:
                    session = LocalQwenSession(ROOT / config["model_path"], settings)
                if "l2_crops" in request:
                    result = classify_l2(session, request["l2_crops"])
                else:
                    image = load_image(request["image"])
                    raw, metrics = session.predict(image, request["instruction"], GROUNDING_PROMPT)
                    prediction = parse_grounding(raw, image.size)
                    result = {"raw": raw, "metrics": metrics, "prediction": prediction}
        except Exception as exc:  # noqa: BLE001 -- private worker boundary.
            result = {"error": f"{type(exc).__name__}: {exc}"}
        print(
            json.dumps({**result, "request_id": request.get("request_id")}, ensure_ascii=False),
            flush=True,
        )


def classify_l2(session, crops):
    categories, records = {}, {}
    for identity, path in crops.items():
        raw, metrics = session.predict(
            load_image(path),
            f"识别这张卡片标记的物体。只输出指定编号：{identity}。",
            L2_CATEGORY_PROMPT,
        )
        record = {"raw": raw, "metrics": metrics}
        try:
            categories[identity] = parse_l2_categories(raw, [identity])[identity]
        except InvalidPrediction as exc:
            categories[identity] = "unknown"
            record["error"] = str(exc)
        records[identity] = record
    return {"categories": categories, "records": records}


if __name__ == "__main__":
    main()
