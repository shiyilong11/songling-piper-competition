"""Language grounding and mask artifacts for the local segmentation playground."""

import json
import os
import select
import subprocess
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw

from piper_sim.calibration import ROOT
from piper_sim.sam31_adapter import Sam31Client
from piper_sim.sam_adapter import Sam2Client
from piper_sim.segmentation_grounding import PALETTE


class GroundingClient:
    def __init__(self):
        self.worker = self.log = None

    @property
    def loaded(self):
        return self.worker is not None and self.worker.poll() is None

    def predict(self, image_path, instruction, output):
        return self._request({"image": str(image_path), "instruction": instruction}, output)

    def predict_l2_categories(self, crops, output):
        return self._request(
            {"l2_crops": {identity: str(path) for identity, path in crops.items()}}, output
        )

    def _request(self, payload, output):
        if not self.loaded:
            self.close()
            env = {
                **os.environ,
                "PYTHONPATH": str(ROOT / "src"),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
            env.pop("CUDA_VISIBLE_DEVICES", None)
            self.log = (output / "qwen_worker.log").open("a")
            self.worker = subprocess.Popen(
                [
                    str(ROOT / ".local/qwen-runtime/bin/python"),
                    str(ROOT / "scripts/qwen_segmentation_worker.py"),
                ],
                cwd=ROOT,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
                bufsize=1,
            )
        request_id = uuid.uuid4().hex
        try:
            self.worker.stdin.write(
                json.dumps(
                    {"request_id": request_id, **payload},
                    ensure_ascii=False,
                )
                + "\n"
            )
            self.worker.stdin.flush()
            ready, _, _ = select.select([self.worker.stdout], [], [], 180)
            if not ready:
                raise TimeoutError("Qwen推理超时，模型进程已释放。")
            line = self.worker.stdout.readline()
            if not line:
                raise RuntimeError("Qwen进程退出，请查看本次任务日志。")
            result = json.loads(line)
            if result.get("request_id") != request_id:
                raise RuntimeError("Qwen response mismatch")
            if "error" in result:
                raise RuntimeError(result["error"])
            (output / "qwen_response.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2)
            )
            return result
        except Exception:
            self.close()
            raise

    def close(self):
        if self.worker is not None:
            self.worker.terminate()
            try:
                self.worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait(timeout=5)
            for stream in (self.worker.stdin, self.worker.stdout):
                stream.close()
            self.worker = None
        if self.log is not None:
            self.log.close()
            self.log = None


class SegmentationPipeline:
    def __init__(self):
        self.qwen = GroundingClient()
        self.sam = Sam2Client()
        self.sam31 = Sam31Client()
        self.last_used = time.monotonic()

    def run_engine(
        self, engine, image_path, instruction, point_xy, output, stage, reference_point_xy=None
    ):
        if engine == "sam2":
            if point_xy is None:
                raise ValueError("SAM 2.1 不接收文本；请先在原图上设置独立参考点。")
            result = self.run(image_path, "", point_xy, output, stage)
            return {
                **result,
                "model": "SAM 2.1 Small / 人工点提示",
                "actual_prompt": None,
                "point_prompt_xy": point_xy,
                "language_understanding": False,
            }
        if engine == "qwen":
            result = self.run(image_path, instruction, point_xy, output, stage)
            return {**result, "model": "Qwen3-VL-4B + SAM2.1 Small", "actual_prompt": instruction}
        if engine == "sam31":
            self.qwen.close()
            stage("segmenting", "SAM 3.1 正在直接处理原始文本…")
            return self.sam31.predict(image_path, instruction, output)
        comparisons = []
        for name, title in [
            ("sam2", "SAM 2.1 / 人工点提示"),
            ("sam31", "SAM 3.1 / 原始文本"),
            ("qwen", "Qwen + SAM 2.1"),
        ]:
            folder = output / name
            folder.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            try:
                point = reference_point_xy if name == "sam2" else None
                value = self.run_engine(name, image_path, instruction, point, folder, stage)
                value["metrics"]["total_s"] = time.monotonic() - started
                comparisons.append({"engine": name, "title": title, "result": value})
            except Exception as exc:  # noqa: BLE001 -- show each arm's real status independently.
                comparisons.append(
                    {"engine": name, "title": title, "error": str(exc), "inference_executed": False}
                )
            finally:
                self.qwen.close()
                self.sam31.close()
        return {
            "status": "comparison",
            "message": "三路对比结束，请分别查看实际输入和结果",
            "targets": [],
            "metrics": {},
            "comparisons": comparisons,
            "actual_prompt": instruction,
        }

    def run(self, image_path, instruction, point_xy, output, stage):
        self.last_used = time.monotonic()
        image = Image.open(image_path).convert("RGB")
        rgb = np.asarray(image)
        metrics = {}
        if point_xy is None:
            stage("understanding", "Qwen 正在理解指令并定位目标…")
            response = self.qwen.predict(image_path, instruction, output)
            grounding = response["prediction"]
            metrics["qwen_s"] = response["metrics"]["request_s"]
        else:
            grounding = {
                "status": "ready",
                "message": "使用手动选点",
                "targets": [
                    {
                        "label": "手动选择的目标",
                        "point_xy": point_xy,
                        "box_xyxy": None,
                    }
                ],
            }
        if grounding["status"] != "ready":
            return {
                "status": grounding["status"],
                "message": grounding["message"],
                "targets": [],
                "metrics": metrics,
            }
        stage("segmenting", "SAM 正在生成目标 mask…")
        union = np.zeros(rgb.shape[:2], dtype=bool)
        overlay = rgb.copy()
        targets = []
        sam_s = 0
        for index, target in enumerate(grounding["targets"]):
            folder = output / f"target_{index + 1}"
            instance, metadata = self.sam.segment(
                rgb, target["point_xy"], time.time(), output.name, folder, target["box_xyxy"]
            )
            mask = instance.mask
            sam_s += metadata["inference_s"]
            color = PALETTE[index]
            overlay[mask] = (overlay[mask] * 0.5 + np.asarray(color) * 0.5).astype(np.uint8)
            union |= mask
            Image.fromarray(mask.astype(np.uint8) * 255).save(folder / "mask.png")
            targets.append(
                {
                    **target,
                    "mask_pixels": int(mask.sum()),
                    "predicted_iou": metadata["predicted_iou"],
                    "color": list(color),
                }
            )
        marked = Image.fromarray(overlay)
        draw = ImageDraw.Draw(marked)
        for index, target in enumerate(targets):
            x, y = target["point_xy"]
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="white", outline="#13362e", width=2)
            draw.text(
                (x + 9, y - 14), str(index + 1), fill="white", stroke_width=2, stroke_fill="#13362e"
            )
        marked.save(output / "overlay.png")
        Image.fromarray(union.astype(np.uint8) * 255).save(output / "mask.png")
        cutout = np.dstack([rgb, union.astype(np.uint8) * 255])
        Image.fromarray(cutout).save(output / "cutout.png")
        metrics["sam_s"] = sam_s
        self.last_used = time.monotonic()
        return {
            "status": "ready",
            "message": f"已分割 {len(targets)} 个目标",
            "targets": targets,
            "metrics": metrics,
            "robot_control_enabled": False,
        }

    def close(self):
        self.qwen.close()
        self.sam.close()
        self.sam31.close()
