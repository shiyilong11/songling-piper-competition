"""CPU inference for the exported YOLOE visual-prompt mask model."""

import numpy as np

from piper_sim.calibration import ROOT

YOLOE_MASK_MODEL = ROOT / "models/yoloe/yoloe26n-visual-640x480.torchscript"


class YoloeMaskModel:
    def __init__(self, path=YOLOE_MASK_MODEL):
        import torch

        self.torch = torch
        self.model = torch.jit.load(str(path), map_location="cpu").eval()

    def predict(self, rgb):
        if rgb.dtype != np.uint8 or rgb.shape != (480, 640, 3):
            raise ValueError("The exported YOLOE model expects 640x480 RGB")
        torch = self.torch
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        with torch.inference_mode():
            rows, prototypes = self.model(tensor[None].float() / 255)
            if rows.shape != (1, 60, 38) or prototypes.shape != (1, 32, 120, 160):
                raise ValueError("Unexpected YOLOE export output dimensions")
            rows = rows[0]
            rows = rows[(rows[:, 4] > 0.20) & torch.isfinite(rows).all(dim=1)]
            if not len(rows):
                return []
            logits = (rows[:, 6:] @ prototypes[0].reshape(32, -1)).reshape(-1, 120, 160)
            pixels = torch.nn.functional.interpolate(
                logits[None], size=rgb.shape[:2], mode="bilinear", align_corners=False
            )[0].numpy()
            rows = rows.numpy()
        results = []
        for index in np.argsort(-rows[:, 4]):
            row = rows[index]
            x1, y1, x2, y2 = np.clip(np.rint(row[:4]), 0, [640, 480, 640, 480]).astype(int)
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            mask[y1:y2, x1:x2] = pixels[index, y1:y2, x1:x2] > 0
            area = np.count_nonzero(mask)
            if area < 60:
                continue
            if any(
                np.count_nonzero(mask & old) / np.count_nonzero(mask | old) >= 0.55
                for old, _ in results
            ):
                continue
            results.append((mask, float(row[4])))
        return results
