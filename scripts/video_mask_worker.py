"""JSON-lines streaming model worker, run in the isolated tracking environment."""

import contextlib
import json
import sys

import numpy as np
import torch

from piper_sim.video_mask_models import CutieStream, SAMStream


def main():
    method = sys.argv[1]
    if method not in ("cutie", "sam2_small", "edgetam"):
        raise ValueError("Unknown tracking method")
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.35)
    stream, expected = None, 0
    for line in sys.stdin:
        request = json.loads(line)
        response = {"request_id": request["request_id"], "index": request["index"]}
        try:
            rgb = np.load(request["image"], allow_pickle=False)
            index = request["index"]
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode(), \
                    torch.autocast("cuda", dtype=torch.bfloat16):
                if request["initial_mask"] is not None:
                    if stream is not None or index != 0:
                        raise ValueError("Initialization requires a fresh worker")
                    stream = CutieStream() if method == "cutie" else SAMStream(method)
                    mask = stream.initialize(rgb, np.load(request["initial_mask"], allow_pickle=False))
                else:
                    if stream is None or index != expected:
                        raise ValueError("Out-of-order or uninitialized video stream")
                    mask = stream.step(rgb, index)
            np.save(request["output"], mask, allow_pickle=False)
            expected = index + 1
            response["method"] = method
        except Exception as exc:  # noqa: BLE001 -- report worker errors then terminate the stream.
            response["error"] = str(exc)
        print(json.dumps(response), flush=True)
        if "error" in response:
            break


if __name__ == "__main__":
    main()
