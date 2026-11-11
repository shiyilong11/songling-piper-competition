"""Isolated SAM JSON-lines worker. Receives RGB and pixel prompts only."""

import contextlib
import json
import resource
import sys
from pathlib import Path

import numpy as np
import yaml

from piper_sim.sam_adapter import Sam2Session


def main():
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    session = None
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if session is None:
                    session = Sam2Session(config)
                mask, metadata = session.predict(
                    np.load(request["image"], allow_pickle=False),
                    request["point_xy"],
                    request.get("box_xyxy"),
                )
                np.save(request["mask"], mask, allow_pickle=False)
                metadata["peak_host_rss_mib"] = (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                )
                if config["device"] != "cpu":
                    metadata["peak_allocated_mib"] = (
                        session.torch.cuda.max_memory_allocated(config["device"]) / 2**20
                    )
                    metadata["peak_reserved_mib"] = (
                        session.torch.cuda.max_memory_reserved(config["device"]) / 2**20
                    )
            result = metadata
        except Exception as exc:  # noqa: BLE001 -- return inference failure, never move robot.
            result = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps({**result, "request_id": request.get("request_id")}), flush=True)


if __name__ == "__main__":
    main()
