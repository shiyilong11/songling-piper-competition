"""SAM 3.1 checkpoint image detector. Text is passed unchanged, with no VLM assistance."""

import json
import os
import select
import subprocess
import uuid

import yaml

from piper_sim.calibration import ROOT


class Sam31Client:
    def __init__(self):
        self.config = ROOT / "config/segmentation/sam31.yaml"
        self.cfg = yaml.safe_load(self.config.read_text())
        self.worker = self.log = None

    def availability(self):
        ready = (ROOT / self.cfg["checkpoint"]).is_file()
        return {
            "ready": ready,
            "model": self.cfg["model"],
            "message": "权重已就绪，首次推理仍需加载"
            if ready
            else "尚未下载官方权重：需申请 facebook/sam3.1 访问权限并在本机登录 Hugging Face。",
        }

    def predict(self, image, text, output):
        if not self.availability()["ready"]:
            raise FileNotFoundError(self.availability()["message"])
        output.mkdir(parents=True, exist_ok=True)
        if self.worker is None or self.worker.poll() is not None:
            self.close()
            env = {
                **os.environ,
                "PYTHONPATH": str(ROOT / "src"),
                "OMP_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "1",
                "HF_HUB_OFFLINE": "1",
            }
            env.pop("CUDA_VISIBLE_DEVICES", None)
            self.log = (output / "sam31_worker.log").open("a")
            self.worker = subprocess.Popen(
                [
                    str(ROOT / ".local/sam3-runtime/bin/python"),
                    str(ROOT / "scripts/sam31_worker.py"),
                ],
                cwd=ROOT,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
                bufsize=1,
            )
        key = uuid.uuid4().hex
        try:
            self.worker.stdin.write(
                json.dumps(
                    {"request_id": key, "image": str(image), "prompt": text, "output": str(output)},
                    ensure_ascii=False,
                )
                + "\n"
            )
            self.worker.stdin.flush()
            ready, _, _ = select.select([self.worker.stdout], [], [], self.cfg["timeout_s"])
            if not ready:
                raise TimeoutError("SAM 3.1 推理超时")
            line = self.worker.stdout.readline()
            if not line:
                raise RuntimeError("SAM 3.1 进程退出，请查看任务日志")
            result = json.loads(line)
            if result.get("request_id") != key:
                raise RuntimeError("SAM 3.1 response mismatch")
            if "error" in result:
                raise RuntimeError(result["error"])
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
