import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from PIL import Image

from piper_sim.l2_qwen_categories import parse_l2_categories
from piper_sim.qwen_adapter import (
    InvalidPrediction,
    check_gpu_budget,
    infer_http,
    make_http_payload,
    parse_prediction,
)


def ready(point=None):
    return {
        "status": "ready",
        "target_point_1000": point or [500, 500],
        "destination_color": "blue",
    }


def test_coordinates_and_fences():
    result = parse_prediction("```json\n" + json.dumps(ready()) + "\n```", (640, 480))
    assert result["target_pixel_xy"] == [319.5, 239.5]
    assert parse_prediction(json.dumps(ready([0, 1000])), (640, 480))["target_pixel_xy"] == [0, 479]


@pytest.mark.parametrize(
    "value",
    [
        ready([True, 0]),
        ready([-1, 0]),
        ready([1001, 0]),
        ready([float("nan"), 0]),
        ready([float("inf"), 0]),
        ready([1]),
        {**ready(), "joint_command": [1]},
        {**ready(), "destination_color": "purple"},
        {**ready(), "status": "ambiguous"},
        [],
        {"status": "ready"},
    ],
)
def test_reject_bad_predictions(value):
    with pytest.raises(InvalidPrediction):
        parse_prediction(json.dumps(value), (640, 480))


@pytest.mark.parametrize(
    "raw", ["{} {}", '{"status":"ready","status":"ambiguous"}', "Here is your answer: {}"]
)
def test_reject_bad_json(raw):
    with pytest.raises(InvalidPrediction):
        parse_prediction(raw, (640, 480))


def test_missing_target():
    result = parse_prediction(
        json.dumps(
            {"status": "target_not_found", "target_point_1000": None, "destination_color": None}
        ),
        (640, 480),
    )
    assert result == {
        "status": "target_not_found",
        "target_pixel_xy": None,
        "destination_color": None,
    }


def test_l2_categories_accept_abstentions_and_reject_unknown_labels_or_ids():
    def parse(*rows):
        return parse_l2_categories(
            json.dumps(
                {"objects": [{"id": identity, "category": label} for identity, label in rows]}
            ),
            ["1", "2", "3"],
        )

    assert parse(("3", "unknown"), ("2", "other"), ("1", "potato")) == {
        "1": "potato",
        "2": "other",
        "3": "unknown",
    }
    for rows in (
        (("1", "potato"), ("2", "apple"), ("3", "unknown")),
        (("1", "potato"), ("1", "pear"), ("3", "unknown")),
        (("1", "potato"), ("2", "other"), ("4", "unknown")),
        (("1", "potato"), ("2", "other")),
    ):
        with pytest.raises(InvalidPrediction):
            parse(*rows)


def test_gpu_budget_blocks_before_loading(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, stdout="5896\n")
    )
    with pytest.raises(RuntimeError, match="requires 12288"):
        check_gpu_budget("cuda:0", 12288)


def test_http_contract_local_server():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            payload = json.dumps(
                {
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": json.dumps(ready())}}
                    ]
                }
            ).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    settings = {
        "base_url": f"http://127.0.0.1:{server.server_port}/v1",
        "model": "mock",
        "api_key_env": "TEST_UNUSED_QWEN_KEY",
        "timeout_s": 2,
        "max_tokens": 192,
    }
    try:
        raw, _ = infer_http(Image.new("RGB", (16, 16)), "把苹果放到蓝色托盘", settings)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert parse_prediction(raw, (16, 16))["destination_color"] == "blue"
    assert len(received[0]["messages"]) == 2
    content = received[0]["messages"][1]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert [v["type"] for v in content] == ["image_url", "text"]


def test_cli_needs_only_public_inputs_and_rejects_invalid_result(tmp_path):
    # No evaluator truth, depth or simulator exists in this fixture.
    root = Path(__file__).resolve().parents[1]
    scene = tmp_path / "scene"
    scene.mkdir()
    Image.new("RGB", (32, 24)).save(scene / "camera_rgb.png")
    (scene / "task.json").write_text(
        json.dumps({"instructions": [{"id": "apple", "text": "把苹果放到蓝色托盘"}]})
    )
    response = tmp_path / "response.json"
    response.write_text(json.dumps(ready()))
    out = tmp_path / "valid"
    command = [
        sys.executable,
        str(root / "scripts/query_qwen.py"),
        "--scene",
        str(scene),
        "--instruction",
        "apple",
        "--backend",
        "mock",
        "--mode",
        "infer",
        "--mock-response",
        str(response),
    ]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    result = subprocess.run(
        command + ["--output", str(out)], capture_output=True, text=True, check=False, env=env
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads((out / "prediction.json").read_text())["target_pixel_xy"] == [15.5, 11.5]
    assert json.loads((out / "report.json").read_text())["inference_executed"] is False
    response.write_text('{"invalid": true}')
    bad = tmp_path / "invalid"
    result = subprocess.run(
        command + ["--output", str(bad)], capture_output=True, text=True, check=False, env=env
    )
    assert result.returncode == 1
    assert not (bad / "prediction.json").exists()
    assert json.loads((bad / "report.json").read_text())["success"] is False


def test_http_payload_has_no_simulator_state():
    payload = make_http_payload(
        Image.new("RGB", (32, 24)), "test", {"model": "mock", "max_tokens": 10}
    )
    assert set(payload) == {"model", "messages", "temperature", "max_tokens"}
