import json
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np

from piper_sim.field_dashboard import FieldDashboard
from piper_sim.field_process import FieldProcess
from piper_sim.l2_progress import L2Progress


def evidence(process):
    return [json.loads(line) for line in process.log.read_text().splitlines()]


def target(x, appearance):
    mask = np.zeros((32, 40), dtype=bool)
    mask[6:21, 12:23] = True
    return SimpleNamespace(
        center=np.array([x, 0.1, 0.08]), mask=mask, appearance_rgb=np.array(appearance)
    )


def test_l1_numbers_recognition_in_the_timed_round(tmp_path):
    process = FieldProcess(
        tmp_path, "l1", {"red": "green", "green": "red", "yellow": "yellow"}, 180
    )
    block = target(0.3, [180, 12, 10])
    process.observe("sensors/0001.jpg", 2.0, [("red", block)])
    assert not any(e["event"] == "identified" for e in evidence(process))
    process.begin(lambda: 0.5)
    process.observe("sensors/0002.jpg", 0.4, [("red", block)])
    process.observe("sensors/0003.jpg", 0.5, [("red", block)])
    process.finish(True, None)
    records = [e for e in evidence(process) if e["event"] == "identified"]
    assert len(records) == 1
    assert records[0]["target_id"] == "L1-01"
    assert records[0]["source_frame"] == "sensors/0002.jpg"
    assert records[0]["destination"] == "green"
    assert records[0]["round_elapsed_s"] == 0.5


def test_l2_number_survives_grasp_order_and_revoked_placement(tmp_path):
    mapping = {"banana": "yellow", "pear": "green"}
    process = FieldProcess(tmp_path, "l2", mapping, 300)
    process.begin(lambda: 3.0)
    process.observe("sensors/0001.jpg", 2.0, [])
    banana = SimpleNamespace(color="banana", key="frame1:0", target=target(0.2, [180, 142, 12]))
    pear = SimpleNamespace(color="pear", key="frame1:1", target=target(0.4, [77, 122, 21]))
    process.fruits([banana, pear], allow_new=True)
    assert [e["target_id"] for e in evidence(process) if e["event"] == "identified"] == [
        "L2-01",
        "L2-02",
    ]
    progress = L2Progress(color_mapping=mapping)
    item = progress.start(pear.color, "green", pear.target)
    process.observe("sensors/0002.jpg", 2.6, [])
    process.bind_fruit(item, pear)
    process.event({"event": "holding_estimate", "id": item["id"]})
    progress.verify(item, pear.target, 3.0)
    process.sync_fruits(progress.items)
    progress.fail(item, "final_recheck_failed")
    process.sync_fruits(progress.items)
    process.finish(False, "最终复查未通过")
    records = {e["event"]: e for e in evidence(process)}
    assert records["attempt_started"]["target_id"] == "L2-02"
    assert records["attempt_started"]["source_frame"] == "sensors/0001.jpg"
    assert records["lift_estimate"]["official_lift_verified"] is False
    assert records["placement_verified"]["target_id"] == "L2-02"
    assert records["retry_or_review"]["target_id"] == "L2-02"
    page = process.path.read_text()
    assert "复查待确认" in page and "0/6" in page
    assert 'http-equiv="refresh"' not in page


def test_l2_selected_banana_replaces_a_stale_unbound_display_candidate(tmp_path):
    mapping = {"banana": "yellow", "pear": "green"}
    process = FieldProcess(tmp_path, "l2", mapping, 300)
    process.begin(lambda: 10.0)
    process.observe("sensors/0001.jpg", 9.0, [])
    pears = [
        SimpleNamespace(color="pear", key=f"scan:{i}", target=target(0.2 + 0.12 * i, [77, 122, 21]))
        for i in range(6)
    ]
    process.fruits(pears, allow_new=True)
    progress = L2Progress(color_mapping=mapping)
    pear = progress.start("pear", "green", pears[0].target)
    process.bind_fruit(pear, pears[0])
    banana = SimpleNamespace(color="banana", key="next:0", target=target(0.3, [180, 142, 12]))
    process.observe("sensors/0002.jpg", 9.5, [])
    process.fruits([banana], allow_new=True)
    item = progress.start("banana", "yellow", banana.target)
    process.bind_fruit(item, banana)
    process.event({"event": "holding_estimate", "id": item["id"]})
    progress.verify(item, banana.target, 10.0)
    process.sync_fruits(progress.items)
    records = {e["event"]: e for e in evidence(process)}
    assert records["candidate_replaced"]["source_frame"] == "sensors/0002.jpg"
    assert records["placement_verified"]["category"] == "banana"
    assert process.rows[process.bindings[pear["id"]]]["category"] == "pear"
    assert len(process.rows) == 6


def test_dashboard_keeps_rgb_depth_paired_and_mask_on_source_frame(tmp_path):
    dashboard = FieldDashboard(0, (0.2, 2.0))
    try:
        block = target(0.3, [180, 12, 10])
        rgb = np.zeros((*block.mask.shape, 3), dtype=np.uint8)
        depth = np.full(block.mask.shape, 0.5, dtype=float)
        dashboard.publish_camera(2.0, rgb, depth)
        process = FieldProcess(tmp_path, "l1", {"red": "red"}, 180, dashboard=dashboard)
        process.begin(lambda: 2.1)
        observation = SimpleNamespace(rgb=rgb, timestamp_s=1.5)
        process.observe("sensors/0001.jpg", 1.5, [("red", block)], observation=observation)
        process.event({"event": "pregrasp", "color": "red", "attempt": 1})
        process.plan("red", [0.3, 0.1, 0.025], [0.5, 0.1, 0.027], 1.5)
        with urlopen(dashboard.url + "api/state", timeout=3) as response:
            state = json.load(response)
        assert state["camera"]["stamp_s"] == 2.0
        assert state["camera"]["rgb"].startswith("data:image/jpeg")
        assert state["camera"]["depth"].startswith("data:image/jpeg")
        assert state["segmentation"]["stamp_s"] == state["plan"]["source_s"] == 1.5
        assert state["segmentation"]["regions"] == [
            {"number": 1, "label": "red", "destination": "red"}
        ]
        assert state["plan"]["target_id"] == "L1-01"
    finally:
        dashboard.close()


def test_task_mask_view_receives_only_mapped_candidates(tmp_path):
    calls = []
    process = FieldProcess(
        tmp_path,
        "l2",
        {"banana": "yellow"},
        300,
        mask_observer=lambda frame, candidates: calls.append((frame, candidates)),
    )
    obs = SimpleNamespace(timestamp_s=4.0)
    mask = np.ones((4, 4), dtype=bool)
    process.show_masks(obs, [("other", mask), ("banana", mask)])
    assert calls == [(obs, [("banana", "yellow", mask)])]
