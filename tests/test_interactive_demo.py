import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from interactive_demo import Cancelled, Demo

from piper_sim.demo_skill import TargetEstimate, select_pixel, select_rule
from piper_sim.l3_competition import CARDS


L3_ROUND = "\n".join(CARDS[number][0] for number in (1, 2, 3))


def controller():
    demo = Demo.__new__(Demo)
    demo.lock = threading.RLock()
    demo.commands = queue.Queue()
    demo.stop_flag = threading.Event()
    demo.pause_flag = threading.Event()
    demo.state_data = {"busy": False}
    return demo


def test_stop_prevents_any_further_physics_and_double_submission():
    demo = controller()
    request = {"instruction": L3_ROUND}
    demo.submit("run", request)
    with pytest.raises(ValueError, match="尚未结束"):
        demo.submit("run", request)
    demo.submit("pause", {})
    demo.submit("stop", {})
    assert not demo.pause_flag.is_set()
    # No world exists: cancellation must happen before any attempted physics access.
    with pytest.raises(Cancelled):
        demo.advance(np.zeros(7))


def test_time_limit_fails_instead_of_spinning_without_simulation_progress():
    demo = controller()
    demo.world = SimpleNamespace(elapsed_s=180, task="l1")
    demo.scene_cfg = {"tasks": {"l1": {"time_limit_s": 180}}}
    with pytest.raises(TimeoutError):
        demo.advance(np.zeros(7))


def test_invalid_input_does_not_reserve_or_queue_an_execution():
    demo = controller()
    for request in (
        {"instruction": "", "backend": "rule"},
        {"instruction": "pick", "backend": "unknown"},
    ):
        with pytest.raises(ValueError):
            demo.submit("run", request)
    assert not demo.state_data["busy"] and demo.commands.empty()


def test_failed_task_can_start_again_without_scene_reset():
    demo = controller()
    demo.state_data.update(needs_reset=True, stage="failed")
    demo.stop_flag.set()
    demo.submit("run", {"instruction": CARDS[2][0]})
    action, request = demo.commands.get_nowait()
    assert action == "run" and request["cards"] == (2,)
    assert not demo.stop_flag.is_set()


def test_random_reset_generates_seed_and_preserves_explicit_seed(monkeypatch):
    demo = controller()
    demo.state_data["scene"] = "fruit"
    monkeypatch.setattr("interactive_demo.secrets.randbelow", lambda limit: 12345)
    demo.submit("reset", {})
    assert demo.commands.get_nowait() == (
        "reset",
        {"scene": "fruit", "seed": 12345, "task_mode": "l3"},
    )
    demo.state_data["busy"] = False
    demo.submit("reset", {"scene": "blocks", "seed": 42})
    assert demo.commands.get_nowait()[1]["seed"] == 42


def test_new_attempt_clears_judging_without_resetting_physical_state():
    from piper_sim.competition_world import CompetitionWorld

    world = CompetitionWorld.__new__(CompetitionWorld)
    world.backend = SimpleNamespace(data=SimpleNamespace(time=350.0, qpos=np.array([0.2, 0.3])))
    world.objects = [{"id": "object_1"}]
    world.events = {"object_1": {"grasped": True, "placed": ["green"]}}
    world.begin_attempt()
    assert world.elapsed_s == 0
    assert world.backend.data.time == 350
    np.testing.assert_array_equal(world.backend.data.qpos, [0.2, 0.3])
    assert not world.events["object_1"]["grasped"]
    assert world.events["object_1"]["placed"] == []


def test_failed_random_reset_preserves_existing_world(monkeypatch, tmp_path):
    demo = controller()

    def must_not_close():
        raise AssertionError("Existing scene must survive failed replacement")

    old_world = SimpleNamespace(close=must_not_close)
    demo.world = old_world
    demo.output = tmp_path
    demo.cfg = {}
    demo.scene_cfg = {"spawn": {}}
    demo.scene, demo.seed, demo.task_mode = "blocks", 42, "l1"

    class FailingWorld:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self, seed):
            raise RuntimeError("packing failed")

        def close(self):
            pass

    monkeypatch.setattr("interactive_demo.CompetitionWorld", FailingWorld)
    with pytest.raises(RuntimeError, match="packing failed"):
        demo.reset_scene(123, "fruit", "l2")
    assert demo.world is old_world
    assert (demo.scene, demo.seed, demo.task_mode) == ("blocks", 42, "l1")


def test_point_must_be_near_visible_depth_and_rule_rejects_duplicate_targets():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:20, 10:20] = True
    candidate = TargetEstimate(
        np.array([0.4, 0, 0.08]), 0.04, 0, mask, np.empty((0, 3)), [15, 15], 0
    )
    assert select_pixel([candidate], [15, 15]) is candidate
    with pytest.raises(ValueError, match="过远"):
        select_pixel([candidate], [90, 90])
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[mask, 0] = 200
    with pytest.raises(ValueError, match="唯一"):
        select_rule([candidate, candidate], rgb, "把红色方块放到蓝色托盘")
