from types import SimpleNamespace

import numpy as np
import pytest

from piper_sim.demo_skill import PickPlaceSkill, TargetEstimate
from piper_sim.grasp_feedback import GraspFeedbackError, WidthGuard, match_fresh_target
from piper_sim.sensor_pick_place import SensorPickPlaceSkill
from piper_sim.skill_runtime import SkillCoordinator, SkillResult


def target(x=0.4, rgb=(200, 20, 20)):
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    return TargetEstimate(
        np.array([x, 0, 0.085]),
        0.05,
        0,
        mask,
        np.zeros((20, 3)),
        [20, 20],
        0,
        np.asarray(rgb, dtype=float),
    )


def test_pregrasp_association_corrects_small_motion_but_rejects_swaps_and_ambiguity():
    previous, fresh = target(), target(0.405)
    assert match_fresh_target(previous, [fresh], {})[0] is fresh
    for candidates in (
        [],
        [target(0.43)],
        [target(0.405), target(0.41)],
        [target(0.4, (20, 200, 20))],
    ):
        with pytest.raises(GraspFeedbackError) as caught:
            match_fresh_target(previous, candidates, {})
        assert caught.value.code == "target_changed" and caught.value.retryable


def test_retention_ignores_one_bad_sample_but_stops_on_sustained_loss():
    monitor = WidthGuard(0.05, {})
    for t, w in (
        (0, 0.05),
        (0.05, 0.002),
        (0.10, 0.05),
        (0.15, 0.05),
        (0.20, 0.002),
        (0.25, 0.002),
    ):
        monitor.update(w, t, "transport")
    with pytest.raises(GraspFeedbackError) as caught:
        monitor.update(0.002, 0.30, "transport")
    assert caught.value.code == "grip_lost" and caught.value.retryable
    assert caught.value.evidence["stage"] == "transport"


def test_width_change_is_uncertain_and_does_not_authorize_automatic_release():
    monitor = WidthGuard(0.05, {})
    monitor.update(0.037, 0, "test_lift")
    with pytest.raises(GraspFeedbackError) as caught:
        monitor.update(0.037, 0.15, "test_lift")
    assert caught.value.code == "grip_unstable"
    assert not caught.value.retryable
    with pytest.raises(GraspFeedbackError) as invalid:
        WidthGuard(0.05, {}).update(float("nan"), 1, "lift")
    assert invalid.value.code == "grip_feedback_invalid"


def test_transport_fault_cancels_remaining_path_and_never_reaches_release(monkeypatch):
    commands, stages = [], []
    measured = np.r_[np.zeros(6), 0.05]
    backend = SimpleNamespace(
        data=SimpleNamespace(time=0.0),
        previous_target=measured.copy(),
        interface={},
        measured=lambda: measured.copy(),
        _set_target=lambda goal: commands.append(goal.copy()),
    )

    def step(goal):
        backend.data.time += 0.05
        measured[:6] += 0.001
        measured[6] = 0.002

    motion = PickPlaceSkill(backend, 0.06, stages.append, {}, stepper=step)
    monkeypatch.setattr("piper_sim.demo_skill.target_to_action", lambda goal, *args: goal)
    motion.guard = WidthGuard(0.05, {})

    def move(*args):
        for _ in range(10):
            motion.tick(np.r_[np.ones(6), 0])

    motion.cartesian = move
    with pytest.raises(GraspFeedbackError) as caught:
        motion.place({"place": np.zeros(3), "above": np.zeros(3), "yaw": 0})
    assert caught.value.code == "grip_lost"
    assert stages and set(stages) == {"transport"}
    assert backend.data.time < 0.25
    assert commands[-1][6] == 0
    np.testing.assert_allclose(commands[-1][:6], measured[:6])


@pytest.mark.parametrize("offset_x,allow", [(0.002, True), (0.020, False)])
def test_release_descent_checks_actual_pose_on_width_change(offset_x, allow):
    place = np.array([0.5, 0.0, 0.035])
    commands = []
    state = np.r_[np.zeros(6), 0.02]
    backend = SimpleNamespace(
        data=SimpleNamespace(
            time=1.0,
            site_xpos=np.array([place + [offset_x, 0.0, 0.012]]),
            site_xmat=np.array([np.diag([1.0, -1.0, -1.0]).ravel()]),
        ),
        model=SimpleNamespace(site=lambda name: SimpleNamespace(id=0)),
        interface={},
        step=lambda action: None,
        sync=lambda: None,
        measured=lambda: state.copy(),
        previous_target=np.zeros(7),
        _set_target=lambda command: commands.append(command),
    )
    motion = SensorPickPlaceSkill(
        backend, 0, lambda stage: None, {"allow_grip_change_at_release": True}
    )
    motion.ik = SimpleNamespace(tilt=0)

    def unstable(*args):
        raise GraspFeedbackError("width changed", "grip_unstable")

    motion.cartesian = unstable
    if allow:
        assert motion.descend_for_release(place, 0) is False
        assert len(commands) == 1 and commands[0][6] == 0
        assert motion.feedback_events[-1]["event"] == "release_after_guarded_descent"
    else:
        with pytest.raises(GraspFeedbackError) as caught:
            motion.descend_for_release(place, 0)
        assert caught.value.evidence["release_tcp_offset_m"][0] == pytest.approx(offset_x)
        assert not commands


def test_transport_loss_reenters_observation_before_retry_and_skips_false_verification():
    calls = []

    class Skills:
        port = SimpleNamespace(bind_evaluator=lambda target: "eval-only")

        def __init__(self):
            self.trace = []

        def call(self, name, **args):
            calls.append(name)
            if name == "resolve_target":
                return SkillResult("ok", {"target": "sensor-target"})
            if name == "place_object" and calls.count(name) == 1:
                return SkillResult("retryable", code="grip_lost")
            if name == "recover_or_continue":
                assert args["failure"].code == "grip_lost"
                return SkillResult("ok", {"next_view": 0})
            if name == "verify_result":
                return SkillResult("ok", {"success": True})
            return SkillResult("ok")

    assert SkillCoordinator(Skills()).run({})["success"]
    assert calls.count("place_object") == 2
    assert calls.count("verify_result") == 1
    index = calls.index("recover_or_continue")
    assert calls[index + 1 : index + 3] == ["observe_scene", "detect_objects"]
