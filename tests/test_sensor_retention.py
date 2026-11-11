from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from piper_sim.grasp_feedback import GraspEmptyError, GraspFeedbackError
from piper_sim.sensor_pick_place import SensorPickPlaceSkill


def motion(widths, fresh=True):
    m = object.__new__(SensorPickPlaceSkill)
    state = np.r_[np.zeros(6), widths[0]]
    m.b = SimpleNamespace(
        interface={"control": {"policy_hz": 20}},
        data=SimpleNamespace(time=0.0),
        previous_target=np.zeros(7),
        measured=lambda: state.copy(),
    )
    m.hold_current = Mock()
    m.feedback = Mock()
    m.feedback_cfg = {}
    iterator = iter(widths)

    def tick(goal):
        state[6] = next(iterator)
        m.b.data.time += 0.05 if fresh else 0.0

    m.tick = tick
    return m


def test_retention_resume_requires_stationary_new_measurements():
    m = motion(np.linspace(0.050, 0.049, 10))
    result = m.recheck_retention()
    m.hold_current.assert_called_once()
    assert result["duration_s"] >= 0.3
    assert m.guard.reference == pytest.approx(0.0495)
    assert "not_contact_proof" in result["evidence_status"]


@pytest.mark.parametrize(
    "widths,fresh,error",
    [
        (np.zeros(10), True, GraspEmptyError),
        (np.full(10, np.nan), True, GraspFeedbackError),
        (np.linspace(0.05, 0.04, 10), True, GraspFeedbackError),
        (np.full(10, 0.05), False, GraspFeedbackError),
    ],
)
def test_invalid_empty_stale_or_continually_changing_width_never_resumes(widths, fresh, error):
    m = motion(widths, fresh)
    with pytest.raises(error):
        m.recheck_retention()
    assert m.guard is None


@pytest.mark.parametrize("stage", ["test_lift", "reorient_transport"])
def test_local_hover_recovery_is_only_for_pickup_not_return_from_drop(stage, monkeypatch):
    from piper_sim.demo_skill import PickPlaceSkill

    m = motion(np.full(10, 0.03))
    m.stage = stage
    m.last_plan = {"local_pickup_hover": True, "hover": np.array([0.27, 0.05, 0.175]), "yaw": 2.36}
    m.cartesian, m.move = Mock(), Mock()
    fallback = Mock()
    monkeypatch.setattr(PickPlaceSkill, "recover_empty", fallback)
    m.recover_empty()
    if stage == "test_lift":
        assert np.allclose(m.cartesian.call_args.args[0], m.last_plan["hover"])
        assert m.move.call_args.args[1] == 0.07
        fallback.assert_not_called()
    else:
        fallback.assert_called_once()
        m.cartesian.assert_not_called()
