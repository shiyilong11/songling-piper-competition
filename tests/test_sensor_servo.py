from types import SimpleNamespace

import numpy as np
import pytest

from piper_sim.sensor_pick_place import SensorPickPlaceSkill


def motion(offset):
    m = object.__new__(SensorPickPlaceSkill)
    m.cfg = {"encoder_bias_correction": True}
    m.stage = "reorient_transport"
    desired = np.array([0.0, 0.4, -0.8, 0.0, 0.6, 0.0])
    state = np.r_[desired - offset, 0.07]
    m.b = SimpleNamespace(
        data=SimpleNamespace(time=0.0),
        previous_target=state.copy(),
        measured=lambda: state.copy(),
        interface={
            "control": {"policy_hz": 20},
            "joints": {"lower_rad": [-2.0] * 6, "upper_rad": [2.0] * 6},
        },
    )
    commands = []

    def tick(command):
        commands.append(command.copy())
        m.b.previous_target = command.copy()
        state[:] = command
        state[:6] -= offset
        m.b.data.time += 0.05

    m.tick = tick
    return m, desired, commands


def test_small_settled_encoder_offset_corrected_without_loosening_acceptance():
    m, desired, commands = motion(np.array([0.0, 0.012, 0.0, 0.0, 0.0, 0.0]))
    m.move(desired, 0.07)
    assert np.max(np.abs(desired - m.b.measured()[:6])) < 0.01
    assert np.max(np.abs(np.asarray(commands)[:, :6] - desired)) <= 0.02


def test_large_tracking_error_is_not_integrated_away():
    m, desired, commands = motion(np.full(6, 0.08))
    with pytest.raises(TimeoutError):
        m.move(desired, 0.07)
    assert np.allclose(np.asarray(commands)[:, :6], desired)


def test_guarded_transport_commands_follow_the_checked_joint_line():
    m, start, commands = motion(np.zeros(6))
    rates = np.array([0.3, 0.3, 0.3, 0.4, 0.4, 0.4])
    m.b.interface["control"]["max_joint_speed_rad_s"] = rates
    m.hold_current = lambda: None
    delta = np.array([0.28, 0.08, -0.20, -0.12, 0.05, 0.11])
    m.move_along_joint_segment(start + delta, 0.07)
    path = np.vstack([start, np.asarray(commands)[:, :6]])
    phase = (path[:, 0] - start[0]) / delta[0]
    assert np.allclose(path, start + phase[:, None] * delta)
    assert np.all(np.abs(np.diff(path, axis=0)) <= rates / 20 + 1e-10)
    assert np.allclose(path[-1], start + delta)
