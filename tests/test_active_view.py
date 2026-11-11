import mujoco
import numpy as np
import pytest

from piper_sim.active_view import recenter_camera
from piper_sim.backend import MujocoBackend
from piper_sim.calibration import camera_to_base


def test_offcenter_depth_target_is_centered_without_mutating_live_robot():
    backend = MujocoBackend()
    try:
        backend.reset()
        backend.data.qpos[backend.qindices] = [0.15, 0.4, -0.8, 0, 1.2, 0]
        mujoco.mj_forward(backend.model, backend.data)
        obs = backend.observe()
        points = camera_to_base(
            np.tile([0.12, -0.09, 0.34], (20, 1)), obs.T_base_end, backend.camera
        )
        original = backend.data.qpos.copy()
        q, result = recenter_camera(backend, obs, points, 0.06, {})
        np.testing.assert_array_equal(backend.data.qpos, original)
        assert np.max(np.abs(q - obs.joint_position_rad)) <= 0.35
        assert (
            result["normalized_image_error_after"] < result["normalized_image_error_before"] * 0.25
        )
        assert result["predicted_depth_m"] >= backend.camera["camera"]["depth"]["min_valid_m"]
        backend.data.qpos[backend.qindices[0]] += 0.05
        with pytest.raises(ValueError, match="旧帧"):
            recenter_camera(backend, obs, points, 0.06, {})
    finally:
        backend.close()


def test_invalid_depth_cannot_produce_an_active_view():
    with pytest.raises(ValueError, match="深度点不足"):
        recenter_camera(None, None, np.full((20, 3), np.nan), 0.06, {})
