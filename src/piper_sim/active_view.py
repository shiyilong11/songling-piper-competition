"""Bounded wrist-camera aiming from visible RGB-D points and robot kinematics."""

import mujoco
import numpy as np
from scipy.optimize import least_squares

from piper_sim.calibration import body_pose, validate_transform


def recenter_camera(backend, observation, points_base, table_top, config):
    """Return a new joint target; never use scene object poses or mutate live state.

    Partial geometry is sufficient to aim the camera, but must be observed again
    before it is used for grasping. This is a local view adjustment, not a general
    collision-free motion planner.
    """
    points = np.asarray(points_base, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 15
        or not np.isfinite(points).all()
    ):
        raise ValueError("有效深度点不足，无法调整观察方向。")
    anchor = np.median(points, axis=0)
    seed = np.asarray(observation.joint_position_rad, dtype=float)
    if np.max(np.abs(seed - backend.measured()[:6])) > 0.02:
        raise ValueError("观察姿态已变化，拒绝使用旧帧调整相机。")
    extrinsic = validate_transform(backend.camera["hand_eye"]["T_end_camera"])
    initial = validate_transform(observation.T_base_end) @ extrinsic
    initial_point = initial[:3, :3].T @ (anchor - initial[:3, 3])
    if initial_point[2] <= 0:
        raise ValueError("观察目标不在相机前方。")
    desired_depth = max(
        initial_point[2] + config.get("backoff_m", 0.025),
        backend.camera["camera"]["depth"]["min_valid_m"] + 0.06,
    )
    limit = config.get("max_joint_step_rad", 0.35)
    lower = np.maximum(backend.interface["joints"]["lower_rad"], seed - limit) + 1e-6
    upper = np.minimum(backend.interface["joints"]["upper_rad"], seed + limit) - 1e-6
    data = mujoco.MjData(backend.model)
    # Only robot joint state is needed. Object free-joint state is never read.
    data.qpos[backend.fingers] = [0.035, -0.035]

    def pose(q):
        data.qpos[backend.qindices] = q
        mujoco.mj_kinematics(backend.model, data)
        return body_pose(backend.model, data, backend.interface["frames"]["end"]) @ extrinsic

    def residual(q):
        transform = pose(q)
        p = transform[:3, :3].T @ (anchor - transform[:3, 3])
        return np.r_[
            p[:2],
            0.65 * (p[2] - desired_depth),
            0.12 * (transform[:3, 3] - initial[:3, 3]),
            0.008 * (q - seed),
        ]

    starts = [seed]
    if config.get("allow_depth_backoff", False):
        starts += [seed + [0, 0.6, -0.6, 0, -0.3, 0], seed + [0, 1.0, -1.0, 0, -0.6, 0]]
    solutions = [
        least_squares(
            residual,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            max_nfev=100,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        for start in starts
    ]
    solution = min(solutions, key=lambda s: np.linalg.norm(residual(s.x)))
    transform = pose(solution.x)
    p = transform[:3, :3].T @ (anchor - transform[:3, 3])
    before = np.linalg.norm(initial_point[:2] / initial_point[2])
    after = np.linalg.norm(p[:2] / max(p[2], 1e-6))
    backed_off = (
        config.get("allow_depth_backoff", False) and p[2] > initial_point[2] + 0.03 and after < 0.10
    )
    if p[2] < backend.camera["camera"]["depth"]["min_valid_m"] or (
        after > before * 0.75 and not backed_off
    ):
        raise ValueError("限幅内无法改善观察位置，切换备用视角。")
    # Check sampled joint interpolation keeps the wrist and TCP above the table.
    # Does not claim obstacle/self-collision checking; execution remains in simulation.
    clearance = config.get("min_tcp_clearance_m", 0.10)
    for q in np.linspace(seed, solution.x, 21):
        camera = pose(q)
        tcp_z = data.site_xpos[backend.model.site("tcp").id][2]
        if tcp_z < table_top + clearance or camera[2, 3] < table_top + clearance:
            raise ValueError("调整路径过低，切换备用观察位。")
    return solution.x, {
        "source": "visible_wrist_depth_and_robot_kinematics",
        "anchor_base_m": anchor.tolist(),
        "normalized_image_error_before": float(before),
        "normalized_image_error_after": float(after),
        "predicted_depth_m": float(p[2]),
        "joint_target_rad": solution.x.tolist(),
    }
