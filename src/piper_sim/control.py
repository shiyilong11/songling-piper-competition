"""Shared action semantics; NumPy for deployment and batched Torch for GPU training."""

from pathlib import Path

import numpy as np
import yaml

from piper_sim.calibration import ROOT


def load_interface(path=None):
    path = Path(path) if path else ROOT / "config/interface.yaml"
    cfg = yaml.safe_load(path.read_text())
    if cfg["schema_version"] != 1 or len(cfg["joints"]["names"]) != 6:
        raise ValueError("Expected schema 1 and six arm joints")
    low = np.asarray(cfg["joints"]["lower_rad"])
    high = np.asarray(cfg["joints"]["upper_rad"])
    speeds = np.asarray(cfg["control"]["max_joint_speed_rad_s"])
    if low.shape != (6,) or high.shape != (6,) or speeds.shape != (6,):
        raise ValueError("Joint limits/speeds must have length six")
    values = np.r_[low, high, speeds]
    c = cfg["control"]
    if not np.isfinite(values).all() or (low >= high).any() or (speeds <= 0).any():
        raise ValueError("Invalid joint limits/speeds")
    if c["policy_hz"] <= 0 or c["physics_hz"] % c["policy_hz"]:
        raise ValueError("Physics frequency must be an integer multiple of policy frequency")
    width = c["gripper_width_range_m"]
    if not (0 <= width[0] < width[1] <= 0.07):
        raise ValueError("Gripper width exceeds current URDF capability")
    if c["max_gripper_speed_m_s"] <= 0 or c["max_target_tracking_error_rad"] <= 0:
        raise ValueError("Rates and tracking bound must be positive")
    return cfg


def action_to_target(action, measured, previous_target, cfg):
    """(...,7): six radians + full gripper width in metres. No ROS dependency.

    Delta targets accumulate from previous command, bounded around measured position.
    Gripper action specifies desired absolute width, then rate limits it.
    """
    a, q, prev = [np.asarray(v, dtype=np.float64) for v in (action, measured, previous_target)]
    if a.shape != q.shape or q.shape != prev.shape or a.shape[-1] != 7:
        raise ValueError("action/state/previous must have matching (...,7) shapes")
    if not all(np.isfinite(v).all() for v in (a, q, prev)):
        raise ValueError("Action and robot state must be finite")
    a = np.clip(a, -1, 1)
    c = cfg["control"]
    dt = 1 / c["policy_hz"]
    dq = np.asarray(c["max_joint_speed_rad_s"]) * dt
    target = prev.copy()
    error = c["max_target_tracking_error_rad"]
    # Only advance from a target close enough to the measured state.
    anchor = np.clip(prev[..., :6], q[..., :6] - error, q[..., :6] + error)
    target[..., :6] = np.clip(anchor + a[..., :6] * dq, q[..., :6] - error, q[..., :6] + error)
    target[..., :6] = np.clip(
        target[..., :6], cfg["joints"]["lower_rad"], cfg["joints"]["upper_rad"]
    )
    lo, hi = c["gripper_width_range_m"]
    desired = lo + (a[..., 6] + 1) * 0.5 * (hi - lo)
    step = c["max_gripper_speed_m_s"] * dt
    target[..., 6] = np.clip(prev[..., 6] + np.clip(desired - prev[..., 6], -step, step), lo, hi)
    return target


def action_to_target_torch(action, measured, previous_target, cfg):
    """Same semantics on (...,7) GPU tensors, without moving data to CPU."""
    import torch

    if (
        action.shape != measured.shape
        or action.shape != previous_target.shape
        or action.shape[-1] != 7
    ):
        raise ValueError("Expected matching (...,7) tensors")
    if not all(torch.isfinite(x).all() for x in (action, measured, previous_target)):
        raise ValueError("Non-finite action/state")

    def tensor(value):
        return torch.as_tensor(value, device=action.device, dtype=action.dtype)

    c = cfg["control"]
    a = action.clamp(-1, 1)
    error = c["max_target_tracking_error_rad"]
    q = measured[..., :6]
    anchor = torch.clamp(previous_target[..., :6], q - error, q + error)
    arm = torch.clamp(
        anchor + a[..., :6] * tensor(c["max_joint_speed_rad_s"]) / c["policy_hz"],
        q - error,
        q + error,
    )
    arm = torch.clamp(arm, tensor(cfg["joints"]["lower_rad"]), tensor(cfg["joints"]["upper_rad"]))
    lo, hi = c["gripper_width_range_m"]
    desired = lo + (a[..., 6:7] + 1) * 0.5 * (hi - lo)
    step = c["max_gripper_speed_m_s"] / c["policy_hz"]
    width = (
        previous_target[..., 6:7] + (desired - previous_target[..., 6:7]).clamp(-step, step)
    ).clamp(lo, hi)
    return torch.cat([arm, width], dim=-1)


def canonical_state(names, positions, velocities, cfg):
    """Reorder ROS JointState by name; gripper position is full opening in metres."""
    required = cfg["joints"]["names"] + ["gripper"]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate joint names")
    if any(n not in names for n in required) or len(positions) != len(names):
        raise ValueError("Incomplete joint feedback")
    indices = [names.index(n) for n in required]
    q = np.asarray(positions, dtype=float)[indices]
    # Official feedback can have six velocities for seven positions.
    if len(velocities) < 6:
        raise ValueError("Six arm velocities required")
    arm_indices = indices[:6]
    if max(arm_indices) >= len(velocities):
        raise ValueError("Velocity names/indices inconsistent")
    dq = np.asarray(velocities, dtype=float)[arm_indices]
    if not np.isfinite(q).all() or not np.isfinite(dq).all():
        raise ValueError("Non-finite feedback")
    return q, dq


def driver_command(target, cfg):
    """Payload for official JointState command, preserving width as seventh position.

    Official humble driver uses velocity[6] as a speed percentage (not SI velocity).
    This peculiarity stays strictly at the driver boundary.
    """
    q = np.asarray(target, dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError("Expected a finite seven-element target")
    if (q[:6] < np.asarray(cfg["joints"]["lower_rad"]) - 1e-8).any() or (
        q[:6] > np.asarray(cfg["joints"]["upper_rad"]) + 1e-8
    ).any():
        raise ValueError("Target outside joint limits")
    lo, hi = cfg["control"]["gripper_width_range_m"]
    if not lo - 1e-8 <= q[6] <= hi + 1e-8:
        raise ValueError("Invalid gripper width")
    r = cfg["ros2"]
    if r["driver_gripper_multiplier"] != 1.0:
        raise ValueError("This contract requires official gripper_val_mutiple=1")
    return {
        "name": cfg["joints"]["names"] + ["gripper"],
        "position": q.tolist(),
        "velocity": [0.0] * 6 + [float(r["driver_speed_percent"])],
        "effort": [0.0] * 6 + [float(r["gripper_effort"])],
    }


def target_to_action(desired_target, previous_target, cfg):
    """Let a traditional planner use the same normalized action/rate-limited path.

    Arm targets are absolute radians; gripper target is absolute full opening in metres.
    Call every policy tick until feedback confirms arrival; this is not a trajectory planner.
    """
    goal = np.asarray(desired_target, dtype=float)
    prev = np.asarray(previous_target, dtype=float)
    if goal.shape != prev.shape or goal.shape[-1] != 7:
        raise ValueError("Expected matching (...,7) targets")
    if not np.isfinite(goal).all() or not np.isfinite(prev).all():
        raise ValueError("Targets must be finite")
    c = cfg["control"]
    a = np.empty_like(goal)
    a[..., :6] = (goal[..., :6] - prev[..., :6]) / (
        np.asarray(c["max_joint_speed_rad_s"]) / c["policy_hz"]
    )
    lo, hi = c["gripper_width_range_m"]
    a[..., 6] = 2 * (goal[..., 6] - lo) / (hi - lo) - 1
    return np.clip(a, -1, 1)
