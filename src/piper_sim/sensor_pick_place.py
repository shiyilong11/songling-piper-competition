"""Pick/place with independently reachable grasp and release orientations.

Plans from the robot FK model, camera observations and encoder feedback.
Sampled table clearance covers the arm model and its configured camera mount.
"""

import itertools
import time
from dataclasses import replace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from piper_sim.demo_skill import PickPlaceSkill
from piper_sim.field_motion import TableClearanceError
from piper_sim.grasp_feedback import GraspEmptyError, GraspFeedbackError, WidthGuard
from piper_sim.sensor_ik import SensorIK


class EncoderConvergenceError(TimeoutError):
    """A joint target stayed outside tolerance while feedback remained available."""


class SensorPickPlaceSkill(PickPlaceSkill):
    ik_type = SensorIK

    def move(self, q, width):
        if not self.cfg.get("encoder_bias_correction", False):
            return super().move(q, width)
        # Bounded outer position loop, driven only by measured joint errors.
        # Compensates a small settled servo offset without changing the target
        # acceptance tolerance or reading payload/contact/gravity truth.
        bias = getattr(self, "_servo_bias", np.zeros(6)).copy()
        previous = self.b.measured()[:6]
        dt = 1 / self.b.interface["control"]["policy_hz"]
        started = self.b.data.time
        while True:
            command = np.clip(
                q + bias,
                self.b.interface["joints"]["lower_rad"],
                self.b.interface["joints"]["upper_rad"],
            )
            self.tick(np.r_[command, width])
            actual = self.b.measured()
            error = q - actual[:6]
            grip_ok = (
                self.b.previous_target[6] < 0.004 if width == 0 else abs(actual[6] - width) < 0.004
            )
            if np.max(np.abs(error)) < 0.010 and grip_ok:
                self._servo_bias = bias
                return
            if np.max(np.abs(error)) < 0.04 and np.max(np.abs(actual[:6] - previous)) / dt < 0.05:
                bias = np.clip(bias + 1.5 * dt * error, -0.02, 0.02)
            previous = actual[:6].copy()
            if self.b.data.time - started > 18:
                raise EncoderConvergenceError(
                    f"{self.stage}: encoder position did not converge with bounded correction "
                    f"(joint error {np.round(error, 4)} rad, width {actual[6]:.4f} m)"
                )

    def recheck_retention(self):
        """Stop first and require a new stationary encoder window; not optical proof."""
        self.guard = None
        self.hold_current()
        self.stage = "recheck_grip"
        widths, stamps, joints = [], [], []
        for _ in range(max(3, int(np.ceil(0.5 * self.b.interface["control"]["policy_hz"])))):
            self.tick(self.b.previous_target.copy())
            state = self.b.measured()
            widths.append(state[6])
            joints.append(state[:6])
            stamps.append(self.b.data.time)
        if (
            not np.isfinite(widths).all()
            or not np.isfinite(joints).all()
            or not np.isfinite(stamps).all()
        ):
            raise GraspFeedbackError("Invalid stationary grip feedback", "grip_feedback_invalid")
        if np.any(np.diff(stamps) <= 0) or stamps[-1] - stamps[0] < 0.3:
            raise GraspFeedbackError(
                "Insufficient fresh stationary feedback", "grip_feedback_invalid"
            )
        evidence = {
            "width_m": float(np.median(widths)),
            "width_span_m": float(np.ptp(widths)),
            "duration_s": float(stamps[-1] - stamps[0]),
            "joint_span_rad": float(np.max(np.ptp(joints, axis=0))),
            "evidence_status": "stationary_width_estimate_not_contact_proof",
        }
        if (
            max(widths) < self.feedback_cfg.get("empty_width_m", 0.006)
            and evidence["joint_span_rad"] <= 0.01
        ):
            raise GraspEmptyError(evidence=evidence)
        if (
            max(widths) > 0.071
            or np.ptp(widths) > self.feedback_cfg.get("close_stability_m", 0.004)
            or evidence["joint_span_rad"] > 0.01
        ):
            raise GraspFeedbackError(
                "Grip remains uncertain after stationary recheck", "grip_unstable", evidence
            )
        self.feedback("stationary_retention_rechecked", **evidence)
        self.guard = WidthGuard(evidence["width_m"], self.feedback_cfg)
        return evidence

    def pick_closed_loop(self, plan):
        try:
            return self.pick(plan)
        except GraspFeedbackError as exc:
            if exc.code != "grip_unstable":
                raise
        evidence = self.recheck_retention()
        self.stage = "test_lift"
        point = self.b.data.site_xpos[self.b.model.site("tcp").id].copy()
        point[2] = min(plan["hover"][2], max(point[2], plan["center"][2] + 0.04))
        self.cartesian(point, plan["yaw"], 0.0)
        self.hold(1.0)
        self.stage = "lift"
        self.cartesian(plan["hover"], plan["yaw"], 0.0)
        self.hold(0.2)
        return {
            "holding_estimate": True,
            "stationary_recheck": evidence,
            "feedback_events": list(self.feedback_events),
            "evidence_status": "width_based_estimate_not_contact_proof",
        }

    def place_closed_loop(self, plan):
        try:
            return self.place(plan)
        except GraspFeedbackError as exc:
            if exc.code != "grip_unstable":
                raise
        self.recheck_retention()
        return self.place(plan)

    def descend_for_release(self, place, yaw):
        try:
            return super().descend_for_release(place, yaw)
        except (TableClearanceError, EncoderConvergenceError, GraspFeedbackError) as exc:
            # A fruit can contact the support before the commanded TCP reaches
            # its nominal drop height. Only release if the held encoder pose
            # is already over the planned slot and allows guarded opening.
            path_guard = getattr(self.b, "path_guard", None)
            contact_feedback = (
                isinstance(exc, GraspFeedbackError)
                and exc.code == "grip_unstable"
                and self.cfg.get("allow_grip_change_at_release", False)
            )
            if isinstance(exc, GraspFeedbackError) and not contact_feedback:
                raise
            if path_guard is None and not contact_feedback:
                raise
            self.b.sync()
            site = self.b.model.site("tcp").id
            current = self.b.data.site_xpos[site].copy()
            actual_rotation = self.b.data.site_xmat[site].reshape(3, 3)
            desired_rotation = (
                Rotation.from_euler("z", yaw).as_matrix()
                @ Rotation.from_euler("y", self.ik.tilt).as_matrix()
                @ np.diag([1.0, -1.0, -1.0])
            )
            offset = current - place
            rotation_error = Rotation.from_matrix(desired_rotation @ actual_rotation.T).magnitude()
            release_error = {
                "release_tcp_offset_m": offset.tolist(),
                "release_rotation_error_rad": float(rotation_error),
            }
            if (
                np.linalg.norm(offset[:2]) > 0.006
                or not -0.003 <= offset[2] <= 0.030
                or rotation_error > 0.08
            ):
                if contact_feedback:
                    exc.evidence.update(release_error)
                    raise
                raise TableClearanceError(
                    f"Guarded release pose is outside the accepted window: "
                    f"offset={np.round(offset, 4)} m, "
                    f"rotation={rotation_error:.3f} rad"
                ) from exc
            actual = self.b.measured()
            if path_guard is not None:
                path_guard(actual, np.r_[actual[:6], 0.07])
            self.hold_current()
            self._servo_bias = np.zeros(6)
            self.feedback(
                "release_after_guarded_descent",
                actual_tcp_m=current.tolist(),
                planned_tcp_m=place.tolist(),
                reason=str(exc),
                **release_error,
            )
            return path_guard is not None

    def retreat_after_release(self, above, yaw, approach_q, guarded_release):
        if not guarded_release:
            return super().retreat_after_release(above, yaw, approach_q, guarded_release)
        # When descent met the table guard, reuse the already reached high
        # pose instead of asking IK for another low Cartesian waypoint.
        current = self.b.measured()
        end = np.r_[approach_q, 0.07]
        path_guard = self.b.path_guard
        try:
            path_guard(current, end)
        except TableClearanceError:
            return super().retreat_after_release(above, yaw, approach_q, guarded_release)
        data = mujoco.MjData(self.b.model)
        tcp = self.b.model.site("tcp").id
        count = max(2, int(np.ceil(np.max(np.abs(approach_q - current[:6])) / 0.03)) + 1)
        points = []
        for q in np.linspace(current[:6], approach_q, count):
            data.qpos[:6] = q
            mujoco.mj_kinematics(self.b.model, data)
            points.append(data.site_xpos[tcp].copy())
        points = np.asarray(points)
        if (
            np.any(np.diff(points[:, 2]) < -0.0003)
            or np.max(np.linalg.norm(points[:, :2] - points[0, :2], axis=1)) > 0.020
        ):
            return super().retreat_after_release(above, yaw, approach_q, guarded_release)
        self._servo_bias = np.zeros(6)
        self.feedback("retreat_to_measured_approach_pose", target_q_rad=approach_q.tolist())
        self.move_along_joint_segment(approach_q, 0.07)

    def validate_segment(self, start, end):
        path_guard = getattr(self.b, "path_guard", None)
        if path_guard is not None:
            path_guard(np.r_[start, 0.07], np.r_[end, 0.07])
            path_guard(np.r_[start, 0.0], np.r_[end, 0.0])

    def move_along_joint_segment(self, end, width):
        """Keep joint targets on the same interpolated path checked by the planner."""
        start = self.b.measured()[:6]
        self.validate_segment(start, end)
        self.hold_current()
        self._servo_bias = np.zeros(6)
        control = self.b.interface["control"]
        step = np.asarray(control["max_joint_speed_rad_s"]) / control["policy_hz"]
        count = max(1, int(np.ceil(np.max(np.abs(end - start) / step))))
        for q in np.linspace(start, end, count + 1)[1:]:
            self.tick(np.r_[q, width])
            if np.max(np.abs(q - self.b.measured()[:6])) > 0.07:
                self.move(q, width)
        self.move(end, width)

    def fixed_orientation_plan(self, target, destination, tray_floor, leg="both"):
        original = self.cfg
        tilts = original.get("grasp_tilt_candidates_rad", [0.6, 0.9, 0.35, 0.15])
        yaws = original.get(
            "grasp_yaw_candidates_rad",
            [target.yaw + np.pi, np.pi, target.yaw + np.pi / 2, target.yaw, target.yaw - np.pi / 2],
        )
        last = None
        try:
            for tilt in tilts:
                for yaw in yaws:
                    deadline = original.get("planning_deadline_monotonic")
                    if deadline is not None and time.monotonic() >= deadline:
                        raise RuntimeError("Sensor planning budget exhausted")
                    self.cfg = {
                        **original,
                        "grasp_tilt_candidates_rad": [tilt],
                        "grasp_yaw_candidates_rad": [yaw],
                    }
                    try:
                        plan = super().plan(target, destination, tray_floor)
                        pickup = [plan["hover"], plan["center"], plan["hover"]]
                        release = [plan["above"], plan["place"], plan["above"]]
                        points = {"pickup": pickup, "release": release, "both": pickup + release}[
                            leg
                        ]
                        actual = self.b.measured()[:6]
                        seed = self.ik.solve(points[0], yaw, actual)
                        if leg != "release":
                            self.validate_segment(actual, seed)
                        for start, end in itertools.pairwise(points):
                            steps = max(1, int(np.ceil(np.linalg.norm(end - start) / 0.02)))
                            for point in np.linspace(start, end, steps + 1)[1:]:
                                new_seed = self.ik.solve(point, yaw, seed)
                                self.validate_segment(seed, new_seed)
                                seed = new_seed
                        return plan
                    except RuntimeError as exc:
                        last = exc
            raise RuntimeError(f"No sampled Cartesian path: {last}")
        finally:
            self.cfg = original

    def plan(self, target, destination, tray_floor):
        requested_drop_yaw = self.cfg.get("place_yaw_rad")
        if not self.cfg.get("prefer_independent_orientation", False) and (
            requested_drop_yaw is None or abs(np.sin(requested_drop_yaw - target.yaw)) < 1e-6
        ):
            try:
                return self.fixed_orientation_plan(target, destination, tray_floor)
            except RuntimeError:
                pass
        # A single fixed wrist orientation can be reachable at the pickup but
        # outside the wrist limits at the goal. Solve each end independently.
        cache = getattr(self, "_pickup_cache", {})
        key = tuple(np.r_[target.center, target.yaw, self.b.measured()[:6]])
        if key not in cache:
            # Near the base, the grasp can be reachable while a tall vertical
            # column at the same XY is not. Use a local pregrasp height, then
            # reorient along the independently checked transport path.
            original_height = self.cfg.get("transport_height_m", 0.22)
            heights = [original_height]
            heights += [
                max(self.table + 0.10, target.center[2] + dz)
                for dz in self.cfg.get("local_pickup_lift_candidates_m", [0.10, 0.08])
            ]
            try:
                for height in dict.fromkeys(heights):
                    self.cfg["transport_height_m"] = height
                    try:
                        pickup = self.fixed_orientation_plan(
                            target, target.center[:2], self.table, leg="pickup"
                        )
                        pickup["local_pickup_hover"] = height < original_height - 1e-6
                        cache[key] = (pickup, self.ik)
                        break
                    except RuntimeError as exc:
                        cache[key] = exc
            finally:
                self.cfg["transport_height_m"] = original_height
        if isinstance(cache[key], RuntimeError):
            raise cache[key]
        pickup, pickup_ik = cache[key]
        old_yaws = self.cfg.get("grasp_yaw_candidates_rad")
        old_tilts = self.cfg.get("grasp_tilt_candidates_rad")
        transport_height = self.cfg.get("transport_height_m", 0.22)
        try:
            # Preserve the pickup orientation as far as reachability permits.
            # Starting every drop at a fixed world yaw unnecessarily rolls fruit
            # inside the fingers and invalidates the rigid-placement estimate.
            drop_yaw = pickup["yaw"] if requested_drop_yaw is None else requested_drop_yaw
            self.cfg["grasp_yaw_candidates_rad"] = sorted(
                [drop_yaw, drop_yaw + np.pi],
                key=lambda y: abs((y - pickup["yaw"] + np.pi) % (2 * np.pi) - np.pi),
            )
            if requested_drop_yaw is None:
                self.cfg["grasp_yaw_candidates_rad"] += [drop_yaw + np.pi / 2, drop_yaw - np.pi / 2]
            self.cfg["grasp_tilt_candidates_rad"] = sorted(
                old_tilts or [0.6, 0.9, 0.35, 0.15], key=lambda t: abs(t - pickup["tilt"])
            )
            virtual = replace(
                target, center=np.r_[destination, tray_floor + target.height / 2], yaw=target.yaw
            )
            release_hover = self.cfg.get("release_hover_above_table_m")
            if release_hover is not None:
                self.cfg["transport_height_m"] = min(
                    transport_height, max(self.table, tray_floor) + release_hover
                )
            drop = self.fixed_orientation_plan(virtual, destination, tray_floor, leg="release")
            drop_ik = self.ik
        finally:
            self.cfg["transport_height_m"] = transport_height
            if old_yaws is not None:
                self.cfg["grasp_yaw_candidates_rad"] = old_yaws
            else:
                self.cfg.pop("grasp_yaw_candidates_rad", None)
            if old_tilts is not None:
                self.cfg["grasp_tilt_candidates_rad"] = old_tilts
            else:
                self.cfg.pop("grasp_tilt_candidates_rad", None)
        qpick = pickup_ik.solve(pickup["hover"], pickup["yaw"], self.b.measured()[:6])
        qdrop = drop_ik.solve(drop["above"], drop["yaw"], qpick)
        self.validate_segment(qpick, qdrop)
        data = mujoco.MjData(self.b.model)
        for q in np.linspace(qpick, qdrop, 61):
            data.qpos[:6] = q
            mujoco.mj_kinematics(self.b.model, data)
            if data.site_xpos[self.b.model.site("tcp").id, 2] < self.table + 0.09:
                raise RuntimeError("Reorientation path fails sampled TCP clearance")
        self.ik = pickup_ik
        return {
            **pickup,
            "place": drop["place"],
            "above": drop["above"],
            "place_yaw": drop["yaw"],
            "place_tilt": drop["tilt"],
            "qdrop": qdrop,
            "reorientation": True,
        }

    def recover_empty(self):
        if (
            self.last_plan is not None
            and self.last_plan.get("local_pickup_hover")
            and self.stage in {"approach", "descend", "close", "check_grip", "test_lift", "lift"}
        ):
            self.guard = None
            self.stage = "recover_lift"
            self.cartesian(self.last_plan["hover"], self.last_plan["yaw"], 0.0)
            self.move(self.b.measured()[:6], 0.07)
            self.feedback("empty_recovery_retracted", local_hover=True)
            return
        return super().recover_empty()

    def place(self, plan):
        if not plan.get("reorientation"):
            return super().place(plan)
        self.stage = "reorient_transport"
        self.feedback(
            "transport_orientation_change",
            source="robot_only_IK_and_encoders",
            complete_collision_check=False,
        )
        self.ik = self.ik_type(self.b, tilt=plan["place_tilt"])
        self.last_plan = {**plan, "yaw": plan["place_yaw"]}
        if getattr(self.b, "path_guard", None) is not None or self.cfg.get(
            "follow_planned_transport_segment", False
        ):
            self.move_along_joint_segment(plan["qdrop"], 0.0)
        else:
            self.move(plan["qdrop"], 0.0)
        return super().place({**plan, "yaw": plan["place_yaw"]})
