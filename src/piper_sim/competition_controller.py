"""L1 controller receiving only a robot port, RGB-D and public configuration.

Import boundary: no simulator world, evaluator, contact or scene configuration.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from piper_sim.active_view import recenter_camera
from piper_sim.competition_sensor import PlacementCheck, SensorScene
from piper_sim.grasp_feedback import GraspFeedbackError, match_fresh_target
from piper_sim.sensor_pick_place import SensorPickPlaceSkill


def placement_candidates(zone, margin=0.045):
    center = zone.center
    yield center
    edges = np.roll(zone.corners, -1, axis=0) - zone.corners
    shortest_edge = float(np.min(np.linalg.norm(edges, axis=1)))
    # Keep the measured cube footprint inside the card, with a few mm extra
    # margin for the executed TCP to differ from the planned TCP.
    inset = 1 - 2 * (margin + 0.003) / shortest_edge if shortest_edge > 0 else 0
    if inset <= 0:
        return
    corners = center + (zone.corners - center) * inset
    edges = (corners + np.roll(corners, -1, axis=0)) / 2
    for point in sorted([*corners, *edges], key=np.linalg.norm):
        if zone.contains(point, margin):
            yield point


class L1SensorController:
    process = None

    def __init__(
        self,
        robot,
        capture,
        config,
        motion_config,
        calibration,
        plane_config,
        output,
        callback,
        *,
        process=None,
    ):
        self.robot, self.capture, self.cfg = robot, capture, config
        self.motion_cfg, self.callback = motion_config, callback
        self.process = process
        self.scene = SensorScene(
            calibration["hand_eye"]["T_end_camera"],
            plane_config,
            target_edge_m=config.get("target_edge_m", 0.30),
        )
        self.robot.camera = calibration
        self.output = Path(output)
        self.frames, self.events, self.completed = 0, [], []
        self.zone_views = {}
        # The retreat and return to an observation view give the object time
        # to settle. Stability is measured by the subsequent RGB-D sequence.
        self.motion = SensorPickPlaceSkill(
            robot, 0.0, callback, {"post_retreat_wait_s": 0.0, **motion_config}
        )

    def _record(self, event):
        self.events.append(event)
        if self.process is not None:
            self.process.event(event)

    def observe(self):
        obs = self.capture()
        if abs(self.robot.data.time - obs.timestamp_s) > self.cfg["max_frame_age_s"]:
            raise ValueError("Stale camera observation")
        obs.T_base_end = self.robot.end_pose(obs.joint_position_rad)
        blocks, zones = self.scene.observe(obs)
        self.motion.table = self.scene.table
        self.frames += 1
        for color in zones:
            pose = (
                self.cfg["observation_q_rad"][0]
                if self.cfg.get("fixed_camera")
                else self.robot.measured()[:6]
            )
            self.zone_views[color] = np.asarray(pose).copy()
        folder = self.output / "sensors"
        folder.mkdir(exist_ok=True)
        np.savez_compressed(
            folder / f"{self.frames:04d}.npz",
            rgb=obs.rgb,
            depth_m=obs.depth_m,
            T_base_end=obs.T_base_end,
            joint_position_rad=obs.joint_position_rad,
            timestamp_s=obs.timestamp_s,
            gripper_width_m=obs.gripper_width_m,
            intrinsics=np.array([obs.intrinsics[k] for k in ("fx", "fy", "cx", "cy")]),
        )
        cv2.imwrite(
            str(folder / f"{self.frames:04d}.jpg"), cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2BGR)
        )
        (folder / f"{self.frames:04d}.json").write_text(
            json.dumps(
                {
                    "timestamp_s": obs.timestamp_s,
                    "table_m": self.scene.table,
                    "blocks": [{"color": c, "center_m": t.center.tolist()} for c, t in blocks],
                    "zones": {c: z.as_dict() for c, z in zones.items()},
                }
            )
        )
        if self.process is not None:
            self.process.observe(
                f"sensors/{self.frames:04d}.jpg", obs.timestamp_s, blocks, observation=obs
            )
        return obs, blocks, zones

    def scan(self):
        self.motion.stage = "scan"
        required = set(self.cfg["l1_mapping"].values())
        if self.cfg.get("fixed_camera"):
            self.observe()
            if self.zone_views.keys() >= required:
                return
            raise ValueError(
                "Fixed camera must see the complete goal regions from the observation pose"
            )
        partial_views = {}
        for q in self.cfg["observation_q_rad"]:
            self.motion.move(np.asarray(q), 0.07)
            self.motion.hold(0.25)
            self.observe()
            if self.zone_views.keys() >= required:
                return
            for color in self.scene.partial_zones.keys() & (required - self.zone_views.keys()):
                support = len(self.scene.partial_zones[color])
                if color not in partial_views or support > partial_views[color][0]:
                    partial_views[color] = (support, self.robot.measured()[:6].copy())

        # Use a fresh observation at the strongest partial view only after the
        # preset sweep has failed to expose the full boundary.
        for color, (_, q) in partial_views.items():
            if color not in self.zone_views:
                self.motion.move(q, 0.07)
                self.motion.hold(0.25)
                obs, _, _ = self.observe()
                if color in self.zone_views or color not in self.scene.partial_zones:
                    continue
                points = self.scene.partial_zones[color]
                try:
                    qnew, detail = recenter_camera(
                        self.robot,
                        obs,
                        points,
                        self.scene.table,
                        {
                            "backoff_m": 0.13,
                            "max_joint_step_rad": 0.55,
                            "min_tcp_clearance_m": 0.10,
                        },
                    )
                    self.events.append({"event": "reobserve_partial_zone", "detail": detail})
                    self.motion.move(qnew, 0.07)
                    self.motion.hold(0.25)
                    self.observe()
                except ValueError as exc:
                    self.events.append({"event": "view_rejected", "reason": str(exc)})
            if self.zone_views.keys() >= required:
                return
        if self.zone_views.keys() >= required:
            return
        raise ValueError(f"Incomplete visible goal boundaries: {sorted(self.zone_views)}")

    def target(self, color, search=False):
        _, blocks, _ = self.observe()
        matching = [t for c, t in blocks if c == color]
        if not matching and search:
            for q in self.zone_views.values():
                self.motion.move(q, 0.07)
                self.motion.hold(0.25)
                _, blocks, _ = self.observe()
                matching = [t for c, t in blocks if c == color]
                if matching:
                    break
        if len(matching) != 1:
            raise ValueError(f"Expected a unique visible {color} cube; got {len(matching)}")
        return matching[0]

    def verify_places(self, color, other_colors=()):
        mapping = self.cfg["l1_mapping"]
        goal = mapping[color]
        self.motion.stage = "reobserve_place"
        self.motion.move(self.zone_views[goal], 0.07)
        self.motion.hold(0.3)
        checkers = {
            item: PlacementCheck(self.scene.zones[mapping[item]]) for item in (color, *other_colors)
        }
        accepted = set()
        self.motion.stage = "verify"
        for _ in range(9):
            _, blocks, _ = self.observe()
            for item, checker in checkers.items():
                choices = [t for c, t in blocks if c == item]
                # Ambiguous identity is not accepted or assigned by proximity.
                target = choices[0] if len(choices) == 1 else None
                done = checker.update(target, self.robot.data.time, self.robot.measured()[6])
                self._record(
                    {
                        "event": "placement_observation",
                        "color": item,
                        "time_s": self.robot.data.time,
                        "visible": len(choices),
                        "accepted": done,
                        "stable_duration_s": (
                            float(checker.last - checker.start)
                            if checker.start is not None
                            else None
                        ),
                    }
                )
                if done:
                    accepted.add(item)
                else:
                    accepted.discard(item)
            if color in accepted:
                return accepted
            self.motion.hold(self.cfg["verification_interval_s"])
        return accepted

    def run(self):
        failure = None
        final_checked = set()
        try:
            self.scan()
            for color, destination in self.cfg["l1_mapping"].items():
                for attempt in range(1, self.cfg["max_attempts"] + 1):
                    self.callback("observe", color=color, attempt=attempt)
                    self.motion.stage = "observe"
                    self.motion.move(np.array(self.cfg["observation_q_rad"][0]), 0.07)
                    self.motion.hold(0.25)
                    target = self.target(color, search=attempt > 1)
                    self.motion.hold(0.25)
                    fresh = self.target(color)
                    target, evidence = match_fresh_target(
                        target, [fresh], self.motion_cfg["grasp_feedback"]
                    )
                    zone = self.scene.zones[destination]
                    self.motion.stage = "planning"
                    self.callback("planning")
                    plan = None
                    last_error = None
                    for xy in placement_candidates(zone):
                        if not zone.contains(xy, 0.045):
                            continue
                        try:
                            plan = self.motion.plan(target, xy, zone.floor_m)
                            break
                        except RuntimeError as exc:
                            last_error = str(exc)
                            self._record(
                                {
                                    "event": "placement_candidate_rejected",
                                    "color": color,
                                    "position_m": xy.tolist(),
                                    "reason": last_error,
                                }
                            )
                    if plan is None:
                        raise RuntimeError(
                            f"No IK solution inside sensed {destination} goal bounds "
                            f"({last_error or 'no placement point with enough edge clearance'})"
                        )
                    self._record(
                        {
                            "event": "pregrasp",
                            "color": color,
                            "attempt": attempt,
                            "evidence": evidence,
                            "goal": zone.as_dict(),
                            "planned_tcp_m": plan["place"].tolist(),
                        }
                    )
                    if self.process is not None:
                        self.process.plan(
                            color, plan["center"], plan["place"], self.process.frame_time
                        )
                    try:
                        grip = self.motion.pick(plan)
                        self._record(
                            dict(
                                event="holding_estimate",
                                color=color,
                                **grip,
                                official_lift_verified=False,
                                limitation="Wrist camera blind near gripper; width retention is indirect evidence",
                            )
                        )
                        self.motion.place(plan)
                        last_placement = len(self.completed) + 1 == len(self.cfg["l1_mapping"])
                        accepted = self.verify_places(
                            color, self.completed if last_placement else ()
                        )
                        if color in accepted:
                            self.completed.append(color)
                            if last_placement:
                                final_checked = accepted
                            break
                        raise ValueError(
                            "Placement not verified from consecutive RGB-D observations"
                        )
                    except GraspFeedbackError as exc:
                        self._record({"event": "grasp_failure", "color": color, "reason": str(exc)})
                        if not exc.retryable or attempt == self.cfg["max_attempts"]:
                            raise
                        self.motion.recover_empty()
                    except ValueError as exc:
                        self._record(
                            {"event": "placement_failure", "color": color, "reason": str(exc)}
                        )
                        if attempt == self.cfg["max_attempts"]:
                            raise
            # The last placement is already after all manipulation. Only
            # revisit colours not confirmed alongside it in the same frames.
            remaining = [color for color in self.completed if color not in final_checked]
            while remaining:
                color, *others = remaining
                accepted = self.verify_places(color, others)
                if color not in accepted:
                    self.completed.remove(color)
                    self._record({"event": "completion_revoked", "color": color})
                    raise ValueError(f"Final recheck failed: {color}")
                remaining = [item for item in others if item not in accepted]
            self.motion.stage = "done"
            self.motion.hold(0.2)
        except (RuntimeError, ValueError, TimeoutError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            self.callback("failed")
        return {
            "completed_colors": self.completed,
            "placement_success": len(self.completed) == 3 and failure is None,
            "failure": failure,
            "official_score": None,
            "official_lift_verified": False,
            "sensor_time_s": self.robot.data.time,
            "events": self.events,
            "source": "RGB-D + calibrated FK + encoder/width feedback; no oracle",
            "limitation": "Indirect grip estimate; no optical lift proof; sim camera/calibration; no real robot validation",
        }
