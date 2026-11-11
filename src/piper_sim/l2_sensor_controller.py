"""L2 RGB-model masks + depth geometry + robot feedback; no hidden object state."""

import json
import time
from dataclasses import replace
from itertools import pairwise, permutations, zip_longest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from piper_sim.active_view import recenter_camera
from piper_sim.competition_controller import L1SensorController
from piper_sim.competition_segmentation import (
    CompetitionSegmenter,
    geometry_targets,
)
from piper_sim.competition_sensor import PlacementCheck, backproject, components
from piper_sim.field_motion import TableClearanceError
from piper_sim.grasp_feedback import GraspFeedbackError, match_fresh_target
from piper_sim.l2_progress import L2Progress


def footprint_gap(first, second):
    if cv2.intersectConvexConvex(first, second)[0] > 1e-8:
        return 0.0
    return min(
        -cv2.pointPolygonTest(a, tuple(map(float, p)), True)
        for a, b in ((first, second), (second, first))
        for p in b
    )


class L2SensorController(L1SensorController):
    def __init__(self, *args, method="sam2", **kwargs):
        super().__init__(*args, **kwargs)
        self.segmenter = CompetitionSegmenter(
            method,
            classify_fruits=self.cfg.get("l2_classify_fruits", False),
            category_backend=self.cfg.get("l2_category_backend", "rules"),
        )
        self.placements = []
        self.inference_s = 0.0
        self.progress = L2Progress(
            6,
            self.cfg.get("l2_max_attempts_per_object", 3),
            color_mapping=self.cfg["l2_color_mapping"],
        )
        self.view_requests = []
        self.verified_zone_views = {}
        self.last_verification = {}
        self.last_seen_target = None
        self.motion.cfg["prefer_independent_orientation"] = True
        self.motion.cfg["encoder_bias_correction"] = True
        # Stability is measured after returning to the observation pose. An
        # extra blind wait before that camera move contributes no evidence.
        self.motion.cfg["post_retreat_wait_s"] = 0.0

    def visible_geometry(self, *, split_depth=False):
        obs, _, _ = self.observe()
        return obs, geometry_targets(
            obs,
            self.scene.calibration,
            self.scene.table,
            split_depth=split_depth,
            compute_grasps=False,
        )

    def detect(self, obs, *, split_depth=False, focus=()):
        targets, report = self.segmenter.detect(
            obs,
            self.scene.calibration,
            self.scene.table,
            self.scene.zones,
            self.output / f"segmentation_{self.frames:04d}",
            include_zones=True,
            split_depth=split_depth,
            focus=focus,
        )
        self.inference_s += report["inference_wall_s"]
        self.events.append({"event": "segmentation", **report})
        if self.process is not None:
            self.process.fruits(targets, allow_new=self.motion.stage == "observe", observation=obs)
        self.view_requests = report.get("view_requests", [])
        # The real transport clock has already advanced during the inference.
        # Physics-only simulation needs to advance its virtual clock explicitly.
        if self.cfg.get("wall_clock"):
            self.robot.sync()
        else:
            self.motion.hold(report["inference_wall_s"])
        return targets

    def find_target(self, *, extended_scan=False):
        self.motion.stage = "observe"
        failed_plans = []
        views = [np.asarray(q) for q in self.cfg["l2_observation_q_rad"]]
        if self.cfg.get("l2_precise_packing"):
            recovery = [
                self.zone_views[item["destination"]]
                for item in self.progress.items
                if item["status"] == "retry" and item["destination"] in self.zone_views
            ]
            views += [q for q in recovery if all(np.linalg.norm(q - old) > 0.12 for old in views)]
        if extended_scan and not self.cfg.get("fixed_camera"):
            for q in self.cfg["observation_q_rad"]:
                q = np.asarray(q)
                if all(np.linalg.norm(q - old) > 0.12 for old in views):
                    views.append(q)
        for view_index, q in enumerate(views):
            if getattr(self.robot, "path_guard", None) is not None:
                try:
                    self.motion.validate_segment(self.robot.measured()[:6], q)
                except TableClearanceError as exc:
                    self.events.append({"event": "search_view_rejected", "reason": str(exc)})
                    continue
            self.events.append(
                {
                    "event": "search_view",
                    "q_rad": q.tolist(),
                    "extended_scan": extended_scan,
                    "time_s": float(self.robot.data.time),
                }
            )
            self.motion.move(np.array(q), 0.07)
            self.motion.hold(0.3)
            obs, _, _ = self.observe()
            proposals = self.detect(obs, split_depth=extended_scan)
            depth_view_requests = self.view_requests.copy()
            current = obs
            if proposals:
                current, fresh = self.visible_geometry(split_depth=extended_scan)
            destination_order = {}
            for proposal in proposals:
                destination = self.cfg["l2_color_mapping"].get(proposal.color)
                destination_order.setdefault(destination, len(destination_order))

            def priority(proposal, group_order=destination_order):
                try:
                    item = self.progress.associate(proposal.color, proposal.target)
                except ValueError:
                    return (99, 0, 0)
                retry = int(item is not None and item["status"] == "retry")
                if not self.cfg.get("l2_precise_packing"):
                    return (retry, 0, 0)
                destination = self.cfg["l2_color_mapping"].get(proposal.color)
                extent = np.linalg.norm(np.ptp(proposal.target.points[:, :2], axis=0))
                return (retry, group_order[destination], -extent)

            for proposal in sorted(proposals, key=priority):
                if proposal.color not in self.cfg["l2_color_mapping"]:
                    self.events.append({"event": "unknown_color", "id": proposal.key})
                    continue
                try:
                    item = self.progress.associate(proposal.color, proposal.target)
                    if item and (
                        item["status"] == "verified"
                        or item["attempts"] >= self.progress.max_attempts
                    ):
                        continue
                    if item is None and len(self.progress.items) >= self.progress.total:
                        continue
                    current_geometry, evidence = match_fresh_target(
                        proposal.target, fresh, self.motion_cfg["grasp_feedback"]
                    )
                    # Depth foreground confirms the pose is still current. Keep
                    # the instance mask's footprint and local grasp directions.
                    target = proposal.target
                except (GraspFeedbackError, ValueError):
                    continue
                repeated = any(
                    proposal.color == color
                    and np.linalg.norm(target.center - center) < 0.012
                    and abs(np.sin(target.yaw - yaw)) < 0.20
                    for color, center, yaw in failed_plans
                )
                planning_started = time.monotonic()
                try:
                    plan = self.make_plan(
                        target,
                        self.cfg["l2_color_mapping"][proposal.color],
                        ignore_id=None if item is None else item["id"],
                        category=proposal.color,
                        **({"budget_s": 3.0} if repeated else {}),
                    )
                except ValueError as exc:
                    failed_plans.append((proposal.color, target.center.copy(), target.yaw))
                    self.events.append(
                        {
                            "event": "unreachable_candidate",
                            "id": proposal.key,
                            "reason": str(exc),
                            "planning_wall_s": time.monotonic() - planning_started,
                            "repeated_geometry": repeated,
                            "span_m": np.ptp(target.points, axis=0).tolist(),
                            "diagnostics": getattr(self, "last_plan_diagnostics", {}),
                        }
                    )
                    continue
                planning_wall_s = time.monotonic() - planning_started
                # IK computation is blocking. Reacquire after it, so execution
                # cannot use a position that moved while the planner was busy.
                self.motion.hold(0.05)
                current, executable = self.visible_geometry(split_depth=extended_scan)
                try:
                    checked, new_evidence = match_fresh_target(
                        current_geometry, executable, self.motion_cfg["grasp_feedback"]
                    )
                    if np.linalg.norm(checked.center - current_geometry.center) > 0.003:
                        raise ValueError("Target moved during planning; recompute grasp")
                except (GraspFeedbackError, ValueError) as exc:
                    self.events.append({"event": "stale_plan_rejected", "reason": str(exc)})
                    continue
                evidence["postplan_check"] = new_evidence
                self.events.append(
                    {
                        "event": "target_selected",
                        "id": proposal.key,
                        "color": proposal.color,
                        "planning_wall_s": planning_wall_s,
                        "center_m": target.center.tolist(),
                        "fresh_check": evidence,
                        "planned_grasp": plan.get("grasp_geometry"),
                    }
                )
                return proposal, target, plan
            # A clipped depth component is a request for another camera view,
            # never a valid grasp or a fabricated full instance mask.
            if not self.cfg.get("fixed_camera") and len(views) < self.cfg.get(
                "l2_max_views_per_scan", 7
            ):
                if not proposals:
                    current, _, _ = self.observe()
                for request in depth_view_requests[:1]:
                    try:
                        qnew, detail = recenter_camera(
                            self.robot,
                            current,
                            np.array(request["points_base_m"]),
                            self.scene.table,
                            {
                                "backoff_m": 0.10,
                                "max_joint_step_rad": 1.5,
                                "min_tcp_clearance_m": 0.10,
                                "allow_depth_backoff": True,
                            },
                        )
                        if all(np.linalg.norm(qnew - old) > 0.12 for old in views):
                            views.insert(view_index + 1, qnew)
                            self.events.append(
                                {"event": "reobserve_missing_depth", "detail": detail}
                            )
                    except ValueError as exc:
                        self.events.append({"event": "depth_view_rejected", "reason": str(exc)})
                cloud, valid = backproject(current, self.scene.calibration)
                fg = valid & (cloud[..., 2] > self.scene.table + 0.012)
                fg &= cloud[..., 2] < self.scene.table + 0.16
                fg &= (
                    (cloud[..., 0] > 0.12) & (cloud[..., 0] < 0.75) & (np.abs(cloud[..., 1]) < 0.5)
                )
                for patch in components(fg, 100):
                    v, u = np.where(patch)
                    if (
                        min(u.min(), v.min()) > 3
                        and u.max() < patch.shape[1] - 4
                        and v.max() < patch.shape[0] - 4
                    ):
                        continue
                    try:
                        qnew, detail = recenter_camera(
                            self.robot,
                            current,
                            cloud[patch],
                            self.scene.table,
                            {
                                "backoff_m": 0.08,
                                "max_joint_step_rad": 0.4,
                                "min_tcp_clearance_m": 0.10,
                            },
                        )
                        if all(np.linalg.norm(qnew - old) > 0.12 for old in views):
                            views.insert(view_index + 1, qnew)
                            self.events.append(
                                {"event": "reobserve_clipped_fruit", "detail": detail}
                            )
                    except ValueError as exc:
                        self.events.append({"event": "fruit_view_rejected", "reason": str(exc)})
                    break
        raise ValueError("No fresh, unambiguous, unsorted fruit with a mapped colour is visible")

    def placement_gap(self, hull, center, span, placed):
        gap = float("inf")
        for item in placed:
            other = item.get("hull")
            if self.cfg.get("l2_precise_packing") and other is not None:
                gap = min(gap, footprint_gap(hull, other))
                if gap < 0.014:
                    break
            elif np.all(np.abs(center - item["center"][:2]) < (span + item["span"]) / 2 + 0.008):
                return 0.0
        return gap

    def make_plan(
        self,
        target,
        destination,
        *,
        ignore_id=None,
        category=None,
        budget_s=None,
        prefer_near_release=False,
        reserve_lateral_strip=False,
    ):
        zone = self.scene.zones[destination]
        occupied = [
            p
            for p in self.placements
            if p["destination"] == destination and (ignore_id is None or p["id"] != ignore_id)
        ]
        self.last_plan_diagnostics = {
            "slot_rejections": 0,
            "occupied_rejections": 0,
            "options": 0,
            "ik_rejections": 0,
            "ik_examples": [],
        }
        diagnostics = self.last_plan_diagnostics
        diagnostics["rotation_rejections"] = 0
        # Placement offsets are generated within camera-measured bounds, never
        # obtained from scene files. Previously observed placed footprints reserve space.
        offsets = (0, -0.045, 0.045, -0.025, 0.025, -0.06, 0.06, -0.065, 0.065)
        if occupied and self.cfg.get("l2_precise_packing"):
            offsets += (-0.030, 0.030, -0.050, 0.050, -0.070, 0.070)
        choices = [zone.center + np.array([dx, dy]) for dx in offsets for dy in offsets]
        geometry = target.grasp_geometry or {}
        candidates = geometry.get("candidates", [])
        if not candidates:
            raise ValueError("No mask-derived local grasp satisfies opening and geometry checks")
        budget = self.cfg.get("l2_planning_budget_s", 12.0)
        deadline = time.monotonic() + (budget if budget_s is None else min(budget, budget_s))
        self.motion.cfg["planning_deadline_monotonic"] = deadline
        self.motion._pickup_cache = {}
        if not geometry.get("major_axis_reliable", True):
            radial_yaw = np.arctan2(target.center[1], target.center[0])
            candidates = sorted(candidates, key=lambda c: abs(np.sin(c["yaw_rad"] - radial_yaw)))
        angles = [0.0, np.pi / 2, -np.pi / 2]
        if geometry.get("major_axis_reliable"):
            axis = geometry["metric_major_axis_base"]
            object_yaw = np.arctan2(axis[1], axis[0])
            for edge in np.roll(zone.corners, -1, axis=0) - zone.corners:
                angle = (np.arctan2(edge[1], edge[0]) - object_yaw + np.pi / 2) % np.pi - np.pi / 2
                if all(abs(angle - old) > 1e-5 for old in angles):
                    angles.append(float(angle))
        if occupied and self.cfg.get("l2_precise_packing"):
            angles.extend(np.deg2rad([-30, -15, 15, 30, -75, 75]))
        free_slots = []
        precise = bool(occupied and self.cfg.get("l2_precise_packing"))
        boundary = zone.corners.astype(np.float32)
        shared = list(self.cfg.get("l2_color_mapping", {}).values()).count(destination) > 1
        first_long = (
            not occupied
            and self.cfg.get("l2_precise_packing")
            and shared
            and geometry.get("major_axis_reliable")
            and np.linalg.norm(np.ptp(target.points[:, :2], axis=0))
            > 0.5 * np.linalg.norm(boundary[1] - boundary[0])
        )
        if first_long:
            axes = np.diff(boundary[:3], axis=0)
            axes /= np.linalg.norm(axes, axis=1, keepdims=True)
            bounds = boundary @ axes.T
            choices = [xy for xy in choices if np.max(np.abs((xy - zone.center) @ axes.T)) <= 0.046]
        planning_margin = 0.023 if prefer_near_release else 0.020 if first_long else 0.015
        strip_axis = None
        if precise and shared and len(occupied) == 1 and occupied[0].get("hull") is not None:
            box = cv2.boxPoints(cv2.minAreaRect(occupied[0]["hull"]))
            edges = np.roll(box, -1, axis=0) - box
            lengths = np.linalg.norm(edges, axis=1)
            longest = int(np.argmax(lengths))
            if lengths[longest] > 0.10 and lengths[longest] > 1.6 * lengths.min():
                strip_axis = edges[longest] / lengths[longest]
        for angle in angles:
            rotation = Rotation.from_euler("z", angle).as_matrix()
            relative = (target.points - target.center) @ rotation.T
            span = np.ptp(relative[:, :2], axis=0) + 0.016
            footprint = cv2.convexHull(relative[:, :2].astype(np.float32)).reshape(-1, 2)
            for xy in choices:
                hull = np.asarray(footprint + xy, dtype=np.float32)
                border_gap = min(
                    cv2.pointPolygonTest(boundary, tuple(map(float, point)), True) for point in hull
                )
                if border_gap < planning_margin:
                    diagnostics["slot_rejections"] += 1
                    continue
                neighbor_gap = self.placement_gap(hull, xy, span, occupied)
                if neighbor_gap < 0.014:
                    diagnostics["occupied_rejections"] += 1
                    continue
                # Prefer clearance to both the boundary and nearby fruit.
                slack = border_gap + min(neighbor_gap, 0.05) if precise else border_gap
                if first_long:
                    projection = hull @ axes.T
                    # Reserve a contiguous strip for subsequent compact fruit.
                    empty_width = max(
                        np.max(projection.min(axis=0) - bounds.min(axis=0)),
                        np.max(bounds.max(axis=0) - projection.max(axis=0)),
                    )
                    slack = empty_width + 1.4 * min(border_gap, 0.030) - 0.05 * np.linalg.norm(xy)
                if strip_axis is not None:
                    # Leave the other end of the free strip for a later fruit.
                    slack += 1.3 * abs(float((xy - zone.center) @ strip_axis))
                free_slots.append((xy, angle, slack))
        if precise or first_long:
            free_slots.sort(key=lambda slot: (-slot[2], np.linalg.norm(slot[0] - zone.center)))
        park_for_banana = reserve_lateral_strip or (
            category == "potato"
            and self.cfg.get("l2_color_mapping", {}).get("banana") == destination
        )
        if (
            not occupied
            and not first_long
            and self.cfg.get("l2_precise_packing")
            and (shared or park_for_banana)
            and np.max(np.ptp(target.points[:, :2], axis=0)) < 0.085
        ):
            if park_for_banana:
                # Occupy one lateral end so the banana can use an entire
                # strip spanning the near and far parts of the measured zone.
                edges = np.diff(np.vstack((zone.corners, zone.corners[0])), axis=0)
                directions = edges / np.linalg.norm(edges, axis=1, keepdims=True)
                lateral = np.array([-zone.center[1], zone.center[0]])
                direction = directions[np.argmax(directions @ lateral)]
                preferred = zone.center + 0.045 * direction
            else:
                preferred = zone.center - 0.025 * np.sign(zone.center)
            free_slots.sort(key=lambda slot: np.linalg.norm(slot[0] - preferred))
        candidate_options = []
        for candidate in candidates:
            grasp = replace(
                target, center=np.array(candidate["position_base_m"]), yaw=candidate["yaw_rad"]
            )
            options = [(xy, grasp.yaw + angle) for xy, angle, _ in free_slots]
            diagnostics["options"] += len(options)
            if (
                precise or (prefer_near_release and not (park_for_banana and not occupied))
            ) and np.linalg.norm(zone.center) > 0.58:
                # Beyond the near part of a side zone, release hover often fails
                # before descent. Keep the checked packing slack as a tie-breaker.
                offset = grasp.center[:2] - target.center[:2]
                options = [
                    (xy, grasp.yaw + angle)
                    for xy, angle, _ in sorted(
                        free_slots,
                        key=lambda slot: max(0, np.linalg.norm(slot[0] + offset) - 0.58),
                    )
                ]
            elif occupied and not precise:
                # Preserve the established candidate order when the original
                # orientations fit. When only edge-alignment makes a narrow
                # slot possible, centre the object along that slot rather than
                # also pushing it towards the nearest table-side boundary.
                original_orientation_fits = any(
                    abs(np.sin(2 * (yaw - grasp.yaw))) < 1e-5 for _, yaw in options
                )
                if original_orientation_fits:
                    options.sort(key=lambda option: np.linalg.norm(option[0]))
                else:
                    options.sort(
                        key=lambda option: (
                            np.linalg.norm(option[0] - zone.center),
                            np.linalg.norm(option[0]),
                        )
                    )
            candidate_options.append([(candidate, grasp, xy, yaw) for xy, yaw in options])
        # Interleave grasp candidates: an unreachable pickup must not monopolize
        # the whole budget by being retried at every destination slot.
        for group in zip_longest(*candidate_options):
            for option in group:
                if option is None:
                    continue
                candidate, grasp, xy, place_yaw = option
                self.motion.cfg["grasp_yaw_candidates_rad"] = [grasp.yaw + np.pi, grasp.yaw]
                if time.monotonic() >= deadline:
                    raise ValueError(
                        "Bounded IK search exhausted; try another observed target/view"
                    )
                self.motion.cfg["place_yaw_rad"] = place_yaw
                grasp_offset = grasp.center - target.center
                if precise and geometry.get("major_axis_reliable"):
                    # Rotate an off-centre grasp with the packed placement
                    # before asking IK to reach the release point.
                    grasp_offset = Rotation.from_euler("z", place_yaw - grasp.yaw).apply(
                        grasp_offset
                    )
                tcp_xy = xy + grasp_offset[:2]
                try:
                    for correction in range(2):
                        plan = self.motion.plan(grasp, tcp_xy, zone.floor_m)
                        pick_rotation = (
                            Rotation.from_euler("z", plan["yaw"]).as_matrix()
                            @ Rotation.from_euler("y", plan["tilt"]).as_matrix()
                            @ np.diag([1.0, -1.0, -1.0])
                        )
                        drop_rotation = (
                            Rotation.from_euler("z", plan.get("place_yaw", plan["yaw"])).as_matrix()
                            @ Rotation.from_euler(
                                "y", plan.get("place_tilt", plan["tilt"])
                            ).as_matrix()
                            @ np.diag([1.0, -1.0, -1.0])
                        )
                        delta = drop_rotation @ pick_rotation.T
                        predicted_center = delta @ (target.center - plan["center"]) + plan["place"]
                        error = xy - predicted_center[:2]
                        if np.linalg.norm(error) <= 0.003:
                            break
                        tcp_xy = plan["place"][:2] + error
                    if np.linalg.norm(error) > 0.003:
                        continue
                except RuntimeError as exc:
                    diagnostics["ik_rejections"] += 1
                    if len(diagnostics["ik_examples"]) < 3:
                        diagnostics["ik_examples"].append(str(exc))
                    continue
                predicted = (target.points - plan["center"]) @ delta.T + plan["place"]
                transport_rotation = Rotation.from_matrix(delta).magnitude()
                if transport_rotation > self.cfg.get("l2_max_transport_rotation_rad", np.pi / 2):
                    diagnostics["rotation_rejections"] += 1
                    continue
                hull = cv2.convexHull(predicted[:, :2].astype(np.float32)).reshape(-1, 2)
                span = np.ptp(predicted[:, :2], axis=0) + 0.016
                if not zone.contains(hull, planning_margin):
                    continue
                if self.placement_gap(hull, predicted_center[:2], span, occupied) < 0.014:
                    continue
                approach = plan["center"] - plan["hover"]
                approach /= np.linalg.norm(approach)
                plan["object_destination_estimate_m"] = predicted_center
                plan["grasp_geometry"] = {
                    **candidate,
                    "commanded_tcp_base_m": plan["center"].tolist(),
                    "commanded_release_tcp_base_m": plan["place"].tolist(),
                    "actual_closing_axis_base": pick_rotation[:, 1].tolist(),
                    "actual_tool_z_axis_base": pick_rotation[:, 2].tolist(),
                    "actual_approach_direction_base": approach.tolist(),
                    "commanded_opening_m": 0.07,
                    "transport_rotation_rad": float(transport_rotation),
                    "object_destination_estimate_m": predicted_center.tolist(),
                    "placement_assumption": "rigid retention without slip; independently rechecked after release",
                }
                return plan
        raise ValueError(
            "No reachable free placement/grasp with observed geometry and 70mm gripper"
        )

    def verify_fruit(self, color, destination, expected, *, item=None):
        self.motion.stage = "reobserve_place"
        self.motion.move(self.zone_views[destination], 0.07)
        self.motion.hold(0.3)
        # A new learned mask establishes the post-release instance. Subsequent
        # fixed-view samples use bounded geometry association, not hidden labels.
        self.last_seen_target = None
        focus = [(expected, 0.15)]
        focus.extend(
            (p["center"], 0.20) for p in self.placements if p["destination"] == destination
        )
        for split_depth in (False, True):
            obs, _, _ = self.observe()
            proposals = self.detect(obs, split_depth=split_depth, focus=focus)
            possible = []
            for proposal in proposals:
                target = proposal.target
                if self.cfg["l2_color_mapping"].get(proposal.color) != destination or (
                    np.linalg.norm(target.center[:2] - expected[:2]) >= 0.12
                ):
                    continue
                if item is not None:
                    try:
                        if self.progress.associate(proposal.color, target) is not item:
                            continue
                    except ValueError:
                        continue
                elif proposal.color != color:
                    continue
                possible.append(target)
            distances = sorted(
                (float(np.linalg.norm(t.center[:2] - expected[:2])), i)
                for i, t in enumerate(possible)
            )
            if distances and (len(distances) == 1 or distances[1][0] - distances[0][0] >= 0.025):
                reference = possible[distances[0][1]]
                break
        else:
            self.last_verification = {
                "reason": "missing_or_ambiguous_postrelease_mask",
                "matches": len(possible),
            }
            return None
        self.refresh_neighbors(destination, proposals)
        check = PlacementCheck(self.scene.zones[destination], cube=False)
        self.motion.stage = "verify"
        for _ in range(9):
            obs, targets = self.visible_geometry(split_depth=split_depth)
            try:
                target, _ = match_fresh_target(
                    reference, targets, self.motion_cfg["grasp_feedback"]
                )
            except GraspFeedbackError:
                target = None
            accepted = check.update(target, obs.timestamp_s, obs.gripper_width_m)
            inside = bool(
                target is not None
                and self.scene.zones[destination].contains(target.points[:, :2], 0.012)
            )
            self.last_verification = {
                "event": "fruit_placement_observation",
                "time_s": obs.timestamp_s,
                "visible_matches": int(target is not None),
                "inside": inside,
                "center_m": None if target is None else target.center.tolist(),
                "accepted": accepted,
                "reason": "verified"
                if accepted
                else (
                    "not_fully_inside"
                    if target is not None and not inside
                    else "awaiting_stable_observation"
                ),
            }
            self.events.append(self.last_verification.copy())
            if target is not None:
                self.last_seen_target = target
                reference = target
            if accepted:
                self.verified_zone_views[destination] = obs.joint_position_rad.copy()
                return target
            if target is not None and not inside:
                return None  # new observation supplies a correction grasp; never count it
            self.motion.hold(0.4)
        return None

    def save_progress(self):
        self.completed = self.progress.completed
        if self.process is not None:
            self.process.sync_fruits(self.progress.items)
        (self.output / "progress.json").write_text(
            json.dumps(
                {
                    "sensor_time_s": float(self.robot.data.time),
                    "completed": self.completed,
                    "objects": self.progress.items,
                    "official_lift_verified": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        with (self.output / "events.jsonl").open("w") as f:
            for event in self.events:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def final_zone_view(self, destination):
        if destination in self.verified_zone_views:
            return self.verified_zone_views[destination]
        return self.zone_views[destination]

    def final_zone_order(self):
        destinations = tuple(dict.fromkeys(item["destination"] for item in self.progress.items))
        if len(destinations) < 2:
            return destinations
        current = self.robot.measured()[:6]
        speeds = np.asarray(self.robot.interface["control"]["max_joint_speed_rad_s"])

        def travel_time(order):
            poses = [current, *(self.final_zone_view(destination) for destination in order)]
            return sum(np.max(np.abs(end - start) / speeds) for start, end in pairwise(poses))

        return min(permutations(destinations), key=travel_time)

    def verify_final_group(self, destination, items):
        """Recheck instances in one zone together with independent stability timers."""
        self.motion.stage = "reobserve_place"
        self.motion.move(self.final_zone_view(destination), 0.07)
        self.motion.hold(0.3)
        split_depth = False
        initial = None
        stored = {
            p["id"]: p["verified_geometry"]
            for p in self.placements
            if p["destination"] == destination and "verified_geometry" in p
        }
        if all(item["id"] in stored for item in items):
            obs, targets = self.visible_geometry()
            references = {}
            cfg = self.motion_cfg["grasp_feedback"]
            try:
                for item in items:
                    target, _ = match_fresh_target(
                        stored[item["id"]],
                        targets,
                        {
                            **cfg,
                            "max_target_correction_m": min(
                                0.008, cfg.get("max_target_correction_m", 0.015)
                            ),
                        },
                    )
                    references[item["id"]] = [target]
                if len({id(ref[0]) for ref in references.values()}) == len(items):
                    initial = obs, targets
            except GraspFeedbackError:
                pass
        self.events.append(
            {
                "event": "final_group_identity",
                "destination": destination,
                "source": "fresh_depth_and_verified_geometry" if initial else "learned_masks",
            }
        )
        if initial is None:
            focus = [(item["center"], self.progress.radius + 0.03) for item in items]
            for split_depth in (False, True):
                obs, _, _ = self.observe()
                proposals = self.detect(obs, split_depth=split_depth, focus=focus)
                references = {}
                for proposal in proposals:
                    try:
                        item = self.progress.associate(proposal.color, proposal.target)
                    except ValueError:
                        continue
                    if item is not None and any(item is member for member in items):
                        references.setdefault(item["id"], []).append(proposal.target)
                if all(len(references.get(item["id"], [])) == 1 for item in items):
                    break
        checks = {
            item["id"]: PlacementCheck(self.scene.zones[destination], cube=False) for item in items
        }
        verified = {}
        self.motion.stage = "verify"
        for _ in range(9):
            if initial is None:
                current, targets = self.visible_geometry(split_depth=split_depth)
            else:
                current, targets = initial
                initial = None
            assignments = {}
            for item in items:
                choices = references.get(item["id"], [])
                if len(choices) == 1:
                    try:
                        assignments[item["id"]], _ = match_fresh_target(
                            choices[0], targets, self.motion_cfg["grasp_feedback"]
                        )
                    except GraspFeedbackError:
                        pass
            verified = {}
            for item in items:
                target = assignments.get(item["id"])
                if target is not None and sum(t is target for t in assignments.values()) != 1:
                    target = None
                accepted = checks[item["id"]].update(
                    target, current.timestamp_s, current.gripper_width_m
                )
                self.events.append(
                    {
                        "event": "final_group_observation",
                        "id": item["id"],
                        "time_s": current.timestamp_s,
                        "accepted": accepted,
                    }
                )
                if accepted:
                    verified[item["id"]] = target
                if target is not None:
                    references[item["id"]] = [target]
            if len(verified) == len(items):
                break
            self.motion.hold(0.4)
        return verified

    def remember_placement(self, item, target):
        self.progress.verify(item, target, self.robot.data.time)
        self.reserve_observed_footprint(item, target)
        self.placements[-1]["verified_geometry"] = target
        self.events.append(
            {
                "event": "object_verified",
                "id": item["id"],
                "attempts": item["attempts"],
                "time_s": float(self.robot.data.time),
            }
        )
        self.save_progress()

    def refresh_neighbors(self, destination, proposals):
        """Reopen a previously verified step if a later placement displaces it."""
        zone = self.scene.zones[destination]
        observed = []
        for proposal in proposals:
            try:
                observed.append(
                    (proposal, self.progress.associate(proposal.color, proposal.target))
                )
            except ValueError:
                continue
        placed = {p["id"]: p for p in self.placements if p["destination"] == destination}
        for item in self.progress.items:
            if item["id"] not in placed or item["status"] != "verified":
                continue
            matches = [p.target for p, binding in observed if binding is item]
            if not matches:
                before = placed[item["id"]]
                for proposal, binding in observed:
                    target = proposal.target
                    if binding is not None or zone.contains(target.center[:2], -0.005):
                        continue
                    if np.linalg.norm(target.center[:2] - before["center"][:2]) > 0.18:
                        continue
                    if abs(target.height - before["height_m"]) > 0.018:
                        continue
                    compatible = [
                        other
                        for other in self.progress.items
                        if self.progress.compatible(other, proposal.color, target)
                    ]
                    if len(compatible) == 1 and compatible[0] is item:
                        matches.append(target)
            if len(matches) != 1:
                continue
            target = matches[0]
            if not zone.contains(target.center[:2], -0.005):
                self.progress.fail(item, "neighbor_displaced_outside_zone", target)
                self._record(
                    {
                        "event": "neighbor_displaced",
                        "id": item["id"],
                        "center_m": target.center.tolist(),
                        "time_s": float(self.robot.data.time),
                    }
                )
            self.reserve_observed_footprint(item, target)

    def reserve_observed_footprint(self, item, target):
        """A failed placement still occupies space; reserving it earns no credit."""
        self.placements = [p for p in self.placements if p["id"] != item["id"]]
        self.placements.append(
            {
                "id": item["id"],
                "color": item["color"],
                "destination": item["destination"],
                "center": target.center.copy(),
                "height_m": float(target.height),
                "span": np.ptp(target.points[:, :2], axis=0) + 0.016,
                "hull": cv2.convexHull(target.points[:, :2].astype(np.float32)).reshape(-1, 2),
            }
        )

    def run(self):
        failure = None
        final_rechecked = False
        search_failures = 0
        try:
            self.scan()
            while True:
                if len(self.progress.completed) == 6:
                    final_rechecked = False
                    final_targets = {}
                    for destination in self.final_zone_order():
                        members = [
                            i for i in self.progress.items if i["destination"] == destination
                        ]
                        final_targets.update(self.verify_final_group(destination, members))
                    final_rechecked = all(
                        final_targets.get(item["id"]) is not None for item in self.progress.items
                    )
                    for item in list(self.progress.items):
                        target = final_targets.get(item["id"])
                        if target is None:
                            final_rechecked = False
                            self.progress.fail(item, "final_recheck_failed")
                            self.placements = [p for p in self.placements if p["id"] != item["id"]]
                            self._record({"event": "completion_revoked", "id": item["id"]})
                        else:
                            self.remember_placement(item, target)
                    self.save_progress()
                    if final_rechecked:
                        break
                try:
                    proposal, target, plan = (
                        self.find_target(extended_scan=True)
                        if search_failures
                        else self.find_target()
                    )
                    search_failures = 0
                except ValueError as exc:
                    search_failures += 1
                    self.events.append(
                        {"event": "search_incomplete", "reason": str(exc), "pass": search_failures}
                    )
                    self.save_progress()
                    if search_failures >= self.cfg.get("l2_max_search_passes", 2):
                        raise ValueError(
                            "No executable unresolved target after bounded fresh scans"
                        ) from exc
                    continue
                destination = self.cfg["l2_color_mapping"][proposal.color]
                item = self.progress.start(proposal.color, destination, target)
                if self.process is not None:
                    self.process.bind_fruit(item, proposal)
                    self.process.plan(
                        proposal.color,
                        plan["center"],
                        plan["place"],
                        target.grasp_geometry.get("timestamp_s"),
                    )
                self.callback("planning", color=destination, attempt=item["attempts"])
                self.events.append(
                    {
                        "event": "attempt_started",
                        "id": item["id"],
                        "attempt": item["attempts"],
                        "source_frame": proposal.key,
                    }
                )
                self.save_progress()
                try:
                    grip = self.motion.pick_closed_loop(plan)
                    self._record(
                        {
                            "event": "holding_estimate",
                            "id": item["id"],
                            **grip,
                            "official_lift_verified": False,
                        }
                    )
                    expected = plan.get("object_destination_estimate_m", plan["place"])
                    # Bind both possible source and destination before transport. A slip
                    # must not silently become a brand new object or a completed placement.
                    self.progress.expect_release(item, expected)
                    if self.process is not None:
                        self.process.sync_fruits(self.progress.items)
                    self.motion.place_closed_loop(plan)
                    verified = self.verify_fruit(proposal.color, destination, expected, item=item)
                    if verified is None:
                        self.progress.fail(item, self.last_verification, self.last_seen_target)
                        if self.last_seen_target is not None:
                            self.reserve_observed_footprint(item, self.last_seen_target)
                        self._record(
                            {
                                "event": "placement_retry",
                                "id": item["id"],
                                "evidence": self.last_verification.copy(),
                            }
                        )
                        self.save_progress()
                        continue
                    self.remember_placement(item, verified)
                except GraspFeedbackError as exc:
                    self.progress.fail(item, exc.code)
                    self._record(
                        {
                            "event": "grasp_failure",
                            "id": item["id"],
                            "reason": str(exc),
                            "code": exc.code,
                            "retryable": exc.retryable,
                            "evidence": exc.evidence,
                        }
                    )
                    self.save_progress()
                    if not exc.retryable:
                        raise
                    self.motion.recover_empty()
                    # Next attempt always scans/resegments; no blind replay of old grasp.
            self.motion.stage = "done"
            self.motion.hold(0.2)
        except (RuntimeError, ValueError, TimeoutError, KeyboardInterrupt) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            self.motion.hold_current()
            self.callback("failed")
        finally:
            self.save_progress()
            self.segmenter.close()
        return {
            "task": "l2",
            "completed_instances": self.completed,
            "placement_success": len(self.completed) == 6 and final_rechecked and failure is None,
            "final_rechecked": final_rechecked,
            "objects": self.progress.items,
            "completed_colors": self.completed,
            "failure": failure,
            "sensor_time_s": float(self.robot.data.time),
            "model_inference_wall_s": self.inference_s,
            "events": self.events,
            "official_score": None,
            "official_lift_verified": False,
            "mapping_status": self.cfg["l2_mapping_status"],
            "source": "RGB model masks + registered depth + calibrated FK + gripper encoders",
            "limitation": "Bounded geometry identity association; width-based holding estimate; partial visible surface; no real robot validation",
        }
