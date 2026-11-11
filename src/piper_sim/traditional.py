"""Traditional vision -> numerical IK -> rate-limited position servo; physical contacts only."""

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from piper_sim.control import target_to_action
from piper_sim.vision import detect_blocks


class IK:
    def __init__(self, backend, tilt=0.6):
        self.tilt = tilt
        self.model = backend.model
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site("tcp").id
        self.lower = np.asarray(backend.interface["joints"]["lower_rad"])
        self.upper = np.asarray(backend.interface["joints"]["upper_rad"])

    def solve(self, position, yaw, seed):
        # Z towards the table; X in the chosen planar grasp direction.
        rotation = (
            Rotation.from_euler("z", yaw).as_matrix()
            @ Rotation.from_euler("y", self.tilt).as_matrix()
            @ np.diag([1.0, -1.0, -1.0])
        )

        def residual(q):
            self.data.qpos[:6] = q
            mujoco.mj_kinematics(self.model, self.data)
            p = self.data.site_xpos[self.site]
            r = self.data.site_xmat[self.site].reshape(3, 3)
            e = Rotation.from_matrix(rotation @ r.T).as_rotvec()
            return np.r_[p - position, 0.15 * e]

        seeds = [seed, [0, 1.4, -0.8, 0, 1.0, 0], [0, 1.8, -1.2, 0, 0.8, 0]]
        best = None
        for start in seeds:
            solution = least_squares(
                residual,
                np.clip(start, self.lower + 1e-5, self.upper - 1e-5),
                bounds=(self.lower + 1e-6, self.upper - 1e-6),
                max_nfev=120,
                ftol=1e-7,
                xtol=1e-7,
                gtol=1e-7,
            )
            error = residual(solution.x)
            if best is None or np.linalg.norm(error) < best[0]:
                best = (np.linalg.norm(error), solution.x, error)
            if np.linalg.norm(error[:3]) < 0.0015 and np.linalg.norm(error[3:]) < 0.005:
                return solution.x
        raise RuntimeError(
            f"IK unreachable: target={np.round(position, 3)}, error={np.round(best[2], 3)}"
        )


class TaskJudge:
    def __init__(self, backend, task, objects, mode):
        self.backend = backend
        self.task = task
        self.objects = objects
        self.mode = mode
        self.records = {
            o["name"]: {
                "color": o["color"],
                "lift_s": 0.0,
                "grasped": False,
                "stable_s": 0.0,
                "placed": False,
            }
            for o in objects
        }
        self.limit = task["judging"][mode + "_limit_s"]
        self.events = []

    def update(self):
        b = self.backend
        cfg = self.task["judging"]
        dt = 1 / b.interface["control"]["policy_hz"]
        half = self.task["scene"]["cube_size_m"] / 2
        for obj in self.objects:
            name = obj["name"]
            r = self.records[name]
            body = b.model.body(name).id
            p = b.data.xpos[body]
            rot = b.data.xmat[body].reshape(3, 3)
            extent = np.abs(rot) @ np.full(3, half)
            bottom = p[2] - extent[2]
            raised = bottom - self.task["scene"]["table_top_m"] >= cfg["lift_height_m"]
            r["lift_s"] = r["lift_s"] + dt if raised else 0.0
            if r["lift_s"] >= cfg["lift_hold_s"] and not r["grasped"]:
                r["grasped"] = True
                self.events.append(
                    {"t": float(b.data.time), "object": name, "event": "valid_grasp"}
                )
            zone = self.task["scene"]["bins"][obj["zone"]]
            inside = np.all(np.abs(p[:2] - zone["center_xy"]) + extent[:2] <= zone["half_size_xy"])
            dof = b.model.jnt_dofadr[b.model.joint(name + "_free").id]
            velocity = b.data.qvel[dof : dof + 6]
            stable = (
                np.linalg.norm(velocity[:3]) < cfg["stable_speed_m_s"]
                and np.linalg.norm(velocity[3:]) < cfg["stable_angular_speed_rad_s"]
            )
            released = b.measured()[6] > 0.05
            valid = (
                inside
                and stable
                and released
                and r["grasped"]
                and abs(bottom - self.task["scene"]["table_top_m"]) < 0.004
            )
            r["stable_s"] = r["stable_s"] + dt if valid else 0.0
            if r["stable_s"] >= cfg["stable_hold_s"] and not r["placed"]:
                r["placed"] = True
                self.events.append(
                    {"t": float(b.data.time), "object": name, "event": "correct_placement"}
                )

    def result(self):
        records = list(self.records.values())
        grasp = sum(r["grasped"] for r in records)
        placed = sum(r["placed"] for r in records)
        finished = all(r["placed"] for r in records) and self.backend.data.time <= self.limit
        return {
            "mode": self.mode,
            "simulated_time_s": float(self.backend.data.time),
            "limit_s": self.limit,
            "completed": finished,
            "valid_grasps": grasp,
            "correct_placements": placed,
            "total": len(records),
            "time_score_eligible": finished,
            "time_score": None,
            "note": "Time score formula and per-object point allocation unavailable; no invented official score.",
            "objects": self.records,
            "events": self.events,
        }


class TraditionalController:
    def __init__(self, backend, task, judge, on_step=None):
        self.b = backend
        self.task = task
        self.judge = judge
        self.ik = IK(backend, task["controller"]["grasp_tilt_rad"])
        self.on_step = on_step or (lambda stage: None)
        self.stage = "initialize"
        self.log = []
        self.placed_counts = {"l1": 0, "red": 0, "green": 0, "blue": 0}
        self.expected = 3 if judge.mode == "l1" else 6
        self.camera_renderer = mujoco.Renderer(
            backend.model,
            height=backend.camera["camera"]["height"],
            width=backend.camera["camera"]["width"],
        )

    def tick(self, action):
        if self.b.data.time >= self.judge.limit:
            raise TimeoutError("Task time limit reached")
        self.b.step(action)
        self.judge.update()
        self.on_step(self.stage)

    def move_joint(self, q, width, tolerance=0.008):
        goal = np.r_[q, width]
        start = self.b.data.time
        while True:
            self.tick(target_to_action(goal, self.b.previous_target, self.b.interface))
            if np.max(np.abs(self.b.measured()[:6] - q)) < tolerance and (
                self.b.previous_target[6] < 0.004
                if width == 0
                else abs(self.b.measured()[6] - width) < 0.004
            ):
                return
            if self.b.data.time - start > self.task["controller"]["max_stage_s"]:
                raise TimeoutError(
                    f"{self.stage}: servo did not reach target, error={self.b.measured() - goal}"
                )

    def hold(self, seconds):
        for _ in range(int(np.ceil(seconds * self.b.interface["control"]["policy_hz"]))):
            # Hold target, do not use a[6]=0 (which would half-open).
            self.tick(
                target_to_action(self.b.previous_target, self.b.previous_target, self.b.interface)
            )

    def cartesian(self, position, yaw, width):
        current = self.b.data.site_xpos[self.b.model.site("tcp").id].copy()
        count = max(1, int(np.ceil(np.linalg.norm(position - current) / 0.035)))
        for point in np.linspace(current, position, count + 1)[1:]:
            q = self.ik.solve(point, yaw, self.b.measured()[:6])
            self.move_joint(q, width, tolerance=0.009)

    def capture(self):
        from piper_sim.calibration import simulated_depth_outputs

        renderer = self.camera_renderer
        b = self.b
        cam = b.camera["camera"]
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = False
        renderer.update_scene(b.data, camera=cam["name"], scene_option=opt)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(b.data, camera=cam["name"], scene_option=opt)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        depth, _, _ = simulated_depth_outputs(depth, cam)
        obs = b.observe()
        obs.rgb = rgb
        obs.depth_m = depth
        return obs

    def observe_targets(self):
        found = []
        for angle in self.task["controller"]["scan_joint1_rad"]:
            q = np.array(self.task["controller"]["observe_q"])
            q[0] = angle
            self.stage = "observe"
            self.move_joint(q, 0.07)
            self.hold(0.15)
            detections = detect_blocks(
                self.capture(), self.b.camera, self.task["scene"]["table_top_m"]
            )
            for detection in detections:
                # Ignore already placed targets and points outside the configured pick workspace.
                if detection.position[1] < -0.16:
                    continue
                if not any(
                    np.linalg.norm(d.position[:2] - detection.position[:2]) < 0.025 for d in found
                ):
                    found.append(detection)
        return found

    def pick_place(self, detection):
        cfg = self.task["controller"]
        scene = self.task["scene"]
        center = detection.position.copy()
        center[2] = scene["table_top_m"] + scene["cube_size_m"] / 2
        yaw = detection.yaw + np.pi
        self.log.append(
            {
                "t": float(self.b.data.time),
                "event": "vision_target",
                "color": detection.color,
                "position_m": center.tolist(),
                "yaw_rad": yaw,
            }
        )
        target_zone = "l1" if self.judge.mode == "l1" else detection.color
        zone = scene["bins"][target_zone]
        # Three lanes within L1; two slots per class in L2.
        placed = self.placed_counts[target_zone]
        destination = np.array(
            [*zone["center_xy"], scene["table_top_m"] + scene["cube_size_m"] / 2 + 0.003]
        )
        if self.judge.mode == "l1":
            destination[0] += (placed - 1) * 0.05
        else:
            destination[1] += (placed - 0.5) * 0.06
        hover = center.copy()
        hover[2] = cfg["transport_height_m"]
        self.stage = "pregrasp"
        self.move_joint(self.ik.solve(hover, yaw, self.b.measured()[:6]), 0.07)
        self.stage = "descend"
        self.cartesian(center, yaw, 0.07)
        self.stage = "close"
        self.move_joint(self.b.measured()[:6], 0.0)
        self.hold(0.3)
        self.stage = "lift"
        self.cartesian(hover, yaw, 0.0)
        self.hold(1.1)
        if self.b.measured()[6] < scene["cube_size_m"] * 0.45:
            self.log.append({"t": float(self.b.data.time), "event": "grasp_feedback_empty"})
            self.move_joint(self.b.measured()[:6], 0.07)
            return False
        self.stage = "transport"
        above = destination.copy()
        above[2] = cfg["transport_height_m"]
        self.cartesian(above, yaw, 0.0)
        self.stage = "place"
        self.cartesian(destination, yaw, 0.0)
        self.stage = "release"
        self.move_joint(self.b.measured()[:6], 0.07)
        self.hold(2.2)
        self.stage = "retreat"
        self.cartesian(above, yaw, 0.07)
        self.stage = "verify"
        q = np.array(cfg["observe_q"])
        q[0] = np.arctan2(destination[1], destination[0])
        self.move_joint(q, 0.07)
        self.hold(0.2)
        visible = detect_blocks(self.capture(), self.b.camera, scene["table_top_m"])
        verified = any(
            d.color == detection.color and np.linalg.norm(d.position[:2] - destination[:2]) < 0.025
            for d in visible
        )
        self.log.append(
            {"t": float(self.b.data.time), "event": "visual_placement_check", "success": verified}
        )
        if verified:
            self.placed_counts[target_zone] += 1
        return verified

    def run(self):
        attempts = 0
        while sum(self.placed_counts.values()) < self.expected:
            detections = self.observe_targets()
            self.log.append(
                {"t": float(self.b.data.time), "event": "scan", "count": len(detections)}
            )
            if not detections:
                raise RuntimeError("No unplaced targets visible")
            # Re-observe after each manipulation so moved targets are not stale.
            target = min(detections, key=lambda d: np.linalg.norm(d.position[:2] - [0.3, 0]))
            self.pick_place(target)
            attempts += 1
            if attempts > self.expected * (self.task["controller"]["max_retries"] + 1):
                raise RuntimeError("Retry budget exhausted")
        return self.judge.result()

    def close(self):
        self.camera_renderer.close()
