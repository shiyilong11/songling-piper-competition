"""Two skill environments with an exclusively public, causal sensor boundary.

RGB is used upstream by SAM/EdgeTAM. Actor and critic receive only mask/depth,
encoder feedback, issued commands and geometry derived from those observations.
Rewards are sensor proxies, not simulator-certified task success.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import cv2
import gymnasium as gym
import numpy as np

from piper_sim.calibration import transform_points
from piper_sim.colored_regions import estimate_colored_regions
from piper_sim.sam_adapter import Sam2Client
from piper_sim.scene_observation import SceneObserver
from piper_sim.video_mask_client import VideoMaskClient


def tcp_from_encoders(observation):
    """Fixed tool offset and measured end pose; forward kinematics, no IK."""
    t = observation.T_base_end
    return t[:3, 3] + t[:3, :3] @ np.array([0., 0., .122])


def backproject_mask(observation, mask, calibration):
    valid = observation.depth_valid_mask & mask
    v, u = np.nonzero(valid)
    if len(u) < 12:
        return np.empty((0, 3))
    # Deterministic subsampling, with no shape/category or renderer information.
    step = max(1, len(u)//2048)
    u, v = u[::step], v[::step]
    z = observation.depth_m[v, u]
    k = observation.intrinsics
    camera = np.column_stack(((u-k["cx"])*z/k["fx"], (v-k["cy"])*z/k["fy"], z))
    return transform_points(observation.T_base_end @ calibration.T_end_camera, camera)


@dataclass
class PublicEstimate:
    timestamp: float
    tcp: np.ndarray
    target: np.ndarray | None
    target_mask: np.ndarray
    goal_mask: np.ndarray
    goal: np.ndarray | None
    goal_half: np.ndarray | None
    table_normal: np.ndarray | None
    table_offset: float | None
    gripper: float
    tool_pixel: np.ndarray | None = None


def mask_tool_error(e):
    if e.tool_pixel is None or e.target_mask.sum() < 30:
        return None
    y, x = np.nonzero(e.target_mask)
    return float(np.linalg.norm((np.array([x.mean(), y.mean()])-e.tool_pixel)
                               / np.array(e.target_mask.shape[::-1])))


class SensorMaskPipeline:
    """Automatic public-depth proposal → SAM → causal EdgeTAM, no oracle masks."""

    def __init__(self, calibration, output):
        self.calibration, self.output = calibration, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.observer, self.sam = SceneObserver(), Sam2Client()
        self.tracker = None
        self.table = self.goal = self.goal_half = None
        self.initialization = None

    def reset(self, obs, *, skill):
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None
        scene = self.observer.build(obs, self.calibration, now_s=obs.timestamp_s)
        self.table = scene.table
        regions = [r for r in estimate_colored_regions(obs, scene) if r.region_id == "green_region"]
        self.goal = regions[0].T_base_region[:3, 3].copy() if len(regions) == 1 else None
        self.goal_half = regions[0].half_size_xy_m.copy() if len(regions) == 1 else None
        h, w = obs.depth_m.shape
        if skill == "pick":
            if len(scene.objects) != 1:
                raise ValueError(f"Public bootstrap requires one unambiguous depth candidate, got {len(scene.objects)}")
            seed = scene.objects[0].observed_mask
            y, x = np.nonzero(seed)
            point = np.array([np.median(x), np.median(y)])
            # Positive point must be inside the actual public candidate.
            i = np.argmin((x-point[0])**2 + (y-point[1])**2)
            point = [float(x[i]), float(y[i])]
            box = [max(0, int(x.min())-4), max(0, int(y.min())-4),
                   min(w, int(x.max())+5), min(h, int(y.max())+5)]
            prompt_source = "unique public depth foreground component"
        else:
            # In-hand curriculum: prompt at the publicly known gripper location.
            # This is a robot tool prior, NOT a projected simulator object pose.
            camera = np.linalg.inv(obs.T_base_end @ self.calibration.T_end_camera)
            p = transform_points(camera, tcp_from_encoders(obs)[None])[0]
            k = obs.intrinsics
            if p[2] <= 0:
                raise ValueError("Measured gripper is behind camera")
            u, v = p[:2]/p[2]*[k["fx"], k["fy"]] + [k["cx"], k["cy"]]
            point = [float(np.clip(u, 5, w-6)), float(np.clip(v, 5, h-6))]
            box = [max(0, point[0]-45), max(0, point[1]-65),
                   min(w, point[0]+45), min(h, point[1]+6)]
            prompt_source = "encoder/tool projection; in-hand object identity unverified"
        instance, meta = self.sam.segment(obs.rgb, point, obs.timestamp_s, "bootstrap",
                                          self.output/"sam", box_xyxy=box)
        self.tracker = VideoMaskClient(self.output/"tracking")
        mask = self.tracker.initialize(obs.rgb, instance.mask)
        self.initialization = {"prompt_source": prompt_source, "sam": meta,
                               "mask_origin": "SAM2.1_RGB_then_EdgeTAM",
                               "object_identity_verified": False}
        return self._estimate(obs, mask)

    def update(self, obs):
        if self.tracker is None:
            raise RuntimeError("Tracker not initialized")
        mask = self.tracker.step(obs.rgb, self.tracker.next_index)
        return self._estimate(obs, mask)

    def _estimate(self, obs, mask):
        points = backproject_mask(obs, mask, self.calibration)
        normal = None if self.table is None else self.table.normal_base.copy()
        offset = None if self.table is None else float(self.table.offset_m)
        # Exclude the observed support plane from target geometry, not from mask.
        if len(points) and normal is not None:
            points = points[(points @ normal + offset) > .008]
        target = np.median(points, axis=0) if len(points) >= 12 else None
        hsv = cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2HSV)
        goal_mask = ((hsv[..., 0] >= 35) & (hsv[..., 0] < 90)
                     & (hsv[..., 1] >= 60) & (hsv[..., 2] >= 35))
        # Colour proposes the destination only; the object mask is learned SAM/Edge.
        tcp = tcp_from_encoders(obs)
        p = transform_points(np.linalg.inv(obs.T_base_end @ self.calibration.T_end_camera), tcp[None])[0]
        k = obs.intrinsics
        tool_pixel = p[:2]/p[2]*[k["fx"], k["fy"]] + [k["cx"], k["cy"]] if p[2] > 0 else None
        return PublicEstimate(float(obs.timestamp_s), tcp, target,
                              mask.copy(), goal_mask, self.goal, self.goal_half,
                              normal, offset, float(obs.gripper_width_m), tool_pixel)

    def close(self):
        self.sam.close()
        if self.tracker is not None:
            self.tracker.close()


class PublicSkillReward:
    """Pure reward: no simulator handle, contacts, target IDs or hidden terminals."""

    def __init__(self, skill):
        if skill not in ("pick", "place"):
            raise ValueError("Unknown skill")
        self.skill = skill

    def reset(self, estimate):
        self.initial = estimate
        self.previous = estimate
        self.hold_frames = self.placement_frames = 0
        self.last_target, self.last_target_time = estimate.target, estimate.timestamp
        self.close_origin = None
        self.closure_frames = 0

    def _distance(self, e):
        if self.skill == "pick":
            target = e.target
            if (target is None and e.timestamp-self.last_target_time <= 6.
                    and e.target_mask.sum() >= 30):
                target = self.last_target
            return None if target is None else float(np.linalg.norm(e.tcp-target))
        if e.goal is None:
            return None
        # Approach above the observed destination, not a hidden ideal release pose.
        return float(np.linalg.norm(e.tcp-(e.goal + np.array([0, 0, .065]))))

    def step(self, e, action):
        before, after = self._distance(self.previous), self._distance(e)
        progress = 0. if before is None or after is None else float(np.clip(before-after, -.03, .03))
        reward = 40.*progress - .005 - .002*float(np.mean(np.square(action[:6])))
        image_before, image_after = mask_tool_error(self.previous), mask_tool_error(e)
        image_progress = (0. if image_before is None or image_after is None
                          else float(np.clip(image_before-image_after, -.05, .05)))
        if self.skill == "pick":
            reward += 2.*image_progress
            if e.target_mask.sum() < 30:
                reward -= .1
        target_valid = e.target is not None
        sensor_success = False
        lift_progress = 0.
        if self.skill == "pick":
            evidence = False
            evidence_kind = None
            if not target_valid:
                reward -= .015
            if target_valid and self.initial.target is not None and self.previous.target is not None:
                object_rise = e.target[2] - self.initial.target[2]
                tcp_rise = e.tcp[2] - self.initial.tcp[2]
                co_motion = np.linalg.norm((e.target-self.previous.target) - (e.tcp-self.previous.tcp)) < .008
                evidence = (object_rise > .04 and tcp_rise > .04 and co_motion
                            and .004 < e.gripper < .055 and e.target_mask.sum() >= 30)
                if evidence:
                    evidence_kind = "visible_3d_lift_and_co_motion"
            # Near-field fallback remains an explicitly unverified sensor proxy.
            # Nonzero width alone is insufficient: require a close command,
            # an object mask at the tool, its stability, and observed robot lift.
            near = after is not None and after < .035
            if self.close_origin is None and near and action[6] < -.95:
                self.close_origin = e.tcp.copy()
            closure = (self.close_origin is not None and action[6] < -.95
                       and .008 < e.gripper < .055 and image_after is not None
                       and image_after < .13 and abs(e.gripper-self.previous.gripper) < .001)
            self.closure_frames = self.closure_frames+1 if closure else 0
            if self.closure_frames >= 10:
                lift_progress = float(np.clip(e.tcp[2]-self.previous.tcp[2], -.01, .01))
                reward += -40.*progress + 20.*lift_progress
            nearfield = (self.close_origin is not None and action[6] < -.95
                         and .008 < e.gripper < .055 and image_after is not None
                         and image_after < .13 and image_before is not None
                         and abs(image_after-image_before) < .015
                         and e.tcp[2]-self.close_origin[2] > .04)
            if nearfield:
                evidence, evidence_kind = True, "nearfield_mask_gripper_lift_proxy_unverified"
            self.hold_frames = self.hold_frames+1 if evidence else 0
            sensor_success = self.hold_frames >= 10
        else:
            evidence = False
            if target_valid and e.goal is not None and e.goal_half is not None:
                inside = np.all(np.abs(e.target[:2]-e.goal[:2]) < e.goal_half)
                stable = self.previous.target is not None and np.linalg.norm(e.target-self.previous.target) < .003
                height = None if e.table_normal is None else float(e.target @ e.table_normal + e.table_offset)
                evidence = inside and stable and height is not None and .008 < height < .065 and e.gripper > .055
            self.placement_frames = self.placement_frames+1 if evidence else 0
            sensor_success = self.placement_frames >= 10
        if sensor_success:
            reward += 3.
        if e.target is not None:
            self.last_target, self.last_target_time = e.target.copy(), e.timestamp
        self.previous = e
        return reward, sensor_success, {
            "distance_m": after, "progress_m": progress, "target_depth_valid": target_valid,
            "mask_pixels": int(e.target_mask.sum()), "sensor_proxy_success": sensor_success,
            "image_progress": image_progress,
            "lift_progress_m": lift_progress,
            "learning_progress": (lift_progress if self.closure_frames >= 10 and self.skill == "pick"
                                  else progress + .05*image_progress),
            "nearfield_evidence": evidence_kind if self.skill == "pick" else None,
            "certified_task_success": None, "reward_source": "public_sensor_proxy",
        }


def public_actor_frame(obs, e, previous_action, previous_valid, *, last_target=None,
                       target_age=0., goal_age=0.):
    """No RGB/class/language in this function's output. Invalid depth stays explicit."""
    size = (96, 72)
    valid = obs.depth_valid_mask
    depth = np.where(valid, np.clip(obs.depth_m/2., 0, 1), 0).astype(np.float32)
    channels = [depth, valid.astype(np.float32), e.target_mask.astype(np.float32),
                e.goal_mask.astype(np.float32)]
    image = np.stack([cv2.resize(x, size, interpolation=cv2.INTER_NEAREST) for x in channels])
    target = e.target if e.target is not None else last_target
    rel = np.zeros(3) if target is None else np.clip((target-e.tcp)/.5, -2, 2)
    goal = np.zeros(3) if e.goal is None else np.clip((e.goal-e.tcp)/.5, -2, 2)
    y, x = np.nonzero(e.target_mask)
    center = [0., 0.] if not len(x) else [x.mean()/e.target_mask.shape[1], y.mean()/e.target_mask.shape[0]]
    height = 0. if e.table_normal is None else float(e.tcp @ e.table_normal + e.table_offset)
    state = np.r_[obs.joint_position_rad/np.pi, obs.joint_velocity_rad_s,
                  obs.gripper_width_m/.07, previous_action, float(previous_valid),
                  rel, float(e.target is not None), min(target_age, 10.)/10.,
                  goal, float(e.goal is not None), min(goal_age, 30.)/30.,
                  center, e.target_mask.mean(), height, float(e.table_normal is not None)].astype(np.float32)
    if state.shape != (36,) or not np.isfinite(state).all():
        raise ValueError("Invalid public actor state")
    return image.astype(np.float32), state


class SensorSkillEnv(gym.Env):
    metadata: ClassVar = {"render_modes": ["rgb_array"]}

    def __init__(self, driver, output, *, skill="pick", horizon=100, history=4):
        self.driver, self.skill = driver, skill
        self.output, self.horizon, self.history = Path(output), horizon, history
        self.perception = SensorMaskPipeline(driver.calibration, self.output/"perception")
        self.reward_model = PublicSkillReward(skill)
        self.action_space = gym.spaces.Box(-1., 1., (7,), np.float32)
        self.observation_space = gym.spaces.Dict({
            "images": gym.spaces.Box(0., 1., (history, 4, 72, 96), np.float32),
            "state": gym.spaces.Box(-np.inf, np.inf, (history, 37), np.float32),
        })
        self.frames = deque(maxlen=history)
        self.steps = self.episode = 0

    def _pack(self):
        current = self.estimate
        if current.target is not None:
            self.last_target, self.last_target_time = current.target.copy(), current.timestamp
        image, state = public_actor_frame(
            self.observation, current, self.driver.previous_action, self.driver.previous_action_valid,
            last_target=self.last_target, target_age=current.timestamp-self.last_target_time,
            goal_age=current.timestamp)
        self.frames.append((image, np.r_[state, 1.].astype(np.float32)))
        frames = [(np.zeros_like(image), np.zeros(37, np.float32))]*(self.history-len(self.frames)) + list(self.frames)
        return {"images": np.stack([f[0] for f in frames]), "state": np.stack([f[1] for f in frames])}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        # Retry only for ambiguity/visibility observable in public sensors.
        rejected = []
        for attempt in range(8):
            actual_seed = int(self.np_random.integers(0, 2**30))
            self.observation = self.driver.reset(actual_seed, in_hand=self.skill == "place")
            try:
                self.estimate = self.perception.reset(self.observation, skill=self.skill)
                if self.skill == "place" and self.estimate.goal is None:
                    raise ValueError("Observed goal region unavailable")
                break
            except ValueError as exc:
                rejected.append(str(exc))
        else:
            raise RuntimeError(f"No public-observable initialization: {rejected}")
        self.steps = 0
        self.frames.clear()
        self.last_target, self.last_target_time = None, self.estimate.timestamp
        self.reward_model.reset(self.estimate)
        return self._pack(), {"reset_seed": actual_seed, "public_rejections": rejected,
                              "initialization": self.perception.initialization,
                              "initial_in_hand_identity_verified": False}

    def step(self, action):
        action = np.asarray(action, np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Finite seven dimensional action required")
        action = np.clip(action, -1, 1)
        self.observation = self.driver.step(action)
        self.estimate = self.perception.update(self.observation)
        reward, success, info = self.reward_model.step(self.estimate, action)
        self.steps += 1
        return self._pack(), float(reward), bool(success), self.steps >= self.horizon, info

    def render(self):
        # Diagnostic only; render output is never included in actor observations.
        return self.observation.rgb.copy()

    def close(self):
        self.perception.close()
        self.driver.close()
