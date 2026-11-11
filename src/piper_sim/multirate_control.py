"""Independent policy, joint and vision clocks with no timestamp restamping.

The 200 Hz joint setting follows AgileX support. A 1000 Hz simulation step is
our numerical choice so every joint interval contains exactly five physics steps.
This transport never enables or communicates with a physical robot.
"""

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

from piper_sim.calibration import ROOT, body_pose
from piper_sim.control import action_to_target, target_to_action
from piper_sim.observation import Observation
from piper_sim.sensor_rl_simulation import SimulationSensorDriver

DEFAULT_PROFILE = ROOT/"config/learning/piper_multirate.yaml"


@dataclass(frozen=True)
class RateProfile:
    policy_hz: int = 20
    joint_command_hz: int = 200
    joint_feedback_hz: int = 200
    depth_hz: int = 10
    mask_hz: int = 10
    physics_hz: int = 1000
    max_visual_age_s: float = .25
    max_joint_age_s: float = .015

    def validate(self):
        rates = [self.policy_hz, self.joint_command_hz, self.joint_feedback_hz,
                 self.depth_hz, self.mask_hz, self.physics_hz]
        if any(type(rate) is not int or rate <= 0 for rate in rates):
            raise ValueError("Rates must be positive integer Hertz")
        if (self.joint_command_hz != self.joint_feedback_hz
                or self.physics_hz % self.joint_command_hz
                or self.joint_command_hz % self.policy_hz
                or self.policy_hz % self.depth_hz or self.depth_hz != self.mask_hz):
            raise ValueError("Rates need integer clock ratios and same-frame depth/mask")
        if self.max_visual_age_s < 1/self.depth_hz or self.max_joint_age_s < 1/self.joint_feedback_hz:
            raise ValueError("Age limits shorter than sensor periods")
        return self


def load_rate_profile(path=DEFAULT_PROFILE):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg["schema_version"] != 1:
        raise ValueError("Unknown rate profile schema")
    return RateProfile(**cfg["rates"], max_visual_age_s=cfg["control"]["max_visual_age_s"],
                       max_joint_age_s=cfg["control"]["max_joint_age_s"]).validate()


@dataclass(frozen=True)
class JointTick:
    timestamp_s: float
    position_rad: np.ndarray
    velocity_rad_s: np.ndarray
    gripper_width_m: float
    T_base_end: np.ndarray
    issued_target: np.ndarray


@dataclass(frozen=True)
class MultiRatePacket:
    robot: JointTick
    vision: Observation
    vision_new: bool
    joint_ticks: tuple[JointTick, ...]
    control_time_s: float

    @property
    def vision_age_s(self):
        return max(0., self.control_time_s-self.vision.timestamp_s)


class MultiRateJointStepper:
    def __init__(self, backend, profile, *, physics_observer=None):
        self.b, self.profile = backend, profile.validate()
        self.b.interface = copy.deepcopy(self.b.interface)
        self.b.interface["control"].update(policy_hz=profile.policy_hz, physics_hz=profile.physics_hz)
        self.b.model.opt.timestep = 1/profile.physics_hz
        self.b.substeps = profile.physics_hz//profile.policy_hz
        self.control_interface = copy.deepcopy(self.b.interface)
        self.control_interface["control"]["policy_hz"] = profile.joint_command_hz
        self.policy_index = 0
        self.vision = None
        self.physics_steps = self.joint_updates = self.visual_updates = 0
        self.physics_observer = physics_observer

    def robot_sample(self):
        b = self.b
        return JointTick(float(b.data.time), b.data.qpos[b.qindices].copy(),
                         b.data.qvel[b.vindices].copy(), float(b.measured()[6]),
                         body_pose(b.model, b.data, b.interface["frames"]["end"]),
                         b.previous_target.copy())

    def reset(self):
        self.policy_index = 0
        self.physics_steps = self.joint_updates = self.visual_updates = 0
        self.vision = self.b.observe()
        return MultiRatePacket(self.robot_sample(), self.vision, True, (), float(self.b.data.time))

    def step(self, action, *, hold=False):
        if self.vision is None:
            raise RuntimeError("Reset the independent clocks first")
        b, profile = self.b, self.profile
        start = b.previous_target.copy()
        goal = start.copy() if hold else action_to_target(action, b.measured(), start, b.interface)
        ticks = []
        count = profile.joint_command_hz//profile.policy_hz
        for index in range(count):
            desired = start+(goal-start)*((index+1)/count)
            tick_action = target_to_action(desired, b.previous_target, self.control_interface)
            target = action_to_target(tick_action, b.measured(), b.previous_target, self.control_interface)
            b.previous_target = target
            b._set_target(target)
            for _ in range(profile.physics_hz//profile.joint_command_hz):
                mujoco.mj_step(b.model, b.data)
                self.physics_steps += 1
                if self.physics_observer is not None:
                    self.physics_observer(b.model, b.data)
            mujoco.mj_forward(b.model, b.data)
            ticks.append(self.robot_sample())
            self.joint_updates += 1
        self.policy_index += 1
        fresh = self.policy_index % (profile.policy_hz//profile.depth_hz) == 0
        if fresh:
            candidate = b.observe()
            fresh = (candidate.timestamp_s > self.vision.timestamp_s
                     and abs(candidate.timestamp_s-float(b.data.time)) < 1e-8
                     and candidate.rgb is not None and candidate.depth_m is not None)
            if fresh:
                self.vision = candidate
                self.visual_updates += 1
        # The cached visual Observation retains its original joints and end pose.
        return MultiRatePacket(ticks[-1], self.vision, fresh, tuple(ticks), float(b.data.time))


class MultiRateSimulationDriver(SimulationSensorDriver):
    def __init__(self, output, *, split="train", profile=DEFAULT_PROFILE):
        super().__init__(output, split=split)
        self.profile = load_rate_profile(profile) if not isinstance(profile, RateProfile) else profile.validate()
        self.clock = None
        source = Path(__file__).resolve()
        (self.output/"multirate_manifest.json").write_text(json.dumps({
            "rates": vars(self.profile), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "joint_rate_source": "https://github.com/agilexrobotics/piper_sdk/issues/130",
            "physics_rate_source": "simulation_design_not_hardware_specification",
            "hardware_connected": False, "historical_data_modified": False,
            "initial_reset_physics_hz": 500, "runtime_physics_hz": self.profile.physics_hz,
        }, indent=2))

    def reset(self, seed=0, *, in_hand=False):
        super().reset(seed, in_hand=in_hand)
        self._world.backend.camera["camera"]["fps"] = self.profile.depth_hz
        self.clock = MultiRateJointStepper(self._world.backend, self.profile)
        return self.clock.reset()

    def step(self, action, *, hold=False):
        packet = self.clock.step(action, hold=hold)
        self.previous_action = np.asarray(action, np.float32).copy() if not hold else np.zeros(7, np.float32)
        self.previous_action_valid = not hold
        return packet
