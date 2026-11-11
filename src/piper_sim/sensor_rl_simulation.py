"""Simulation transport for sensor-only skill RL.

The simulator owns hidden object state for physics/reset. The RL side receives
only Observation and previously issued robot commands. Its transitions bypass
the competition evaluator; no true-object reward or termination is returned.
"""

import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

from piper_sim.calibration import ROOT, load_config
from piper_sim.competition_world import CompetitionWorld
from piper_sim.scene_observation import CalibrationEstimate, Provenance

CALIBRATION = ROOT / "config/cameras/dabai_dc1_calibrated.yaml"


class _PhysicsResetWorld(CompetitionWorld):
    def export_metadata(self):
        # The RL transport does not create or read evaluator answers.
        pass


def prepare_scene(output, *, width=320, height=240, split="train"):
    """Separate training reset specification from measured camera parameters."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    camera = copy.deepcopy(load_config(CALIBRATION))
    original = camera["camera"]
    sx, sy = width/original["width"], height/original["height"]
    k = original["intrinsics"]
    k.update(fx=k["fx"]*sx, fy=k["fy"]*sy,
             cx=(k["cx"]+.5)*sx-.5, cy=(k["cy"]+.5)*sy-.5)
    original.update(width=width, height=height)
    # MuJoCo generates rectified pinhole RGB/depth, not distorted source images.
    original["distortion"] = {"model": "opencv_plumb_bob", "coefficients": [0., 0., 0., 0., 0.]}
    camera_path = output / "rectified_camera.yaml"
    camera_path.write_text(yaml.safe_dump(camera, allow_unicode=True, sort_keys=False))
    cfg = yaml.safe_load((ROOT / "config/competition/scene_single_table.yaml").read_text())
    cfg["status"] = "sensor_only_rl_training_curriculum_not_official_layout"
    cfg["camera_config"] = str(camera_path)
    cfg["camera"]["mount"] = "wrist"
    cfg["robot_initial_qpos"] = camera["simulation"]["initial_qpos"]
    cfg["table"].update(center_xy_m=[.28, 0.], half_size_xy_m=[.20, .30], top_m=0., thickness_m=.035)
    cfg.setdefault("environment", {})["ground_z_m"] = -.08
    cfg["spawn"].update(x_m=[.17, .23], y_m=[.02, .07], settle_s=.35)
    cfg["trays"] = {"green": copy.deepcopy(cfg["trays"]["green"])}
    cfg["trays"]["green"].update(center_xy_m=[.25, -.13], half_size_xy_m=[.08, .08],
                                  placement_xy_m=[.25, -.13])
    shapes = {
        "box": [("box", [.018, .017, .021], [0, 0, 0])],
        "sphere": [("sphere", [.022], [0, 0, 0])],
        "cylinder": [("cylinder", [.018, .025], [0, 0, 0])],
        "capsule": [("capsule", [.016, .015], [0, 0, 0])],
        "ellipsoid": [("ellipsoid", [.026, .018, .022], [0, 0, 0])],
        "compound_l": [("box", [.025, .010, .018], [0, 0, 0]),
                       ("box", [.010, .019, .018], [-.015, .009, 0])],
    }
    names = list(shapes)[:5] if split == "train" else ["compound_l"]
    catalog = {name: {"mass_kg_range": [.025, .065], "parts": [
        {"type": kind, "size": size, "pos": pos, "rgba": [.82, .13, .12, 1.]}
        for kind, size, pos in parts]} for name, parts in shapes.items()}
    catalog_path = output / "reset_only_object_catalog.yaml"
    catalog_path.write_text(yaml.safe_dump({"objects": catalog}, sort_keys=False))
    cfg["object_catalog"] = str(catalog_path)
    cfg["tasks"] = {"sensor_rl": {"object_pool": names, "object_count": 1,
                                  "time_limit_s": 30.}}
    # Curriculum controls the distribution only, never supplies a target ID/mask.
    cfg["object_variation"]["scale_ranges"] = {name: [.85, 1.1] for name in shapes}
    cfg["object_variation"]["color_palettes"] = {name: ["original"] for name in shapes}
    path = output / "training_scene.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (CALIBRATION, ROOT/"assets/piper/scene.xml", ROOT/"config/interface.yaml")}
    (output / "source_manifest.json").write_text(json.dumps({
        "source_sha256": hashes, "camera_pixels_rescaled_from_measured": True,
        "uses_ik": False, "robot_control_enabled": False,
        "physical_dynamics": "Existing simulator estimates, not identified from hand-eye calibration",
        "hidden_state_policy": "Physics/reset only; no actor, critic, reward or termination access",
    }, indent=2))
    return path, camera


class SimulationSensorDriver:
    def __init__(self, output, *, split="train"):
        self.output = Path(output).resolve()
        path, camera = prepare_scene(self.output, split=split)
        self.calibration = CalibrationEstimate(
            np.asarray(camera["hand_eye"]["T_end_camera"], float),
            Provenance("manual_config", "Measured DC1 hand-eye; intrinsics rescaled for rectified simulation"),
        )
        self._world = _PhysicsResetWorld("sensor_rl", config=path,
                                        output=self.output/"simulator_private", render=True)
        self.previous_action = np.zeros(7, np.float32)
        self.previous_action_valid = False

    @property
    def interface(self):
        return copy.deepcopy(self._world.backend.interface)

    @property
    def previous_command(self):
        return self._world.backend.previous_target.copy()

    def reset(self, seed=0, *, in_hand=False):
        self._world.reset(seed)
        if in_hand:
            self._initialize_in_hand()
        self.previous_action[:] = 0
        self.previous_action_valid = False
        return self._world.backend.observe()

    def _initialize_in_hand(self):
        """Physical reset curriculum only: no weld, no teleport after episode start.

        This initializer may use the generated object geometry. Its output is
        exclusively normal camera/encoder observations; it does not certify a grasp.
        """
        b = self._world.backend
        joint = b.model.joint("object_1_free")
        address = int(joint.qposadr[0])
        end = b.observe().T_base_end
        b.data.qpos[address:address+3] = end[:3, 3] + end[:3, :3] @ [0., 0., .113]
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, end[:3, :3].ravel())
        b.data.qpos[address+3:address+7] = quaternion
        b.data.qvel[:] = 0.
        # All generated widths fit the physical 70 mm gripper. Close by servo,
        # rather than constrain/attach the free body to either finger.
        b.previous_target = b.measured()
        b.previous_target[6] = .005
        b._set_target(b.previous_target)
        for _ in range(round(.65 / b.model.opt.timestep)):
            mujoco.mj_step(b.model, b.data)
        mujoco.mj_forward(b.model, b.data)
        b.data.time = 0.

    def step(self, action):
        # Do not call world.step/evaluation: they maintain oracle judge events.
        observation = self._world.backend.step(action)
        self.previous_action = np.asarray(action, np.float32).copy()
        self.previous_action_valid = True
        return observation

    def close(self):
        self._world.close()
