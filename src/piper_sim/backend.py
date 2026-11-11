"""Single-scene development backend implementing the deployment observation/action contract."""

import mujoco
import numpy as np

from piper_sim.calibration import ROOT, body_pose, load_config, simulated_depth_outputs
from piper_sim.control import action_to_target, load_interface
from piper_sim.observation import Observation


class MujocoBackend:
    """step(action) advances exactly one policy period, not one physics step.

    Targets use position servos. Simulation gains are still estimates, not hardware matching.
    This is a control backend, not yet a grasp RL task with rewards/reset randomization.
    """

    def __init__(self, interface=None, camera=None, render=False, model_path=None):
        self.interface = interface or load_interface()
        self.camera = camera or load_config(ROOT / self.interface["observation"]["camera_config"])
        self.model = mujoco.MjModel.from_xml_path(
            str(model_path or ROOT / "assets/piper/scene.xml")
        )
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 1 / self.interface["control"]["physics_hz"]
        self.substeps = int(
            self.interface["control"]["physics_hz"] / self.interface["control"]["policy_hz"]
        )
        self.joints = [self.model.joint(n).id for n in self.interface["joints"]["names"]]
        self.qindices = [self.model.jnt_qposadr[j] for j in self.joints]
        self.vindices = [self.model.jnt_dofadr[j] for j in self.joints]
        self.actuators = [
            self.model.actuator(f"pos_{n}").id for n in self.interface["joints"]["names"]
        ]
        self.fingers = [self.model.joint(f"joint{i}").qposadr[0] for i in [7, 8]]
        self.finger_actuators = [self.model.actuator(f"pos_joint{i}").id for i in [7, 8]]
        np.testing.assert_allclose(
            self.model.jnt_range[self.joints, 0], self.interface["joints"]["lower_rad"]
        )
        np.testing.assert_allclose(
            self.model.jnt_range[self.joints, 1], self.interface["joints"]["upper_rad"]
        )
        cam = self.camera["camera"]
        self.renderer = (
            mujoco.Renderer(self.model, height=cam["height"], width=cam["width"])
            if render
            else None
        )
        self.previous_target = None

    def measured(self):
        width = self.data.qpos[self.fingers[0]] - self.data.qpos[self.fingers[1]]
        return np.r_[self.data.qpos[self.qindices], width]

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:8] = self.camera["simulation"]["initial_qpos"]
        self.previous_target = self.measured()
        self._set_target(self.previous_target)
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def _set_target(self, target):
        self.data.ctrl[self.actuators] = target[:6]
        self.data.ctrl[self.finger_actuators] = [target[6] / 2, -target[6] / 2]

    def step(self, action):
        if self.previous_target is None:
            raise RuntimeError("Call reset before step")
        target = action_to_target(action, self.measured(), self.previous_target, self.interface)
        self.previous_target = target
        self._set_target(target)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        # Refresh poses to match final joint state, not pre-integration state.
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def observe(self, *, render_depth=True):
        rgb = depth = None
        cam = self.camera["camera"]
        if self.renderer is not None:
            opt = mujoco.MjvOption()
            opt.geomgroup[3] = False
            if self.camera["simulation"].get("hide_sensor_view_markers", False):
                opt.geomgroup[2] = False
                opt.sitegroup[:] = False
            self.renderer.update_scene(self.data, camera=cam["name"], scene_option=opt)
            rgb = self.renderer.render().copy()
            if render_depth:
                self.renderer.enable_depth_rendering()
                self.renderer.update_scene(self.data, camera=cam["name"], scene_option=opt)
                depth = self.renderer.render().copy()
                self.renderer.disable_depth_rendering()
                depth, _, _ = simulated_depth_outputs(depth, cam)
        return Observation(
            float(self.data.time),
            self.data.qpos[self.qindices].copy(),
            self.data.qvel[self.vindices].copy(),
            float(self.measured()[6]),
            body_pose(self.model, self.data, self.interface["frames"]["end"]),
            rgb,
            depth,
            cam["intrinsics"].copy(),
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
