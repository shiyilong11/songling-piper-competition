"""Robot-only kinematic facade. Populated from encoders, never from scene state."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from piper_sim.calibration import ROOT, body_pose
from piper_sim.control import action_to_target


class SensorRobot:
    def __init__(self, interface, read_state, send_action, send_hold):
        path = ROOT / "assets/piper/scene.xml"
        root = ET.parse(path).getroot()
        # from_xml_string has no originating file for relative mesh paths.
        # Resolve against the packaged model directory after relocation.
        for mesh in root.findall("./asset/mesh"):
            mesh.set("file", str(path.parent / mesh.attrib["file"]))
        world = root.find("worldbody")
        for item in list(world):
            if item.tag != "body" or item.get("name") != "dummy_link":
                world.remove(item)
        root.find(".//site[@name='tcp']").set("pos", "0 0 .122")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        assert self.model.nq == 8, "Robot model must contain only six joints and two fingers"
        self.data = mujoco.MjData(self.model)
        self.interface = interface
        self.qindices, self.fingers = list(range(6)), [6, 7]
        self.read_state, self.send_action, self.send_hold = read_state, send_action, send_hold
        self.sync()
        self.previous_target = self.measured()

    def sync(self):
        stamp, state = self.read_state()
        state = np.asarray(state)
        if state.shape != (7,) or not np.isfinite(state).all() or not np.isfinite(stamp):
            raise ValueError("Invalid encoder feedback")
        self.state = state.copy()
        self.data.qpos[:] = np.r_[state[:6], state[6] / 2, -state[6] / 2]
        self.data.time = stamp
        mujoco.mj_forward(self.model, self.data)

    def measured(self):
        return self.state.copy()

    def step(self, action):
        previous = self.previous_target
        self.previous_target = action_to_target(action, self.measured(), previous, self.interface)
        try:
            self.send_action(action)
        except (RuntimeError, TimeoutError, ValueError):
            self.previous_target = previous
            raise
        self.sync()

    def _set_target(self, target):
        self.send_hold(target)

    def end_pose(self, joint_position_rad=None):
        if joint_position_rad is None:
            return body_pose(self.model, self.data, self.interface["frames"]["end"])
        q = np.asarray(joint_position_rad)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Invalid synchronized camera-frame joint feedback")
        data = mujoco.MjData(self.model)
        data.qpos[:6] = q
        mujoco.mj_kinematics(self.model, data)
        return body_pose(self.model, data, self.interface["frames"]["end"])
