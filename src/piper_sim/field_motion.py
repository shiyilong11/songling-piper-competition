"""Check each sampled joint command against robot geometry and the observed table."""

import mujoco
import numpy as np

from piper_sim.calibration import body_pose


class TableClearanceError(RuntimeError):
    """A proposed command was rejected by the robot/table clearance check."""


class TableClearance:
    def __init__(self, robot, calibration, table_m):
        if not np.isfinite(table_m) or not -0.03 < table_m < 0.30:
            raise ValueError("Invalid observed table height for PiPER")
        self.model = robot.model
        self.data = mujoco.MjData(self.model)
        self.table = table_m
        self.calibration = calibration
        self.end_frame = robot.interface["frames"]["end"]
        self.tcp = self.model.site("tcp").id
        self.geoms = []
        for g in range(self.model.ngeom):
            name = self.model.body(self.model.geom_bodyid[g]).name
            if name in ("dummy_link", "base_link", "link1") or not self.model.geom_contype[g]:
                continue
            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = self.model.geom_dataid[g]
                start = self.model.mesh_vertadr[mesh]
                points = self.model.mesh_vert[start : start + self.model.mesh_vertnum[mesh]]
            elif self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                points = self.model.geom_size[g] * np.array(
                    [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
                )
            else:
                raise ValueError(f"Unsupported field clearance geom on {name}")
            self.geoms.append((g, name, np.ascontiguousarray(points, dtype=float)))

    def __call__(self, measured, target):
        measured, target = np.asarray(measured, float), np.asarray(target, float)
        if (
            measured.shape != (7,)
            or target.shape != (7,)
            or not np.isfinite([measured, target]).all()
        ):
            raise ValueError("Expected finite arm feedback and command")
        steps = max(1, int(np.ceil(np.max(np.abs(target[:6] - measured[:6])) / 0.03)))
        for state in np.linspace(measured, target, steps + 1):
            self.data.qpos[:] = np.r_[state[:6], state[6] / 2, -state[6] / 2]
            mujoco.mj_kinematics(self.model, self.data)
            if self.data.site_xpos[self.tcp, 2] < self.table + 0.010:
                raise TableClearanceError(
                    "Command would bring the TCP too close to the observed table"
                )
            for geom, name, points in self.geoms:
                z_axis = self.data.geom_xmat[geom].reshape(3, 3)[2]
                z_min = self.data.geom_xpos[geom, 2] + np.min(points @ z_axis)
                if z_min < self.table - 0.001:
                    raise TableClearanceError(
                        f"Command would bring {name} through the observed table "
                        f"({1000 * (self.table - z_min):.1f} mm)"
                    )
            if not self.calibration.fixed:
                camera = (
                    body_pose(self.model, self.data, self.end_frame) @ self.calibration.transform
                )
                if camera[2, 3] < self.table + 0.035:
                    raise TableClearanceError(
                        "Command would bring the wrist camera too close to the table"
                    )
