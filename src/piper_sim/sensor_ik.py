"""Robot-model-only IK using analytic kinematic derivatives, with unchanged tolerances."""

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from piper_sim.traditional import IK


def rotation_log_right_inverse(vector):
    x, y, z = vector
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    theta = np.linalg.norm(vector)
    coefficient = (
        1 / 12 + theta**2 / 720
        if theta < 1e-4
        else (1 - theta / (2 * np.tan(theta / 2))) / theta**2
    )
    return np.eye(3) + 0.5 * cross + coefficient * cross @ cross


class SensorIK(IK):
    def residual_and_jacobian(self, q, position, rotation):
        self.data.qpos[:6] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        actual = self.data.site_xmat[self.site].reshape(3, 3)
        error = Rotation.from_matrix(rotation @ actual.T).as_rotvec()
        linear = np.zeros((3, self.model.nv))
        angular = np.zeros_like(linear)
        mujoco.mj_jacSite(self.model, self.data, linear, angular, self.site)
        residual = np.r_[self.data.site_xpos[self.site] - position, 0.15 * error]
        jacobian = np.vstack(
            (linear[:, :6], -0.15 * rotation_log_right_inverse(error) @ angular[:, :6])
        )
        return residual, jacobian

    def solve(self, position, yaw, seed):
        rotation = (
            Rotation.from_euler("z", yaw).as_matrix()
            @ Rotation.from_euler("y", self.tilt).as_matrix()
            @ np.diag([1.0, -1.0, -1.0])
        )
        cache = {}

        def evaluate(q):
            if "q" not in cache or not np.array_equal(cache["q"], q):
                cache["q"] = q.copy()
                cache["values"] = self.residual_and_jacobian(q, position, rotation)
            return cache["values"]

        best = None
        for start in [seed, [0, 1.4, -0.8, 0, 1.0, 0], [0, 1.8, -1.2, 0, 0.8, 0]]:
            solution = least_squares(
                lambda q: evaluate(q)[0],
                np.clip(start, self.lower + 1e-5, self.upper - 1e-5),
                jac=lambda q: evaluate(q)[1],
                bounds=(self.lower + 1e-6, self.upper - 1e-6),
                max_nfev=120,
                ftol=1e-7,
                xtol=1e-7,
                gtol=1e-7,
            )
            error = evaluate(solution.x)[0]
            if best is None or np.linalg.norm(error) < np.linalg.norm(best):
                best = error.copy()
            if np.linalg.norm(error[:3]) < 0.0015 and np.linalg.norm(error[3:]) < 0.005:
                return solution.x
        raise RuntimeError(
            f"IK unreachable: target={np.round(position, 3)}, error={np.round(best, 3)}"
        )
