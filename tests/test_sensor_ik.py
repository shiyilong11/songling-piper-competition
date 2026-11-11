import numpy as np
from scipy.spatial.transform import Rotation

from piper_sim.control import load_interface
from piper_sim.sensor_ik import SensorIK
from piper_sim.sensor_robot import SensorRobot


def test_analytic_robot_jacobian_matches_numerical_derivative():
    state = np.array([0.0, 0.4, -0.8, 0.0, 1.2, 0.0, 0.07])
    robot = SensorRobot(load_interface(), lambda: (0.0, state), lambda a: None, lambda t: None)
    ik = SensorIK(robot)
    rng = np.random.default_rng(12)
    for _ in range(8):
        q = rng.uniform(ik.lower + 0.01, ik.upper - 0.01)
        rotation = Rotation.random(random_state=rng).as_matrix()
        position = np.array([0.4, -0.1, 0.2])
        _, analytic = ik.residual_and_jacobian(q, position, rotation)
        numeric = np.column_stack(
            [
                (
                    ik.residual_and_jacobian(q + np.eye(6)[i] * 1e-6, position, rotation)[0]
                    - ik.residual_and_jacobian(q - np.eye(6)[i] * 1e-6, position, rotation)[0]
                )
                / 2e-6
                for i in range(6)
            ]
        )
        assert np.allclose(analytic, numeric, atol=1e-7)


def test_solved_pose_matches_position_and_orientation_tolerances():
    state = np.array([0.0, 0.4, -0.8, 0.0, 1.2, 0.0, 0.07])
    robot = SensorRobot(load_interface(), lambda: (0.0, state), lambda a: None, lambda t: None)
    ik = SensorIK(robot, tilt=0.6)
    rotation = (
        Rotation.from_euler("z", np.pi).as_matrix()
        @ Rotation.from_euler("y", 0.6).as_matrix()
        @ np.diag([1.0, -1.0, -1.0])
    )
    position = np.array([0.36, 0.03, 0.22])
    solution = ik.solve(position, np.pi, state[:6])
    error, _ = ik.residual_and_jacobian(solution, position, rotation)
    assert np.linalg.norm(error[:3]) < 0.0015
    assert np.linalg.norm(error[3:]) < 0.005


def test_near_base_grasp_uses_reachable_local_hover_and_keeps_transport_height():
    from piper_sim.demo_skill import TargetEstimate
    from piper_sim.sensor_pick_place import SensorPickPlaceSkill

    # Rounded sensor-observed regression pose. No simulator object/contact state.
    state = np.array([0.118, 0.027, -0.636, -0.129, 1.220, -1.900, 0.07])
    robot = SensorRobot(load_interface(), lambda: (0.0, state), lambda a: None, lambda t: None)
    center = np.array([0.271, 0.054, 0.075])
    target = TargetEstimate(
        center, 0.0294, 2.3635, np.ones((2, 2), bool), np.array([center]), [0, 0], 0.0
    )
    cfg = {
        "prefer_independent_orientation": True,
        "transport_height_m": 0.22,
        "grasp_yaw_candidates_rad": [target.yaw + np.pi, target.yaw],
        "grasp_tilt_candidates_rad": [0.6, 0.9, 0.35, 0.15, -0.15, -0.35, -0.6, -0.9],
        "place_yaw_rad": np.pi / 2,
    }
    motion = SensorPickPlaceSkill(robot, 0.05991, lambda stage: None, cfg)
    plan = motion.plan(target, np.array([0.137, 0.298]), 0.0605)
    assert plan["local_pickup_hover"]
    assert plan["hover"][2] >= 0.15991
    assert plan["hover"][2] < 0.22
    assert plan["hover"][2] - plan["center"][2] >= 0.04
    assert plan["above"][2] == 0.22
    assert motion.cfg["transport_height_m"] == 0.22
    assert plan["reorientation"]
