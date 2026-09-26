import numpy as np

from robot_ik import KinematicModel
from robot_ik.kinematics import pose_error, quaternion_to_matrix


def test_fk_zero_angles_planar(planar2_config):
    model = KinematicModel(planar2_config)
    T = model.forward_kinematics(np.zeros(2))
    np.testing.assert_allclose(T[:3, 3], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)


def test_fk_quarter_turn_planar(planar2_config):
    model = KinematicModel(planar2_config)
    T = model.forward_kinematics(np.array([np.pi / 2, 0.0]))
    np.testing.assert_allclose(T[:3, 3], [0.0, 1.0, 0.0], atol=1e-12)


def test_fk_elbow_planar(planar2_config):
    # j2 carries no outgoing link, so bending it changes orientation only.
    model = KinematicModel(planar2_config)
    T = model.forward_kinematics(np.array([0.0, np.pi / 2]))
    np.testing.assert_allclose(T[:3, 3], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        T[:3, :3],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-12,
    )


def test_jacobian_matches_finite_difference(arm6_model):
    rng = np.random.default_rng(123)
    for _ in range(5):
        q = rng.uniform(-2.0, 2.0, arm6_model.dof)
        J = arm6_model.jacobian(q)
        T0 = arm6_model.forward_kinematics(q)
        eps = 1e-7
        J_num = np.zeros((6, arm6_model.dof))
        for i in range(arm6_model.dof):
            dq = q.copy()
            dq[i] += eps
            J_num[:, i] = pose_error(T0, arm6_model.forward_kinematics(dq)) / eps
        np.testing.assert_allclose(J, J_num, atol=1e-5)


def test_max_reach_is_conservative(arm6_model):
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(-np.pi, np.pi, arm6_model.dof)
        T = arm6_model.forward_kinematics(q)
        assert np.linalg.norm(T[:3, 3]) <= arm6_model.max_reach() + 1e-9


def test_quaternion_round_trip():
    from conftest import mat_to_quat_wxyz

    rng = np.random.default_rng(9)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = quaternion_to_matrix(q)
        q2 = np.array(mat_to_quat_wxyz(R))
        # quaternions are double-cover: allow sign flip
        assert min(np.linalg.norm(q - q2), np.linalg.norm(q + q2)) < 1e-9


def test_pose_error_zero_for_identical_poses(arm6_model):
    T = arm6_model.forward_kinematics(np.array([0.4, -0.3, 0.2, 1.0, 0.5, -0.7]))
    assert np.linalg.norm(pose_error(T, T)) < 1e-12
