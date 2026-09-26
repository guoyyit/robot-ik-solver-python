import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app, service
from robot_ik import KinematicModel
from conftest import mat_to_quat_wxyz


@pytest.fixture()
def client():
    return TestClient(app)


def _fk_target(q):
    T = KinematicModel(service.current_config()).forward_kinematics(np.array(q))
    return {
        "position": [float(x) for x in T[:3, 3]],
        "orientation_wxyz": [float(x) for x in mat_to_quat_wxyz(T[:3, :3])],
    }


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_get_config(client):
    body = client.get("/config").json()
    assert body["dof"] == 6
    assert body["version"]
    assert len(body["joints"]) == 6


def test_solve_success(client):
    res = client.post("/solve", json={"target": _fk_target([0.3, -0.5, 0.8, 0.2, -0.4, 0.6])})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert len(body["joint_angles"]) == 6
    assert body["singularity"]["is_singular"] is False
    assert body["config_version"]


def test_solve_unreachable(client):
    res = client.post("/solve", json={"target": {"position": [10.0, 0.0, 0.0]}})
    assert res.json()["status"] == "unreachable"
    assert res.json()["joint_angles"] is None


def test_solve_invalid_request_structured(client):
    res = client.post(
        "/solve",
        json={"target": {"position": [0, 0, 0]}, "previous_joints": [1.0, 2.0]},
    )
    body = res.json()
    assert body["status"] == "invalid_request"
    assert body["joint_angles"] is None


def test_config_update_flow_and_conflict(client):
    current = client.get("/config").json()
    joints = [
        {
            "name": j["name"],
            "axis": list(j["axis"]),
            "translation": list(j["translation"]),
            "lower_limit": j["lower_limit"],
            "upper_limit": j["upper_limit"],
        }
        for j in current["joints"]
    ]
    # conflicting expected version -> 409
    conflict = client.put(
        "/config",
        json={
            "name": "arm6",
            "version": "vX",
            "joints": joints,
            "expected_version": "does-not-exist",
        },
    )
    assert conflict.status_code == 409

    # valid update with correct expected version
    ok = client.put(
        "/config",
        json={
            "name": "arm6",
            "version": "v2-api",
            "joints": joints,
            "expected_version": current["version"],
        },
    )
    assert ok.status_code == 200
    assert client.get("/config").json()["version"] == "v2-api"


def test_config_update_invalid_body_rejected(client):
    version_before = client.get("/config").json()["version"]
    res = client.put("/config", json={"name": "bad", "version": "v9", "joints": []})
    assert res.status_code == 422
    assert client.get("/config").json()["version"] == version_before


def test_api_reproducibility(client):
    target = _fk_target([0.3, -0.5, 0.8, 0.2, -0.4, 0.6])
    r1 = client.post("/solve", json={"target": target}).json()
    r2 = client.post("/solve", json={"target": target}).json()
    assert r1["joint_angles"] == r2["joint_angles"]
