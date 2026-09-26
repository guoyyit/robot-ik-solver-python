"""HTTP contract: diagnoses, config hot reload, version checks."""

from __future__ import annotations

import pytest

from fastapi.testclient import TestClient

from ik_service.api import create_app
from ik_service.config import ConfigStore

from .conftest import planar_3r


@pytest.fixture
def client():
    store = ConfigStore(planar_3r())
    app = create_app_from_store(store)
    with TestClient(app) as c:
        yield c, store


def create_app_from_store(store):
    from ik_service.service import IKService

    return create_app(IKService(store))


def test_health_and_config(client):
    c, _ = client
    assert c.get("/health").json()["version"] == 1
    body = c.get("/config").json()
    assert body["name"] == "planar-3r"
    assert len(body["joints"]) == 3


def test_solve_success_contract(client):
    c, _ = client
    r = c.post(
        "/ik",
        json={"target": {"position": [1.5, 0.5, 0.0], "quaternion_wxyz": [1, 0, 0, 0]}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert isinstance(body["joints"], list) and len(body["joints"]) == 3
    assert body["reason"] is None
    assert body["config_version"] == 1
    assert set(body["singularity"]) == {
        "is_singular",
        "is_near_singular",
        "smallest_singular_value",
        "manipulability",
        "affected_direction",
    }


def test_unreachable_http_422_distinguishable(client):
    c, _ = client
    r = c.post(
        "/ik",
        json={"target": {"position": [10.0, 10.0, 0.0], "quaternion_wxyz": [1, 0, 0, 0]}},
    )
    assert r.status_code == 422
    assert r.json()["reason"] == "unreachable"
    assert r.json()["joints"] is None


def test_timeout_http(client):
    c, _ = client
    r = c.post(
        "/ik",
        json={
            "target": {"position": [10.0, 0.0, 0.0], "quaternion_wxyz": [1, 0, 0, 0]},
            "timeout_override": 0.005,
        },
    )
    assert r.status_code == 422
    assert r.json()["reason"] == "timeout"
    assert r.json()["joints"] is None


def test_invalid_payload_http(client):
    c, _ = client
    r = c.post(
        "/ik",
        json={"target": {"position": [0.0, 0.0], "quaternion_wxyz": [1, 0, 0, 0]}},
    )
    assert r.status_code == 422


def test_version_mismatch_http(client):
    c, _ = client
    r = c.post(
        "/ik",
        json={
            "target": {"position": [1.0, 0.0, 0.0], "quaternion_wxyz": [1, 0, 0, 0]},
            "expected_version": 42,
        },
    )
    assert r.status_code == 422
    assert r.json()["reason"] == "version_mismatch"


def test_config_hot_reload_http(client):
    c, store = client
    payload = {
        "name": "reloaded",
        "joints": [
            {"name": "j0", "origin_xyz": [0, 0, 0]},
            {"name": "j1", "origin_xyz": [1, 0, 0]},
            {"name": "j2", "origin_xyz": [1, 0, 0]},
        ],
        "random_seed": 123,
    }
    r = c.put("/config", json=payload)
    assert r.status_code == 200
    assert r.json()["version"] == 2
    assert store.current().version == 2
    # In-flight-equivalent pin: requests may demand the new version.
    ok = c.post(
        "/ik",
        json={
            "target": {"position": [1.0, 0.0, 0.0], "quaternion_wxyz": [1, 0, 0, 0]},
            "expected_version": 2,
        },
    )
    assert ok.status_code == 200


def test_config_reload_rejects_invalid(client):
    c, store = client
    r = c.put("/config", json={"name": "bad", "joints": []})
    assert r.status_code == 422
    assert r.json()["reason"] == "invalid_config"
    assert store.current().version == 1


def test_reproducibility_over_http(client):
    c, _ = client
    body = {"target": {"position": [1.4, 0.6, 0.0], "quaternion_wxyz": [1, 0, 0, 0]},
            "seed_override": 99}
    a = c.post("/ik", json=body).json()
    b = c.post("/ik", json=body).json()
    assert a["joints"] == b["joints"]
