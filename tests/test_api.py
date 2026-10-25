"""REST API smoke tests (fastapi.testclient: in-process, no server needed). Fast platforms only; ~20 s in total."""
import time

from fastapi.testclient import TestClient

from api import app
from circuitsleuth import extract
from ui.demos import DEMOS, make_demo

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["version"]


def test_demos_lists_every_platform():
    d = client.get("/demos").json()
    assert {"circuit", "nv", "dot", "ion", "atom"} <= set(d)
    assert any(n.startswith("F") for n in d["nv"])


def test_identify_demo_nv_fast():
    t0 = time.time()
    r = client.post("/identify/demo", json=dict(platform="nv", name="F", seed=1, preset="fast", workers=1))
    assert r.status_code == 200, r.text[:300]
    rep = r.json()
    assert rep["status"] == "ok" and rep["best"]["family"] == "nv_ensemble"
    assert abs(rep["best"]["params"]["D"] - 2.870) < 0.01
    assert rep["truth"]["params"]["D"] == 2.870  # the hidden truth rides along for demos
    assert time.time() - t0 < 60


def test_identify_points_from_dot_demo():
    name = next(n for n in DEMOS["dot"] if n.startswith("H"))
    truth, r = make_demo("dot", name, 1)
    pts = extract.extract_points(r["image"], r["x"], r["f"])
    body = dict(points=pts.tolist(), band=[float(r["f"][0]), float(r["f"][-1])], families=["charge_qubit"], preset="fast", workers=1)
    resp = client.post("/identify", json=body)
    assert resp.status_code == 200, resp.text[:300]
    best = resp.json()["best"]
    assert best["family"] == "charge_qubit"
    assert abs(best["params"]["t_c"] - truth["params"]["t_c"]) / truth["params"]["t_c"] < 0.05


def test_bad_input_is_rejected_before_fitting():
    r = client.post("/identify", json=dict(image=[[1, 2, 3], [4, 5, 6]], x=[0, 1], f=[5, 6]))  # 3 columns, len(x) == 2
    assert r.status_code == 422
    r = client.post("/identify", json=dict(points=[[0.0, 1.0, 1.0]] * 3))  # too few points
    assert r.status_code == 422
    r = client.post("/identify", json=dict(points=[[0.0, 1.0]] * 12, families=["warp_drive"]))  # unknown family
    assert r.status_code == 422
    r = client.post("/identify/demo", json=dict(platform="moon", name="F"))  # unknown platform
    assert r.status_code == 422
    r = client.post("/identify/demo", json=dict(platform="nv", name="Z"))  # unknown demo
    assert r.status_code == 404
