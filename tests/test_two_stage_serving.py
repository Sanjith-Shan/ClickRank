"""The two stage recommend service, on the synthetic artifacts.

Skipped when the synthetic artifacts are not there. They are produced by
make_synthetic_ids.py, run_retrieval.py and run_rankers.py on the synthetic
sample (see docs/TWO_STAGE_SERVING.md).
"""

import os

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("fastapi")

ART = os.path.join("results", "synthetic", "retrieval", "artifacts")
NEEDED = ["two_tower.pt", "corpus.npy", "ranker_deepfm.pt"]
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(os.path.join(ART, f)) for f in NEEDED)
    or not os.path.exists(os.path.join("data", "synthetic_ids", "raw_sample.csv")),
    reason="synthetic retrieval artifacts are missing",
)


@pytest.fixture(scope="module")
def bundle():
    import torch

    from src.retrieval import index as ix
    from src.serving.two_stage import TwoStageConfig, load_bundle

    # load_bundle sets one torch and one FAISS thread for the process, as a
    # serving worker should. Put the defaults back so later tests are not slowed.
    before_torch, before_faiss = torch.get_num_threads(), ix.faiss.omp_get_max_threads()
    b = load_bundle(TwoStageConfig(source="synthetic", index="flat", ranker="deepfm"))
    yield b
    torch.set_num_threads(before_torch)
    ix.set_threads(before_faiss)


@pytest.fixture(scope="module")
def client(bundle):
    from fastapi.testclient import TestClient

    from src.serving.two_stage import create_app

    return TestClient(create_app(bundle))


def _known_user(bundle):
    return int(bundle.enc.user_raw_ids[0])


def test_round_trip(client, bundle):
    r = client.post("/v1/recommend", json={"user_id": _known_user(bundle), "pid": "430548_1007",
                                           "hour": 12, "k": 100, "n": 5})
    assert r.status_code == 200
    js = r.json()
    assert js["cold_start"] is False and js["pid_known"] is True
    assert js["retrieved"] == 100 and len(js["ads"]) == 5
    p = [a["p_click"] for a in js["ads"]]
    assert p == sorted(p, reverse=True) and all(0.0 <= x <= 1.0 for x in p)
    assert all(a["ad_id"] in set(bundle.enc.ad_raw_ids.tolist()) for a in js["ads"])
    for s in ("user_embed", "search", "feature_build", "rank", "total"):
        assert js["timings_us"][s] >= 0


def test_cold_start_user_and_unknown_pid(client):
    r = client.post("/v1/recommend", json={"user_id": 10**12, "pid": "no_such_slot", "k": 50, "n": 10})
    assert r.status_code == 200
    js = r.json()
    assert js["cold_start"] is True and js["pid_known"] is False
    assert len(js["ads"]) == 10
    m = client.get("/metrics", params={"format": "json"}).json()
    assert m["cold_start_total"] >= 1 and m["unknown_pid_total"] >= 1


def test_cold_row_has_no_features_or_history(bundle):
    row = bundle.cold_user_row
    assert (bundle.enc.user_feat[row] == 0).all()
    h = bundle.hist.history(np.array([row]), np.array([bundle.enc.cutoff_ts]), 20)
    assert (h == 0).all()


def test_bounds_rejected(client, bundle):
    u = _known_user(bundle)
    assert client.post("/v1/recommend", json={"user_id": u, "pid": 0, "k": 100000}).status_code == 422
    assert client.post("/v1/recommend", json={"user_id": u, "pid": 0, "k": 0}).status_code == 422
    assert client.post("/v1/recommend", json={"user_id": u, "pid": 0, "hour": 24}).status_code == 422
    assert client.post("/v1/recommend", json={"user_id": u, "pid": 0, "extra": 1}).status_code == 422


def test_health_ready_metrics(client):
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/readyz").status_code == 200
    text = client.get("/metrics").text
    assert "clickrank_two_stage_total_seconds_bucket" in text
    assert "clickrank_two_stage_requests_total" in text


def test_parity_served_equals_offline(bundle):
    from src.serving.two_stage import parity_check

    p = parity_check(bundle, samples=10)
    assert p["compared"] >= 5
    assert p["passed"], p


def test_readyz_fails_when_parity_failed(bundle):
    from fastapi.testclient import TestClient

    from src.serving.two_stage import create_app

    c = TestClient(create_app(bundle, parity={"passed": False, "compared": 3}))
    assert c.get("/readyz").status_code == 503
