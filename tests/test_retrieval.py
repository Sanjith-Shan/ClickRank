"""Tests for the retrieval stage, on synthetic data only.

Every test builds its own small sample under tmp_path, so nothing here reads
the real Taobao or Avazu files and the suite runs in seconds on a CPU.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.retrieval import two_tower as tt
from src.retrieval.data import RetrievalData, load
from src.retrieval.features import ClickHistory, encode
from src.retrieval.metrics import (
    hit_rate,
    popularity_baseline,
    popularity_hit_rate,
    random_baseline_rate,
    rank_of,
    recall_at_k,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_module():
    path = os.path.join(ROOT, "scripts", "make_synthetic_ids.py")
    spec = importlib.util.spec_from_file_location("make_synthetic_ids", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("synthetic_ids"))
    _make_module().make(out, n_users=2000, n_ads=500, n_cates=20, n_impressions=60_000, seed=7)
    return out


@pytest.fixture(scope="module")
def synth(synth_dir):
    return load("synthetic", synth_dir)


@pytest.fixture(scope="module")
def enc(synth):
    return encode(synth)


# ---------------------------------------------------------------------------
# Loader and encoder


def test_loader_reads_synthetic_sample(synth):
    days = sorted(synth.impressions["day"].unique().tolist())
    assert days == list(range(1, 9))
    assert len(synth.train) + len(synth.test) == len(synth.impressions)
    s = synth.summary()
    assert s["dataset"].startswith("SYNTHETIC")
    assert s["train_impressions"] + s["test_impressions"] == s["impressions"]
    assert synth.label().startswith("SYNTHETIC")


def test_loader_cache_round_trip(synth_dir, synth):
    again = load("synthetic", synth_dir)
    pd.testing.assert_frame_equal(again.impressions, synth.impressions)


def _tiny(impressions, ads, users=None):
    imp = pd.DataFrame(impressions, columns=["user", "ad", "ts", "day", "pid", "clk"])
    for c, t in (("user", np.int64), ("ad", np.int64), ("ts", np.int64), ("day", np.int8),
                 ("pid", np.int16), ("clk", np.int8)):
        imp[c] = imp[c].astype(t)
    ad_df = pd.DataFrame(ads, columns=["ad", "cate", "campaign", "customer", "brand", "price"])
    if users is None:
        users = pd.DataFrame({"user": sorted(imp["user"].unique()), "age_level": 1})
    return RetrievalData(name="tiny", impressions=imp, ads=ad_df, users=users,
                         train_days=(1, 7), test_day=8, synthetic=True)


def test_encode_oov_rule_for_test_only_ad():
    day = 86400
    data = _tiny(
        impressions=[
            (1, 10, 1 * day, 1, 0, 1),
            (2, 11, 2 * day, 2, 0, 0),
            (1, 12, 8 * day, 8, 0, 1),  # ad 12 appears only on the test day
        ],
        ads=[(10, 5, 100, 1000, 7, 9.0), (11, 6, 101, 1001, 8, 19.0), (12, 5, 102, 1002, 7, 29.0)],
    )
    e = encode(data)
    row12 = int(np.flatnonzero(e.ad_raw_ids == 12)[0])
    row10 = int(np.flatnonzero(e.ad_raw_ids == 10)[0])
    assert e.ad_feat[row12, 0] == 0  # id is out of vocabulary
    assert e.ad_feat[row12, 1] != 0  # but its category was seen in training
    assert e.ad_feat[row12, 1] == e.ad_feat[row10, 1]
    assert e.ad_feat[row12, 2] == 0  # its campaign was not
    assert e.ad_feat[row10, 0] != 0


def test_vocab_sizes_cover_codes(enc):
    for col, size in enumerate(enc.ad_vocab_sizes):
        assert enc.ad_feat[:, col].max() < size
        assert enc.ad_feat[:, col].min() >= 0
    for col, size in enumerate(enc.user_vocab_sizes):
        assert enc.user_feat[:, col].max() < size
        assert enc.user_feat[:, col].min() >= 0
    assert enc.imp_ad.max() < enc.n_ads
    assert enc.imp_user.max() < enc.n_user_rows


# ---------------------------------------------------------------------------
# Click history


def test_history_is_strictly_before_the_query_time():
    day = 86400
    data = _tiny(
        impressions=[
            (1, 10, 1 * day + 100, 1, 0, 1),
            (1, 11, 1 * day + 200, 1, 0, 1),
            (1, 12, 1 * day + 200, 1, 0, 1),  # same timestamp as the ad 11 click
            (1, 13, 2 * day, 2, 0, 1),
            (1, 14, 8 * day, 8, 0, 1),  # test day click
            (2, 10, 1 * day + 50, 1, 0, 1),
        ],
        ads=[(a, 1, a, a, a, 1.0) for a in (10, 11, 12, 13, 14)],
    )
    e = encode(data)
    hist = ClickHistory(e)
    row = {int(r): i for i, r in enumerate(e.ad_raw_ids)}
    u1 = int(np.flatnonzero(e.user_raw_ids == 1)[0])

    # The click at +200 sees only the +100 click, not itself and not the other
    # click at the same second.
    h = hist.history(np.array([u1]), np.array([1 * day + 200]), 5)[0]
    assert list(h[h > 0] - 1) == [row[10]]

    # Most recent first, and user 2's click never leaks into user 1.
    h = hist.history(np.array([u1]), np.array([3 * day]), 5)[0]
    got = list(h[h > 0] - 1)
    assert got[0] == row[13]
    assert set(got) == {row[10], row[11], row[12], row[13]}

    # Frozen test history holds training clicks only, never the test click.
    h = hist.history(np.array([u1]), np.array([e.cutoff_ts]), 10)[0]
    assert row[14] not in set(h[h > 0] - 1)
    assert hist.counts(np.array([u1]), np.array([e.cutoff_ts]))[0] == 4


def test_frozen_test_histories_only_hold_train_clicks(enc):
    hist = ClickHistory(enc)
    users, h, clicked = tt.test_queries(enc, hist, 20, max_users=200)
    train_click_ads = set(enc.imp_ad[enc.split_mask("train") & (enc.imp_clk == 1)].tolist())
    rows = h[h > 0] - 1
    assert set(rows.tolist()) <= train_click_ads
    assert len(users) == len(clicked) == len(h)


# ---------------------------------------------------------------------------
# Two tower pieces


def test_streaming_frequency_estimator_orders_by_frequency():
    est = tt.StreamingFrequencyEstimator(n_buckets=1 << 12, alpha=0.1)
    for step in range(500):
        items = [1]
        if step % 10 == 0:
            items.append(2)
        est.update(np.array(items))
    lq = est.log_q(np.array([1, 2]))
    assert lq[0] > lq[1]
    # Gap estimates should be near 1 and near 10.
    assert abs(np.exp(-lq[0]) - 1.0) < 0.5
    assert abs(np.exp(-lq[1]) - 10.0) < 3.0


def test_short_training_run_beats_random(enc):
    cfg = tt.TrainConfig(embed_dim=16, id_dim=16, small_dim=4, hidden=[32], batch_size=256,
                         epochs=3, eval_every_epoch=False, lr=5e-3)
    device = torch.device("cpu")
    model, hist, curve = tt.train(enc, cfg, device, log=lambda *a: None)
    assert len(curve) == 3
    users, h, clicked = tt.test_queries(enc, hist, cfg.history_len, max_users=500)
    score = tt.clicked_vs_random(model, users, h, clicked, enc.n_ads, device)
    assert score > 0.55
    corpus = tt.embed_corpus(model, enc.n_ads, device)
    assert corpus.shape == (enc.n_ads, 16)
    np.testing.assert_allclose(np.linalg.norm(corpus, axis=1), 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Metrics


def test_recall_at_k_hand_built():
    exact = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    approx = np.array([[1, 2, 9, -1], [8, 7, 6, 5]])
    assert recall_at_k(approx, exact, 4) == pytest.approx((2 + 4) / 8)
    assert recall_at_k(approx, exact, 2) == pytest.approx((2 + 0) / 4)
    assert recall_at_k(exact, exact, 3) == 1.0


def test_hit_rate_hand_built():
    retrieved = np.array([[3, 1, 2], [9, 8, 7]])
    clicked = [np.array([1, 5]), np.array([7])]
    out = hit_rate(retrieved, clicked, ks=[1, 2, 3])
    assert out[1] == {"rate": 0.0, "hits": 0, "pairs": 3}
    assert out[2]["hits"] == 1 and out[2]["rate"] == pytest.approx(1 / 3)
    assert out[3]["hits"] == 2 and out[3]["rate"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        hit_rate(retrieved, clicked, ks=[4])


def test_popularity_and_random_baselines():
    clicks = np.array([4, 4, 4, 2, 2, 7, 7, 0])
    top = popularity_baseline(clicks, n_ads=10, k=4)
    assert list(top) == [4, 2, 7, 0]
    top2 = popularity_baseline(np.array([3, 1]), n_ads=5, k=3)
    assert list(top2) == [1, 3, 0]  # ties broken by lower row
    out = popularity_hit_rate(clicks, 10, [np.array([4]), np.array([9])], ks=[1])
    assert out[1]["rate"] == 0.5
    assert random_baseline_rate(50, 1000) == 0.05
    assert random_baseline_rate(5000, 1000) == 1.0


def test_rank_of_splits_ties():
    assert rank_of(0.9, np.array([0.9, 0.5, 0.1])) == 1.0
    assert rank_of(0.5, np.array([0.9, 0.5, 0.1])) == 2.0
    assert rank_of(0.5, np.array([0.5, 0.5, 0.5])) == 2.0


# ---------------------------------------------------------------------------
# FAISS indexes


@pytest.fixture(scope="module")
def vectors():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 32)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    q = rng.normal(size=(100, 32)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return x, q


def test_flat_matches_torch_topk(vectors):
    index_mod = pytest.importorskip("src.retrieval.index")
    x, q = vectors
    idx = index_mod.build_index("flat", x)
    scores, ids = idx.search(q, 10)
    ref = torch.topk(torch.from_numpy(q) @ torch.from_numpy(x).T, 10, dim=1)
    np.testing.assert_array_equal(ids, ref.indices.numpy())
    np.testing.assert_allclose(scores, ref.values.numpy(), rtol=1e-5, atol=1e-5)
    assert idx.memory_bytes() > x.nbytes
    assert idx.name == "flat"


def test_ivf_full_probe_is_exact(vectors):
    pytest.importorskip("faiss")
    from src.retrieval.index import build_index

    x, q = vectors
    exact = build_index("flat", x).search(q, 20)[1]
    ivf = build_index("ivf_flat", x, nlist=32, nprobe=1)
    low = recall_at_k(ivf.search(q, 20)[1], exact, 20)
    ivf.set_search_param("nprobe", 32)
    assert ivf.name == "ivf_flat nlist=32 nprobe=32"
    assert recall_at_k(ivf.search(q, 20)[1], exact, 20) == 1.0
    assert low < 1.0
    assert ivf.memory_bytes() > 0 and ivf.build_seconds >= 0
    with pytest.raises(ValueError):
        ivf.set_search_param("efSearch", 10)


def test_hnsw_recall_at_generous_ef(vectors):
    pytest.importorskip("faiss")
    from src.retrieval.index import build_index

    x, q = vectors
    exact = build_index("flat", x).search(q, 10)[1]
    h = build_index("hnsw", x, M=16, efConstruction=100, efSearch=256)
    assert recall_at_k(h.search(q, 10)[1], exact, 10) >= 0.9
    assert h.memory_bytes() > 0


def test_ivf_pq_builds_and_is_smaller(vectors):
    pytest.importorskip("faiss")
    from src.retrieval.index import build_index

    x, q = vectors
    pq = build_index("ivf_pq", x, nlist=16, m=8, nbits=8, nprobe=16, train_size=4000)
    flat = build_index("flat", x)
    assert pq.memory_bytes() < flat.memory_bytes()
    assert recall_at_k(pq.search(q, 10)[1], flat.search(q, 10)[1], 10) > 0.3


def test_default_sweep_shape_and_time_search(vectors):
    pytest.importorskip("faiss")
    from src.retrieval.index import build_index, default_sweep, time_search

    sweep = default_sweep(846_811)
    kinds = [k for k, _, _ in sweep]
    assert kinds[0] == "flat"
    assert {"ivf_flat", "hnsw", "ivf_pq"} <= set(kinds)
    for kind, build, settings in sweep:
        assert settings
        if kind.startswith("ivf"):
            assert all(s["nprobe"] <= build["nlist"] for s in settings)
    small = default_sweep(4000)
    for kind, build, _ in small:
        if kind.startswith("ivf"):
            assert build["nlist"] <= 4000 // 39

    x, q = vectors
    idx = build_index("hnsw", x, M=16, efSearch=32)
    one = time_search(idx, q, 10, batch_size=1, warmup=5)
    assert one["batch_size"] == 1 and one["queries"] == len(q)
    assert 0 < one["p50_us"] <= one["p99_us"]
    many = time_search(idx, q, 10, batch_size=len(q))
    assert many["qps"] > 0
