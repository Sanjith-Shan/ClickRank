"""Tests for the retrieval ranking spec and the DIN ranker, on a tiny synthetic log.

Everything here runs on data written by scripts/make_synthetic_ids.py into a
temporary directory. No number from these tests is a result.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
import torch

from src.evaluation.metrics import compute_auc
from src.retrieval.data import load
from src.retrieval.features import ClickHistory, encode
from src.retrieval import ranking as R
from src.train.trainer import predict_torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPU = torch.device("cpu")


def _make_module():
    spec = importlib.util.spec_from_file_location(
        "make_synthetic_ids", os.path.join(ROOT, "scripts", "make_synthetic_ids.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("synthetic_ids"))
    _make_module().make(out, n_users=3000, n_ads=800, n_cates=20, n_impressions=120_000, seed=7)
    data = load("synthetic", out)
    enc = encode(data)
    hist = ClickHistory(enc)
    spec = R.RankingSpec(enc, hist, history_len=10)
    return enc, hist, spec


def _split(enc, n_train=60_000, seed=0):
    rng = np.random.default_rng(seed)
    tr = np.flatnonzero(enc.split_mask("train"))
    tr = rng.choice(tr, size=min(n_train, len(tr)), replace=False)
    te = np.flatnonzero(enc.split_mask("test"))
    return tr, te


def test_rows_dataset_shapes_and_dtypes(world):
    enc, hist, spec = world
    idx = np.arange(500)
    ds = R.rows_dataset(spec, enc, hist, idx, frozen=False)
    assert ds.categorical.shape == (500, spec.n_regular)
    assert ds.categorical.dtype == np.int32
    assert ds.numerical.shape == (500, 2) and ds.numerical.dtype == np.float32
    assert ds.crosses.shape == (500, 0) and ds.cat_freq.shape == (500, 0)
    assert ds.label.dtype == np.float32
    assert len(spec.field_names) == spec.n_regular == len(spec.cat_vocab_sizes)
    sizes = np.array(spec.cat_vocab_sizes)
    assert (ds.categorical >= 0).all() and (ds.categorical < sizes[None, :]).all()
    dh = R.rows_dataset(spec, enc, hist, idx, frozen=True, with_history=True)
    assert dh.categorical.shape == (500, spec.n_regular + spec.history_len)
    assert (dh.categorical[:, spec.n_regular:] <= enc.n_ads).all()
    assert spec.meta(True).n_cat == spec.n_regular + spec.history_len


def test_candidate_rows_match_logged_rows(world):
    """The request path builds exactly the row the offline test set holds."""
    enc, hist, spec = world
    te = np.flatnonzero(enc.split_mask("test"))[:200]
    logged = R.rows_dataset(spec, enc, hist, te, frozen=True, with_history=True)
    for j, i in enumerate(te[:50]):
        u = enc.imp_user[i]
        h = hist.history(np.array([u]), np.array([enc.cutoff_ts]), spec.history_len)[0]
        c = int(hist.counts(np.array([u]), np.array([enc.cutoff_ts]))[0])
        cand = np.array([enc.imp_ad[i], (enc.imp_ad[i] + 1) % enc.n_ads])
        ds = R.candidate_dataset(spec, enc, u, h, c, int(enc.imp_pid[i]),
                                 int(R.hour_of_day(enc.imp_ts[i])), cand)
        np.testing.assert_array_equal(ds.categorical[0], logged.categorical[j])
        np.testing.assert_array_equal(ds.numerical[0], logged.numerical[j])
        # And without history for DeepFM and DCN.
        ds2 = R.candidate_dataset(spec, enc, u, None, c, int(enc.imp_pid[i]),
                                  int(R.hour_of_day(enc.imp_ts[i])), cand)
        np.testing.assert_array_equal(ds2.categorical[0], logged.categorical[j, :spec.n_regular])


def test_training_history_excludes_the_click_itself(world):
    enc, hist, spec = world
    clicks = np.flatnonzero(enc.split_mask("train") & (enc.imp_clk == 1))[:300]
    ds = R.rows_dataset(spec, enc, hist, clicks, frozen=False, with_history=True)
    counts = hist.counts(enc.imp_user[clicks], enc.imp_ts[clicks])
    # A user's first click has an empty history.
    first = counts == 0
    assert (ds.categorical[first, spec.n_regular:] == 0).all()


def test_din_forward_all_padding(world):
    enc, hist, spec = world
    from src.models.din import DINModule
    for pooling, norm in (("attention", "none"), ("attention", "softmax"), ("mean", "none")):
        m = DINModule(spec.meta(True), spec.history_len, spec.din_query_positions, enc.ad_feat,
                      R.DIN_AD_COLS, pooling=pooling, attention_norm=norm).eval()
        ds = R.rows_dataset(spec, enc, hist, np.arange(64), frozen=True, with_history=True)
        ds.categorical[:, spec.n_regular:] = 0
        out = R.score_module(m, ds, CPU)
        assert out.shape == (64,) and np.isfinite(out).all()


@pytest.mark.parametrize("name", ["deepfm", "dcn", "din"])
def test_rankers_learn(world, name):
    enc, hist, spec = world
    tr, te = _split(enc)
    wh = name == "din"
    train_ds = R.rows_dataset(spec, enc, hist, tr, frozen=False, with_history=wh)
    test_ds = R.rows_dataset(spec, enc, hist, te, frozen=True, with_history=wh)
    torch.manual_seed(0)
    model = R.fit_ranker(name, train_ds, test_ds, spec, enc,
                         overrides={"epochs": 3, "batch_size": 1024, "embed_dim": 8,
                                    "hidden": [32, 16], "dropout": 0.0, "lr": 3e-3},
                         device=CPU)
    p = R.score(model, test_ds)
    assert compute_auc(test_ds.label, p) > 0.55
    # The fast path gives what the trainer's DataLoader path gives.
    ref = predict_torch(model.module, test_ds, CPU)
    np.testing.assert_allclose(p, ref, atol=1e-6)
