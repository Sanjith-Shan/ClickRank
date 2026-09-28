"""Tests for the freshness study, on the synthetic id data.

The data is data/synthetic_ids, written by scripts/make_synthetic_ids.py and
generated here if it is missing. It has no drift over time, so nothing here
checks that fresher is better. These tests check the protocol: the rows each
model trains on, the unseen id rule, the compute accounting and the result rows.
No number from these tests is a result.
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest
import torch

from src.retrieval import freshness as F
from src.retrieval.data import load
from src.retrieval.features import ClickHistory, encode
from src.retrieval.ranking import RankingSpec, rows_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "synthetic_ids")
CPU = torch.device("cpu")
TINY = dict(train_rows=3_000, val_rows=1_000, epochs=1, test_rows=5_000, overrides={"batch_size": 512})


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def world():
    if not os.path.exists(os.path.join(DATA, "raw_sample.csv")):
        _load_script("make_synthetic_ids").make(DATA)
    data = load("synthetic", DATA)
    assert data.synthetic
    enc = encode(data)
    hist = ClickHistory(enc)
    spec = RankingSpec(enc, hist)
    return enc, hist, spec


def test_sampler_rows_are_disjoint_equal_and_seeded(world):
    enc, _, _ = world
    s = F.DaySampler(enc, seed=0, train_rows=4_000, val_rows=500)
    sizes = set()
    for d in s.days:
        tr, va = s.day(d)
        assert len(np.intersect1d(tr, va)) == 0
        assert (enc.imp_day[tr] == d).all() and (enc.imp_day[va] == d).all()
        sizes.add(len(tr))
    assert sizes == {4_000}
    assert np.array_equal(s.day(3)[0], F.DaySampler(enc, 0, 4_000, 500).day(3)[0])
    assert not np.array_equal(s.day(3)[0], F.DaySampler(enc, 1, 4_000, 500).day(3)[0])
    tr, va = s.window([2, 3, 4])
    assert len(tr) == 12_000 and len(va) == 500
    assert np.array_equal(np.sort(np.concatenate([s.day(d)[0] for d in (2, 3, 4)])), tr)
    # A request larger than the smallest day is capped so every day still matches.
    big = F.DaySampler(enc, 0, 10**9, 500)
    assert big.train_rows == min(int((enc.imp_day == d).sum()) for d in big.days) - big.val_per_day


def test_zero_unseen_clears_only_untrained_codes(world):
    enc, hist, spec = world
    from src.models.deepfm import DeepFMModule

    meta = spec.meta()
    m = DeepFMModule(meta, 8, [16], 0.0)
    ds = rows_dataset(spec, enc, hist, np.flatnonzero(enc.imp_day == 1)[:2_000], frozen=False)
    seen = F.seen_codes(spec, ds)
    assert seen.shape[0] == m.embedding.embedding.weight.shape[0]
    with torch.no_grad():
        m.embedding.linear.weight.fill_(1.0)
    before = m.embedding.embedding.weight.detach().clone()
    n = F.zero_unseen(m, seen)
    w = m.embedding.embedding.weight.detach()
    assert n == int((~seen).sum()) > 0
    assert (w[torch.as_tensor(~seen)] == 0).all()
    assert (m.embedding.linear.weight.detach()[torch.as_tensor(~seen)] == 0).all()
    assert torch.equal(w[torch.as_tensor(seen)], before[torch.as_tensor(seen)])


def test_staleness_curve_rows(world):
    enc, hist, spec = world
    cfg = F.FreshnessConfig(**TINY, stale_days=[5, 7])
    sampler = F.DaySampler(enc, 0, cfg.train_rows, cfg.val_rows)
    test = F.build_test(spec, enc, hist, cfg.test_rows)
    got = []
    rows = F.staleness_curve(spec, enc, hist, sampler, test, cfg, CPU, 0, emit=got.append)
    assert [r["train_end_day"] for r in rows] == [5, 7] and len(got) == 2
    assert [r["staleness_days"] for r in rows] == [3, 1]
    assert {r["train_rows"] for r in rows} == {3_000}
    assert rows[1]["ne_rel_to_freshest_pct"] == 0.0
    for r in rows:
        assert r["window"] == [r["train_end_day"]]
        assert r["rows_processed"] == 3_000 * cfg.epochs
        for k in ("auc", "ne", "gauc_by_user", "logloss", "train_seconds"):
            assert np.isfinite(r[k])


def test_update_strategies_accounting_and_warm_start(world, monkeypatch):
    enc, hist, spec = world
    cfg = F.FreshnessConfig(**TINY, base_days=5)
    sampler = F.DaySampler(enc, 0, cfg.train_rows, cfg.val_rows)
    test = F.build_test(spec, enc, hist, cfg.test_rows)

    # Record which rows every step trains on, and check warm continues one module.
    calls = []
    real_ft = F.finetune

    def spy(t, *a, **kw):
        calls.append((id(t.module), a[3].copy()))
        return real_ft(t, *a, **kw)

    monkeypatch.setattr(F, "finetune", spy)
    rows = F.update_strategies(spec, enc, hist, sampler, test, cfg, CPU, 0)
    by = {r["strategy"]: r for r in rows}
    assert set(by) == {"none", "warm", "full"}
    r = sampler.train_rows
    assert by["none"]["rows_processed"] == 0
    assert by["warm"]["rows_processed"] == 2 * r * cfg.finetune_epochs
    assert by["full"]["rows_processed"] == 7 * r * cfg.epochs
    assert [s["step"] for s in by["warm"]["steps"]] == ["day 6", "day 7"]
    assert by["warm"]["base_cost"]["rows_processed"] == 5 * r * cfg.epochs
    assert by["full"]["schedule_rows_processed"] == (6 + 7) * r * cfg.epochs
    # Each warm step is on that new day's rows only, and on the same module.
    assert len(calls) == 2 and calls[0][0] == calls[1][0]
    assert np.array_equal(calls[0][1], sampler.day(6)[0])
    assert np.array_equal(calls[1][1], sampler.day(7)[0])
    # none is scored before the warm start mutates the base model.
    assert by["none"]["ne"] != by["warm"]["ne"]
    s = F.update_summary(rows)
    assert s["compute_ratio_rows"] == pytest.approx(7 * cfg.epochs / 2)
    gap = by["none"]["ne"] - by["full"]["ne"]
    assert s["warm_gap_recovered_ne"] == pytest.approx((by["none"]["ne"] - by["warm"]["ne"]) / gap)


def test_finetune_trains_codes_first_seen_on_the_new_day(world):
    enc, hist, spec = world
    cfg = F.FreshnessConfig(**TINY)
    sampler = F.DaySampler(enc, 0, cfg.train_rows, cfg.val_rows)
    tr, va = sampler.day(1)
    t = F.train_scratch(spec, enc, hist, tr, va, cfg, CPU, 0)
    seen_before = t.seen.copy()
    tr2, va2 = sampler.day(2)
    F.finetune(t, spec, enc, hist, tr2, va2, cfg, CPU, 0)
    new = t.seen & ~seen_before
    assert new.any()
    w = t.module.embedding.embedding.weight.detach().cpu()
    assert (w[torch.as_tensor(new)].abs().sum(dim=1) > 0).all()
    assert (w[torch.as_tensor(~t.seen)] == 0).all()
    assert t.rows_processed == len(tr) * cfg.epochs + len(tr2) * cfg.finetune_epochs


def test_cli_quick_writes_labelled_rows(world, tmp_path):
    mod = _load_script("run_freshness")
    mod.main(["--source", "synthetic", "--data", DATA, "--results", str(tmp_path), "--quick",
              "--train-rows", "3000", "--val-rows", "1000", "--test-rows", "5000", "--device", "cpu"])
    path = tmp_path / "synthetic" / "retrieval" / "freshness.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    kinds = [r["experiment"] for r in rows]
    assert kinds.count("staleness") == 7 and kinds.count("update") == 3
    assert "staleness_summary" in kinds and "update_summary" in kinds
    for r in rows:
        assert r["dataset"].startswith("SYNTHETIC")
        assert "machine" in r and "load" in r
        assert r["quick"] is True and r["model"] == "deepfm"
    assert not (tmp_path / "retrieval").exists()
