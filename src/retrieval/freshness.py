"""Model freshness on the retrieval data, after He et al. 2014, Section 5.

He et al., "Practical Lessons from Predicting Clicks on Ads at Facebook"
(ADKDD 2014) trained a model on one day and evaluated it on each later day, and
found that normalised entropy got worse the older the training data was. This
module runs the same experiment with the DeepFM ranker from
src/retrieval/ranking.py on the eight day Taobao log, and adds the question a
team actually has to answer, which is how to keep a model fresh.

Two experiments, both scored on every impression of the test day (day 8).

Staleness curve. For each training day d, train a DeepFM from scratch on a
fixed number of rows sampled from the window of `window_days` days ending at d
(one day by default), and score it on day 8. Every model sees the same number
of rows for the same number of epochs, so only the recency of the rows changes.
NE is reported relative to the freshest model (d = 7) at each staleness 8 - d.

Update strategy. Train a base model on days 1 to k. Then compare
    none     the base model, never updated
    warm     the base model fine tuned for one pass on each new day k+1 .. 7
             in turn, on that day's rows only
    full     a model trained from scratch on days 1 to 7
and report NE, AUC and GAUC for each, with the training wall clock and the rows
processed. warm_gap_recovered_ne = (NE_none - NE_warm) / (NE_none - NE_full).
compute_ratio_rows is the rows the one full retrain processed over the rows
every warm update processed together, and compute_ratio_seconds the same for
training wall clock. Both are conservative for warm, which is charged for all
of its updates and full for only its last. compute_ratio_rows_daily_schedule
charges full for a retrain on every new day, as a daily schedule would.

Rows. A seeded sampler draws, for every (seed, day), a disjoint validation part
and a training part of `train_rows` rows. Every experiment in one seed uses the
same rows for the same day. Multi day training sets are the union of their
days' parts, so they sample each day at the same rate.

Features. All models share one encoding (vocabularies fitted on days 1 to 7) and
the test rows' features are frozen at the end of day 7 for every model. Only the
model's parameters age, not the serving features, which is how a feature store
behaves when only the model push is delayed.

Unseen ids. The vocabulary covers days 1 to 7, so a model trained on day 3 has
embedding rows for ids that first appear on day 5. Those rows never receive a
gradient and would keep their random initial values. After every training step
this module zeroes the embedding and first order rows of every (field, code)
the model has not trained on, which makes an unseen id contribute nothing, the
same rule the two tower retriever applies. A warm started model keeps a running
record of the codes it has seen.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.evaluation.metrics import compute_auc, compute_logloss, normalized_entropy
from src.retrieval.features import ClickHistory, Encoded
from src.retrieval.metrics import fast_group_auc
from src.retrieval.ranking import RankingSpec, fit_ranker, rows_dataset, score_module
from src.schema import Dataset
from src.train.config import get_config
from src.train.trainer import set_seed, train_torch_model

MODEL = "deepfm"


@dataclass
class FreshnessConfig:
    """What one freshness run does. Defaults are the real run's settings."""

    train_rows: int = 1_500_000      # training rows per day
    val_rows: int = 200_000          # validation rows per training set, drawn from its own days
    epochs: int = 2                  # epochs for every model trained from scratch
    finetune_epochs: int = 1         # passes over each new day for the warm start
    finetune_lr: Optional[float] = None  # None keeps the ranker's lr
    window_days: int = 1             # days in each staleness training window
    base_days: int = 4               # k, the base model for the update comparison
    test_rows: Optional[int] = None  # None scores every test day impression
    stale_days: Optional[Sequence[int]] = None  # None means window_days .. last train day
    overrides: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rows


class DaySampler:
    """Seeded, disjoint train and validation rows for each training day.

    For a given seed, day d always yields the same rows, so the staleness model
    for day d, the warm start update on day d and the full retrain all train on
    identical day d rows. train_rows is capped at the smallest training day
    (less its validation part) so every day gives the same count.
    """

    def __init__(self, enc: Encoded, seed: int, train_rows: int, val_rows: int):
        self.enc = enc
        self.seed = int(seed)
        self.days = list(range(int(enc.train_days[0]), int(enc.train_days[1]) + 1))
        self._pool = {d: np.flatnonzero(enc.imp_day == d) for d in self.days}
        smallest = min(len(p) for p in self._pool.values())
        self.val_per_day = int(min(val_rows, smallest // 5))
        self.train_rows = int(min(train_rows, smallest - self.val_per_day))
        self.val_rows = int(val_rows)
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def day(self, d: int) -> Tuple[np.ndarray, np.ndarray]:
        """(train_idx, val_idx) for day d, sorted, disjoint."""
        if d not in self._cache:
            pool = self._pool[d]
            perm = np.random.default_rng([self.seed, int(d)]).permutation(len(pool))
            va = pool[perm[: self.val_per_day]]
            tr = pool[perm[self.val_per_day: self.val_per_day + self.train_rows]]
            self._cache[d] = (np.sort(tr), np.sort(va))
        return self._cache[d]

    def window(self, days: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Union of the days' training parts, and val_rows drawn from their val parts."""
        days = sorted(int(d) for d in days)
        tr = np.concatenate([self.day(d)[0] for d in days])
        va = np.concatenate([self.day(d)[1] for d in days])
        if len(va) > self.val_rows:
            rng = np.random.default_rng([self.seed, 999, *days])
            va = rng.choice(va, size=self.val_rows, replace=False)
        return np.sort(tr), np.sort(va)


# ---------------------------------------------------------------------------
# Unseen id handling


def seen_codes(spec: RankingSpec, ds: Dataset) -> np.ndarray:
    """Boolean mask over the flat embedding table, True for every code present in ds."""
    sizes = np.asarray(spec.cat_vocab_sizes, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    mask = np.zeros(int(sizes.sum()), dtype=bool)
    cat = ds.categorical[:, : spec.n_regular].astype(np.int64)
    mask[(cat + offsets[None, :]).ravel()] = True
    return mask


def zero_unseen(module: torch.nn.Module, seen: np.ndarray) -> int:
    """Zero the embedding and first order rows of every code not in seen.

    Returns how many rows were zeroed. Works on the DeepFM and DCN modules,
    whose EmbeddingLayer holds one flat table with per field offsets.
    """
    emb = module.embedding
    keep = torch.as_tensor(seen, device=emb.embedding.weight.device)
    if keep.numel() != emb.embedding.weight.shape[0]:
        raise ValueError("seen mask does not match the embedding table")
    with torch.no_grad():
        emb.embedding.weight[~keep] = 0.0
        emb.linear.weight[~keep] = 0.0
    return int((~keep).sum())


# ---------------------------------------------------------------------------
# Training and evaluation


@dataclass
class Trained:
    """A module plus what it cost and which codes it has trained on."""

    module: torch.nn.Module
    seen: np.ndarray
    train_seconds: float = 0.0
    build_seconds: float = 0.0
    rows_processed: int = 0          # training rows times passes, summed over every step
    val_rows_scored: int = 0
    steps: List[Dict] = field(default_factory=list)


def _cfg(cfg: FreshnessConfig, epochs: int, lr: Optional[float] = None) -> Dict:
    # patience = epochs so every epoch runs and rows_processed is exact. The
    # trainer still restores the best epoch by validation logloss.
    out = {**cfg.overrides, "epochs": int(epochs), "patience": int(epochs)}
    if lr is not None:
        out["lr"] = float(lr)
    return out


def train_scratch(spec: RankingSpec, enc: Encoded, hist: ClickHistory, tr: np.ndarray, va: np.ndarray,
                  cfg: FreshnessConfig, device, seed: int, label: str = "") -> Trained:
    """A fresh DeepFM through fit_ranker, then unseen codes zeroed."""
    t0 = time.time()
    train_ds = rows_dataset(spec, enc, hist, tr, frozen=False)
    val_ds = rows_dataset(spec, enc, hist, va, frozen=False)
    build = time.time() - t0
    set_seed(seed)
    t1 = time.time()
    model = fit_ranker(MODEL, train_ds, val_ds, spec, enc, overrides=_cfg(cfg, cfg.epochs), device=device)
    secs = time.time() - t1
    seen = seen_codes(spec, train_ds)
    zero_unseen(model.module, seen)
    rows = int(len(tr)) * int(cfg.epochs)
    return Trained(module=model.module, seen=seen, train_seconds=secs, build_seconds=build,
                   rows_processed=rows, val_rows_scored=int(len(va)) * int(cfg.epochs),
                   steps=[{"step": label or "scratch", "train_rows": int(len(tr)), "passes": int(cfg.epochs),
                           "train_seconds": secs}])


def finetune(t: Trained, spec: RankingSpec, enc: Encoded, hist: ClickHistory, tr: np.ndarray, va: np.ndarray,
             cfg: FreshnessConfig, device, seed: int, label: str = "") -> Trained:
    """Continue training t.module on new rows for cfg.finetune_epochs passes, in place.

    A new Adam optimiser is created, as the shared trainer always does. Codes
    first seen in these rows start from zero, since they were zeroed before.
    """
    t0 = time.time()
    train_ds = rows_dataset(spec, enc, hist, tr, frozen=False)
    val_ds = rows_dataset(spec, enc, hist, va, frozen=False)
    t.build_seconds += time.time() - t0
    lr = cfg.finetune_lr if cfg.finetune_lr is not None else get_config(MODEL)["lr"]
    set_seed(seed)
    t1 = time.time()
    meta = spec.meta(with_history=False)
    t.module = train_torch_model(t.module, train_ds, val_ds, meta,
                                 {**get_config(MODEL), **_cfg(cfg, cfg.finetune_epochs, lr)}, device=device)
    secs = time.time() - t1
    t.seen = t.seen | seen_codes(spec, train_ds)
    zero_unseen(t.module, t.seen)
    t.train_seconds += secs
    t.rows_processed += int(len(tr)) * int(cfg.finetune_epochs)
    t.val_rows_scored += int(len(va)) * int(cfg.finetune_epochs)
    t.steps.append({"step": label or "finetune", "train_rows": int(len(tr)),
                    "passes": int(cfg.finetune_epochs), "train_seconds": secs})
    return t


@dataclass
class TestSet:
    ds: Dataset
    y: np.ndarray
    groups: np.ndarray
    n: int


def build_test(spec: RankingSpec, enc: Encoded, hist: ClickHistory, test_rows: Optional[int],
               seed: int = 0) -> TestSet:
    """Every test day impression (or a seeded sample), features frozen at the cutoff."""
    idx = np.flatnonzero(enc.split_mask("test"))
    if test_rows is not None and test_rows < len(idx):
        idx = np.sort(np.random.default_rng([seed, 8]).choice(idx, size=test_rows, replace=False))
    ds = rows_dataset(spec, enc, hist, idx, frozen=True)
    return TestSet(ds=ds, y=enc.imp_clk[idx].astype(np.float32), groups=enc.imp_user[idx], n=int(len(idx)))


def evaluate(module: torch.nn.Module, test: TestSet, device) -> Dict:
    module.eval()
    t0 = time.time()
    p = score_module(module, test.ds, device)
    secs = time.time() - t0
    g = fast_group_auc(test.y, p, test.groups)
    return {
        "auc": compute_auc(test.y, p),
        "logloss": compute_logloss(test.y, p),
        "ne": normalized_entropy(test.y, p),
        "gauc_by_user": g["gauc"],
        "gauc_groups_scored": g["groups_scored"],
        "mean_pred": float(p.mean()),
        "test_ctr": float(test.y.mean()),
        "test_rows": test.n,
        "test_score_seconds": secs,
    }


# ---------------------------------------------------------------------------
# The two experiments


def staleness_curve(spec: RankingSpec, enc: Encoded, hist: ClickHistory, sampler: DaySampler,
                    test: TestSet, cfg: FreshnessConfig, device, seed: int,
                    emit: Callable[[Dict], None] = lambda r: None) -> List[Dict]:
    """One model per training day d, each on the window of days ending at d."""
    first, last = sampler.days[0], sampler.days[-1]
    days = list(cfg.stale_days) if cfg.stale_days else list(range(first + cfg.window_days - 1, last + 1))
    rows = []
    for d in days:
        window = list(range(max(first, d - cfg.window_days + 1), d + 1))
        tr, va = sampler.window(window)
        t = train_scratch(spec, enc, hist, tr, va, cfg, device, seed, label=f"days {window[0]}-{d}")
        m = evaluate(t.module, test, device)
        row = {"experiment": "staleness", "seed": seed, "train_end_day": d, "window": window,
               "staleness_days": int(enc.test_day) - d, **m,
               "train_rows": int(len(tr)), "val_rows": int(len(va)), "epochs": cfg.epochs,
               "rows_processed": t.rows_processed, "train_seconds": t.train_seconds,
               "feature_build_seconds": t.build_seconds}
        rows.append(row)
        print(f"  staleness seed {seed} d={d}: NE {m['ne']:.4f} AUC {m['auc']:.4f} "
              f"GAUC {m['gauc_by_user']:.4f} train {t.train_seconds:.0f}s")
        del t
    fresh = max(rows, key=lambda r: r["train_end_day"])
    for r in rows:
        r["ne_rel_to_freshest_pct"] = 100.0 * (r["ne"] / fresh["ne"] - 1.0)
        r["auc_minus_freshest"] = r["auc"] - fresh["auc"]
        r["freshest_train_end_day"] = fresh["train_end_day"]
        emit(r)
    return rows


def update_strategies(spec: RankingSpec, enc: Encoded, hist: ClickHistory, sampler: DaySampler,
                      test: TestSet, cfg: FreshnessConfig, device, seed: int,
                      emit: Callable[[Dict], None] = lambda r: None) -> List[Dict]:
    """none, warm and full, all scored on the test day, with their compute."""
    first, last = sampler.days[0], sampler.days[-1]
    k = int(cfg.base_days)
    if not first <= k < last:
        raise ValueError(f"base_days must be in [{first}, {last - 1}], got {k}")
    base_days = list(range(first, k + 1))
    new_days = list(range(k + 1, last + 1))

    tr, va = sampler.window(base_days)
    base = train_scratch(spec, enc, hist, tr, va, cfg, device, seed, label=f"base days {first}-{k}")
    base_cost = {"train_seconds": base.train_seconds, "rows_processed": base.rows_processed}
    none_m = evaluate(base.module, test, device)

    # warm continues from the base in place, so score none before this.
    warm = base
    warm.train_seconds, warm.rows_processed, warm.val_rows_scored, warm.build_seconds = 0.0, 0, 0, 0.0
    warm.steps = []
    for d in new_days:
        dtr, dva = sampler.day(d)
        finetune(warm, spec, enc, hist, dtr, dva, cfg, device, seed, label=f"day {d}")
    warm_m = evaluate(warm.module, test, device)
    warm_steps = warm.steps
    warm_cost = {"train_seconds": warm.train_seconds, "rows_processed": warm.rows_processed}
    del warm, base

    ftr, fva = sampler.window(sampler.days)
    full = train_scratch(spec, enc, hist, ftr, fva, cfg, device, seed, label=f"full days {first}-{last}")
    full_m = evaluate(full.module, test, device)
    full_cost = {"train_seconds": full.train_seconds, "rows_processed": full.rows_processed,
                 "train_rows": int(len(ftr))}
    del full

    # What a daily schedule from day k+1 to the last day would process. Warm
    # does one pass on one day per update. Full retrains on all days so far,
    # epochs times, every day. Only the final full retrain was run, so the
    # schedule figure for full is rows, computed, not measured seconds.
    per_day = sampler.train_rows
    full_schedule_rows = sum(per_day * (d - first + 1) * cfg.epochs for d in new_days)

    common = {"experiment": "update", "seed": seed, "base_days": base_days, "new_days": new_days,
              "train_rows_per_day": per_day, "epochs_scratch": cfg.epochs,
              "finetune_epochs": cfg.finetune_epochs}
    out = []
    for name, m, cost, extra in (
        ("none", none_m, {"train_seconds": 0.0, "rows_processed": 0}, {"base_cost": base_cost}),
        ("warm", warm_m, warm_cost, {"steps": warm_steps, "base_cost": base_cost}),
        ("full", full_m, full_cost, {"schedule_rows_processed": full_schedule_rows}),
    ):
        out.append({**common, "strategy": name, **m, **cost, **extra})
        print(f"  update seed {seed} {name}: NE {m['ne']:.4f} AUC {m['auc']:.4f} "
              f"GAUC {m['gauc_by_user']:.4f} train {cost['train_seconds']:.0f}s rows {cost['rows_processed']:,}")
    summ = update_summary(out)
    for r in out:
        r.update(summ)
        emit(r)
    return out


def update_summary(rows: List[Dict]) -> Dict:
    """Gap recovered and compute ratios from the none, warm and full rows of one seed."""
    by = {r["strategy"]: r for r in rows}
    none, warm, full = by["none"], by["warm"], by["full"]
    gap_ne = none["ne"] - full["ne"]
    gap_auc = full["auc"] - none["auc"]
    return {
        "freshness_gap_ne": gap_ne,
        "freshness_gap_auc": gap_auc,
        "warm_gap_recovered_ne": (none["ne"] - warm["ne"]) / gap_ne if gap_ne != 0 else float("nan"),
        "warm_gap_recovered_auc": (warm["auc"] - none["auc"]) / gap_auc if gap_auc != 0 else float("nan"),
        "compute_ratio_rows": full["rows_processed"] / max(warm["rows_processed"], 1),
        "compute_ratio_seconds": full["train_seconds"] / max(warm["train_seconds"], 1e-9),
        "compute_ratio_rows_daily_schedule": full["schedule_rows_processed"] / max(warm["rows_processed"], 1),
    }


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": int(len(a))}


def summarise(stale_rows: List[Dict], update_rows: List[Dict]) -> List[Dict]:
    """Mean and std across seeds of the headline figures."""
    out = []
    if stale_rows:
        for d in sorted({r["train_end_day"] for r in stale_rows}):
            rs = [r for r in stale_rows if r["train_end_day"] == d]
            out.append({"experiment": "staleness_summary", "train_end_day": d,
                        "staleness_days": rs[0]["staleness_days"], "seeds": [r["seed"] for r in rs],
                        **{m: _mean_std([r[m] for r in rs])
                           for m in ("ne", "auc", "gauc_by_user", "ne_rel_to_freshest_pct")}})
    if update_rows:
        seeds = sorted({r["seed"] for r in update_rows})
        s = {"experiment": "update_summary", "seeds": seeds}
        for name in ("none", "warm", "full"):
            rs = [r for r in update_rows if r["strategy"] == name]
            s[name] = {m: _mean_std([r[m] for r in rs])
                       for m in ("ne", "auc", "gauc_by_user", "train_seconds", "rows_processed")}
        firsts = [next(r for r in update_rows if r["seed"] == sd) for sd in seeds]
        for m in ("warm_gap_recovered_ne", "warm_gap_recovered_auc", "compute_ratio_rows",
                  "compute_ratio_seconds", "compute_ratio_rows_daily_schedule",
                  "freshness_gap_ne"):
            s[m] = _mean_std([r[m] for r in firsts])
        out.append(s)
    return out
