"""Ranking and calibration metrics for the ClickRank benchmark.

This module collects the metrics that matter for click through rate models in
real ad systems. AUC measures ranking quality. Logloss and normalized entropy
measure probability quality. Relative improvement compares a model against the
naive base rate predictor. Group AUC (GAUC) measures within group ranking
quality, which is the production relevant view because an ad auction ranks ads
inside a single user request rather than across the whole population.

All functions accept array like inputs and return plain Python floats so they
can be serialized into reports and JSON directly.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

# Probability clip bounds. Logloss is undefined at exactly 0 or 1 so we pull
# predictions inside the open interval before scoring.
_EPS = 1e-7


def _as_float_array(values: Sequence) -> np.ndarray:
    """Coerce array like input into a 1D float64 numpy array."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr


def compute_auc(y_true: Sequence, y_pred: Sequence) -> float:
    """Area under the ROC curve via sklearn roc_auc_score.

    Returns 0.5 when only one class is present because AUC is undefined for a
    single class and 0.5 is the neutral random ranking value.
    """
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_pred))


def compute_logloss(y_true: Sequence, y_pred: Sequence) -> float:
    """Binary cross entropy with predictions clipped to [1e-7, 1 - 1e-7]."""
    y_true = _as_float_array(y_true)
    y_pred = np.clip(_as_float_array(y_pred), _EPS, 1.0 - _EPS)
    return float(log_loss(y_true, y_pred, labels=[0, 1]))


def _base_rate_entropy(base_rate: float) -> float:
    """Binary entropy of the base rate in nats matching logloss units.

    Returns 0.0 for a degenerate base rate of 0 or 1 so callers can guard a
    division by zero.
    """
    p = float(np.clip(base_rate, _EPS, 1.0 - _EPS))
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def normalized_entropy(y_true: Sequence, y_pred: Sequence) -> float:
    """Normalized entropy NE = logloss / entropy(base_rate).

    The base rate is the mean of y_true. The denominator is the logloss you
    would get by always predicting that base rate. A value below 1 means the
    model is more informative than the base rate. This is the standard metric
    from the Facebook ad CTR literature because it is insensitive to the overall
    click rate of the traffic.
    """
    y_true = _as_float_array(y_true)
    base_rate = float(np.mean(y_true)) if len(y_true) else 0.0
    denom = _base_rate_entropy(base_rate)
    if denom == 0.0:
        return 0.0
    return compute_logloss(y_true, y_pred) / denom


def relative_improvement(
    y_true: Sequence,
    y_pred: Sequence,
    base_pred: Optional[Sequence] = None,
) -> float:
    """Relative improvement RelaImpr = (CE_base - CE_model) / CE_base.

    CE_base is the cross entropy of a constant predictor. When base_pred is not
    given the constant is the base rate of y_true, which is the best possible
    constant. A positive value means the model beats that constant predictor.
    Returns 0.0 when the base cross entropy is degenerate.
    """
    y_true = _as_float_array(y_true)
    n = len(y_true)
    if n == 0:
        return 0.0
    if base_pred is None:
        base_rate = float(np.mean(y_true))
        base_arr = np.full(n, base_rate, dtype=np.float64)
    else:
        base_arr = _as_float_array(base_pred)

    ce_base = compute_logloss(y_true, base_arr)
    ce_model = compute_logloss(y_true, y_pred)
    if ce_base == 0.0:
        return 0.0
    return float((ce_base - ce_model) / ce_base)


def group_auc(y_true: Sequence, y_pred: Sequence, groups: Sequence) -> float:
    """Group AUC (GAUC), an impression weighted average of per group AUC.

    GAUC is the production relevant ranking metric in ad systems. A real auction
    ranks candidate ads inside one user request or one query, so what matters is
    whether the model orders ads correctly within each such group rather than
    across the whole dataset. In practice the group key is a user id or a query
    id and each group is one set of competing impressions.

    The metric computes AUC inside every group, weights each group by its number
    of impressions, and averages. Groups that contain only a single class are
    skipped because AUC is undefined there and they carry no ranking signal.
    Returns 0.5 when no group is scorable.
    """
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    groups = np.asarray(groups).reshape(-1)

    total_weight = 0.0
    weighted_auc = 0.0
    for g in np.unique(groups):
        mask = groups == g
        gt = y_true[mask]
        gp = y_pred[mask]
        if len(np.unique(gt)) < 2:
            continue
        weight = float(mask.sum())
        weighted_auc += weight * float(roc_auc_score(gt, gp))
        total_weight += weight

    if total_weight == 0.0:
        return 0.5
    return weighted_auc / total_weight


def compute_all(
    y_true: Sequence,
    y_pred: Sequence,
    groups: Optional[Sequence] = None,
) -> Dict[str, float]:
    """Bundle the core metrics into a single dict.

    Always returns auc, logloss, ne, and relaimpr. The gauc key is added only
    when groups are provided since GAUC needs a grouping key to be meaningful.
    """
    out: Dict[str, float] = {
        "auc": compute_auc(y_true, y_pred),
        "logloss": compute_logloss(y_true, y_pred),
        "ne": normalized_entropy(y_true, y_pred),
        "relaimpr": relative_improvement(y_true, y_pred),
    }
    if groups is not None:
        out["gauc"] = group_auc(y_true, y_pred, groups)
    return out
