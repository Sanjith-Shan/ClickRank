"""fast_group_auc must agree with the reference group_auc it replaces for speed."""

import numpy as np

from src.evaluation.metrics import group_auc
from src.retrieval.metrics import fast_group_auc


def test_fast_gauc_matches_reference_with_ties_and_single_class_groups():
    rng = np.random.default_rng(3)
    g = rng.integers(0, 60, 4000)
    y = (rng.random(4000) < 0.15).astype(float)
    y[g == 0] = 0.0  # a group with one class is skipped by both
    s = np.round(rng.random(4000) + 0.4 * y, 1)  # coarse rounding forces ties
    fast = fast_group_auc(y, s, g)
    assert abs(fast["gauc"] - group_auc(y, s, g)) < 1e-12
    assert fast["groups_scored"] < fast["groups_total"]
