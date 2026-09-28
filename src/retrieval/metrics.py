"""Retrieval metrics: recall against exact search, and hit rate on real clicks.

Two different questions, and the results always carry both.

recall@K asks how much an approximate index loses. It compares the top K an
approximate index returns with the top K of exact search over the same
embeddings. It says nothing about whether those embeddings are any good.

hit rate@K asks whether retrieval gets the right ad in front of the ranker at
all. For every (user, clicked ad) pair on the held out day it checks whether
the clicked ad is in that user's top K. A pair is the unit, so a user with
three clicks counts three times. It is always reported next to the two
baselines a reader would ask about: the K most clicked training ads for
everyone (popularity), and K random ads (chance, which is K / N in
expectation).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def recall_at_k(approx_ids: np.ndarray, exact_ids: np.ndarray, k: int) -> float:
    """Mean over queries of |approx top k and exact top k| / k.

    Both arrays are (Q, >= k) of corpus ids. Padding ids of -1 never count as
    a match. The denominator is k, so an index that returns fewer than k real
    ids is charged for the gap.
    """
    a = np.asarray(approx_ids)[:, :k]
    e = np.asarray(exact_ids)[:, :k]
    if len(a) == 0:
        return float("nan")
    total = 0
    for ra, re in zip(a, e):
        ra = ra[ra >= 0]
        re = re[re >= 0]
        total += len(np.intersect1d(ra, re, assume_unique=False))
    return float(total / (len(a) * k))


def hit_rate(retrieved_ids: np.ndarray, clicked: List[np.ndarray],
             ks: Sequence[int]) -> Dict[int, Dict[str, float]]:
    """Fraction of (query, clicked ad) pairs whose ad is in the query's top k.

    retrieved_ids is (Q, K) in rank order, clicked[i] holds the corpus rows
    query i clicked. Returns k -> {"rate", "hits", "pairs"} for every k up to
    the width of retrieved_ids.
    """
    r = np.asarray(retrieved_ids)
    out: Dict[int, Dict[str, float]] = {}
    pairs = int(sum(len(c) for c in clicked))
    for k in ks:
        if k > r.shape[1]:
            raise ValueError(f"k={k} exceeds the {r.shape[1]} ids retrieved")
        hits = 0
        for row, cl in zip(r[:, :k], clicked):
            if len(cl):
                hits += int(np.isin(cl, row[row >= 0]).sum())
        out[int(k)] = {"rate": hits / pairs if pairs else float("nan"), "hits": hits, "pairs": pairs}
    return out


def popularity_baseline(train_click_ads: np.ndarray, n_ads: int, k: int) -> np.ndarray:
    """The k most clicked corpus rows in training, ties broken by lower row.

    This is the list a system with no personalisation would show everyone.
    """
    counts = np.bincount(np.asarray(train_click_ads, dtype=np.int64), minlength=n_ads)
    # A stable sort on the negated counts keeps lower rows first among ties.
    order = np.argsort(-counts, kind="stable")
    return order[:k]


def popularity_hit_rate(train_click_ads: np.ndarray, n_ads: int, clicked: List[np.ndarray],
                        ks: Sequence[int]) -> Dict[int, Dict[str, float]]:
    """Hit rate of the popularity list, the same list for every query."""
    top = popularity_baseline(train_click_ads, n_ads, max(ks))
    return hit_rate(np.tile(top, (len(clicked), 1)), clicked, ks)


def random_baseline_rate(k: int, n_ads: int) -> float:
    """Expected hit rate of k ads drawn at random from the corpus."""
    return min(1.0, k / n_ads) if n_ads else float("nan")


def rank_of(target_score: float, all_scores: np.ndarray) -> float:
    """1 based rank of a target score within all_scores.

    Ties are split: rank = 1 + (number strictly greater) + (other ties) / 2.
    all_scores is expected to contain the target itself once, which is not
    counted as a tie with itself. The half tie rule is the expected rank under
    random tie breaking, so a model that gives everything the same score
    lands in the middle rather than at the top or the bottom.
    """
    s = np.asarray(all_scores)
    greater = int((s > target_score).sum())
    ties = int((s == target_score).sum()) - 1
    return 1.0 + greater + max(ties, 0) / 2.0


def fast_group_auc(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    """Impression weighted GAUC over every group with both classes, in one pass.

    Same definition as src.evaluation.metrics.group_auc, which loops over the
    groups with a mask each time and is quadratic in practice. This uses the
    rank sum form of AUC inside each group, with average ranks for ties, so a
    few million impressions over a few hundred thousand users take seconds.
    """
    import pandas as pd

    df = pd.DataFrame({"g": np.asarray(groups), "y": np.asarray(y_true, dtype=np.float64),
                       "s": np.asarray(y_pred, dtype=np.float64)})
    df["r"] = df.groupby("g")["s"].rank(method="average")
    agg = df.groupby("g").agg(n=("y", "size"), p=("y", "sum"),
                              rp=("r", lambda r: 0.0))
    agg["rp"] = df[df["y"] > 0].groupby("g")["r"].sum().reindex(agg.index).fillna(0.0)
    agg["neg"] = agg["n"] - agg["p"]
    ok = (agg["p"] > 0) & (agg["neg"] > 0)
    a = agg[ok]
    auc = (a["rp"] - a["p"] * (a["p"] + 1) / 2) / (a["p"] * a["neg"])
    w = a["n"]
    return {"gauc": float((auc * w).sum() / w.sum()) if len(a) else 0.5,
            "groups_scored": int(ok.sum()), "groups_total": int(len(agg)),
            "impressions_in_scored_groups": int(w.sum())}
