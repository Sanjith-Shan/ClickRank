"""Turn raw ids into dense indices, and build click histories without leakage.

Every vocabulary that could carry label information is fitted on the training
days only. An ad that never appeared in training keeps its own row in the
corpus but its id embedding is the shared out of vocabulary slot (index 0), so
the ad tower can still place it from its category, campaign, customer, brand
and price. The same holds for users: a user first seen on the test day gets the
out of vocabulary id and is represented by profile and history alone.

Click history. For a training click at time t the history is the user's last L
clicks strictly before t, which is what a real time system would have known.
For every test day query the history is frozen at the end of the last training
day, so no test day information reaches either the retrieval or the ranking
stage. That is conservative, and it is stated wherever a number depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.retrieval.data import AD_FIELDS, RetrievalData, user_fields

AD_FEATURE_NAMES = ["ad_id"] + AD_FIELDS + ["price_bin"]
N_PRICE_BINS = 20


def _vocab(values: np.ndarray) -> pd.Index:
    """Index of the distinct values. Position p maps to code p + 1, 0 is OOV."""
    return pd.Index(pd.unique(values))


def _encode(index: pd.Index, values: np.ndarray) -> np.ndarray:
    return (index.get_indexer(values) + 1).astype(np.int64)


@dataclass
class Encoded:
    """The dataset as integer arrays, ready for the towers and the rankers.

    ad_feat    (n_ads, 6) int64. Row r is corpus ad r: id, cate, campaign,
               customer, brand, price bin. Each column is its own vocabulary
               with 0 as the out of vocabulary code.
    user_feat  (n_user_rows, 1 + n_profile) int64. Row r is a user seen in the
               log. Column 0 is the user id code, the rest are profile fields.
    imp_*      one entry per impression, aligned with data.impressions after
               rows whose ad is missing from the corpus are dropped.
    """

    ad_feat: np.ndarray
    ad_vocab_sizes: List[int]
    ad_raw_ids: np.ndarray
    ad_price: np.ndarray
    user_feat: np.ndarray
    user_vocab_sizes: List[int]
    user_field_names: List[str]
    user_raw_ids: np.ndarray
    imp_user: np.ndarray
    imp_ad: np.ndarray
    imp_ts: np.ndarray
    imp_day: np.ndarray
    imp_pid: np.ndarray
    imp_clk: np.ndarray
    n_pid: int
    train_days: tuple
    test_day: int
    cutoff_ts: int
    notes: Dict[str, object] = field(default_factory=dict)

    @property
    def n_ads(self) -> int:
        return int(self.ad_feat.shape[0])

    @property
    def n_user_rows(self) -> int:
        return int(self.user_feat.shape[0])

    def split_mask(self, which: str) -> np.ndarray:
        if which == "train":
            return (self.imp_day >= self.train_days[0]) & (self.imp_day <= self.train_days[1])
        if which == "test":
            return self.imp_day == self.test_day
        raise ValueError(which)


def encode(data: RetrievalData) -> Encoded:
    """Fit every vocabulary on the training days and encode the whole dataset."""
    imp = data.impressions
    ads = data.ads.reset_index(drop=True)
    train_mask = ((imp["day"] >= data.train_days[0]) & (imp["day"] <= data.train_days[1])).to_numpy()

    # Corpus rows. An impression whose ad is not in the ad table cannot be
    # retrieved or ranked from features, so it is dropped and counted.
    ad_index = pd.Index(ads["ad"].to_numpy())
    imp_ad = ad_index.get_indexer(imp["ad"].to_numpy())
    known = imp_ad >= 0
    notes = {"impressions_with_unknown_ad_dropped": int((~known).sum())}

    train_ads = np.unique(imp_ad[known & train_mask])
    in_train = np.zeros(len(ads), dtype=bool)
    in_train[train_ads] = True
    notes["corpus_ads_seen_in_train"] = int(in_train.sum())

    cols = []
    sizes = []
    id_vocab = _vocab(ads["ad"].to_numpy()[in_train])
    cols.append(_encode(id_vocab, ads["ad"].to_numpy()))
    sizes.append(len(id_vocab) + 1)
    for f in AD_FIELDS:
        voc = _vocab(ads[f].to_numpy()[in_train])
        cols.append(_encode(voc, ads[f].to_numpy()))
        sizes.append(len(voc) + 1)
    # Price is binned on log price quantiles over the training ads. No label is
    # involved, but fitting on training ads keeps the rule uniform.
    logp = np.log1p(np.nan_to_num(ads["price"].to_numpy(np.float64), nan=0.0))
    edges = np.unique(np.quantile(logp[in_train] if in_train.any() else logp,
                                  np.linspace(0, 1, N_PRICE_BINS + 1)[1:-1]))
    cols.append(np.searchsorted(edges, logp).astype(np.int64) + 1)
    sizes.append(len(edges) + 2)
    ad_feat = np.stack(cols, axis=1)

    # User rows cover every user in the log. The id code is only non zero for
    # users with a training impression. Profile fields are static attributes
    # with no label in them, so their vocabularies use the whole profile table.
    log_users = pd.unique(imp["user"].to_numpy())
    user_rows = pd.Index(log_users)
    imp_user = user_rows.get_indexer(imp["user"].to_numpy())
    train_users = pd.unique(imp["user"].to_numpy()[train_mask])
    uid_vocab = pd.Index(train_users)
    ucols = [_encode(uid_vocab, log_users)]
    usizes = [len(uid_vocab) + 1]
    pfields = user_fields(data)
    prof = data.users.set_index("user").reindex(log_users)
    for f in pfields:
        vals = data.users[f].to_numpy()
        voc = _vocab(vals)
        ucols.append(_encode(voc, prof[f].to_numpy()))  # NaN (no profile) -> 0
        usizes.append(len(voc) + 1)
    user_feat = np.stack(ucols, axis=1)
    notes["log_users_with_profile"] = int(prof[pfields[0]].notna().sum()) if pfields else 0

    ts = imp["ts"].to_numpy(np.int64)
    cutoff = int(ts[train_mask].max()) + 1 if train_mask.any() else int(ts.min())

    k = known
    return Encoded(
        ad_feat=ad_feat,
        ad_vocab_sizes=sizes,
        ad_raw_ids=ads["ad"].to_numpy(),
        ad_price=np.nan_to_num(ads["price"].to_numpy(np.float32), nan=0.0),
        user_feat=user_feat,
        user_vocab_sizes=usizes,
        user_field_names=["user_id"] + pfields,
        user_raw_ids=np.asarray(log_users),
        imp_user=imp_user[k].astype(np.int64),
        imp_ad=imp_ad[k].astype(np.int64),
        imp_ts=ts[k],
        imp_day=imp["day"].to_numpy()[k],
        imp_pid=imp["pid"].to_numpy(np.int64)[k],
        imp_clk=imp["clk"].to_numpy(np.int8)[k],
        n_pid=int(imp["pid"].max()) + 1,
        train_days=data.train_days,
        test_day=data.test_day,
        cutoff_ts=cutoff,
        notes=notes,
    )


class ClickHistory:
    """Every training click, sorted by (user, time), queryable as of any time.

    history(users, times, L) returns an (N, L) array of corpus ad rows plus one,
    most recent first, with 0 as padding. Only clicks strictly before the query
    time count, so a training click never sees itself.
    """

    def __init__(self, enc: Encoded, max_ts: Optional[int] = None):
        m = enc.split_mask("train") & (enc.imp_clk == 1)
        if max_ts is not None:
            m &= enc.imp_ts < max_ts
        u = enc.imp_user[m]
        t = enc.imp_ts[m] - int(enc.imp_ts.min())
        a = enc.imp_ad[m]
        order = np.lexsort((t, u))
        self.users = u[order]
        self.keys = self.users * (1 << 32) + t[order]
        self.ads = a[order]
        self.t0 = int(enc.imp_ts.min())

    def __len__(self) -> int:
        return int(len(self.ads))

    def history(self, users: np.ndarray, times: np.ndarray, length: int) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        rel = np.asarray(times, dtype=np.int64) - self.t0
        rel = np.clip(rel, 0, (1 << 32) - 1)
        end = np.searchsorted(self.keys, users * (1 << 32) + rel, side="left")
        start = np.searchsorted(self.keys, users * (1 << 32), side="left")
        pos = end[:, None] - np.arange(1, length + 1)[None, :]
        valid = pos >= start[:, None]
        pos = np.where(valid, pos, 0)
        out = np.where(valid, self.ads[pos] + 1 if len(self.ads) else 0, 0)
        return out.astype(np.int64)

    def counts(self, users: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Number of prior clicks, uncapped, for a history length feature."""
        users = np.asarray(users, dtype=np.int64)
        rel = np.clip(np.asarray(times, dtype=np.int64) - self.t0, 0, (1 << 32) - 1)
        end = np.searchsorted(self.keys, users * (1 << 32) + rel, side="left")
        start = np.searchsorted(self.keys, users * (1 << 32), side="left")
        return (end - start).astype(np.int64)
