"""The feature spec that lets the existing rankers train on the retrieval data.

DeepFM and DCN in src/models take a Dataset (numerical, categorical, cat_freq,
crosses, label) and a FeatureMeta. This module builds those containers from
the encoded Taobao shaped data, two ways that must agree exactly:

rows_dataset       one row per logged impression, for training and for the
                   offline test set.
candidate_dataset  one row per candidate ad for a single request, built by
                   broadcasting the request's user and context across the
                   candidate rows. This is the request path.

The parity property the repo cares about holds here too: for the same user,
ad and time, candidate_dataset produces the same row rows_dataset produced in
frozen mode, and a test pins that.

Categorical fields, in order:
    user fields     enc.user_feat columns: user id, then the profile fields
    ad fields       enc.ad_feat columns: ad id, cate, campaign, customer,
                    brand, price bin
    pid             placement code plus one (0 is out of vocabulary)
    hour            hour of day in Beijing time plus one, from the impression
                    timestamp, or the request hour for candidates
    history         with_history only: L trailing columns, the user's click
                    history as corpus ad rows plus one, most recent first,
                    0 for padding. Only DIN reads these. DeepFM and DCN never
                    see them, so their code runs unchanged.

Numerical features, in order:
    log1p(price), standardised with the mean and std over training impressions
    log1p(prior click count), standardised the same way

There are no crosses and no frequency encodings, so crosses and cat_freq are
(N, 0) arrays. Categorical columns are stored as int32 (4 bytes per field) to
keep a sampled training set in memory. The shared trainer's make_loader copies
them to int64 when it builds its tensors, so at training time a row costs its
int32 storage plus an int64 copy. For the 17 regular Taobao fields that is 68
bytes stored and 136 in the loader, plus 8 bytes of numerical features and 4 of
label. With history (L = 20) add 80 and 160.

Time. Training rows (frozen=False) use each impression's own timestamp for the
history and the prior click count. Test rows (frozen=True) use enc.cutoff_ts,
the end of the training days, so no test day information reaches a feature.

Scoring. score() runs the module directly on tensors under inference_mode,
chunked, with no DataLoader. predict_torch in the trainer wraps the arrays in a
TensorDataset and a DataLoader, which is fine for offline evaluation but adds
per call overhead that would show up in a per request latency figure. The two
give the same probabilities, and a test checks it.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from src.models.base import BaseModel
from src.models.dcn import DCNModel, DCNModule
from src.models.deepfm import DeepFMModel, DeepFMModule
from src.models.din import DIN_CONFIG, DINModel
from src.retrieval.data import BEIJING_OFFSET_S
from src.retrieval.features import ClickHistory, Encoded
from src.schema import Dataset, FeatureMeta
from src.train.config import get_config
from src.train.trainer import get_device, train_torch_model

N_HOURS = 24
# Columns of enc.ad_feat the DIN attention compares: ad id, cate, brand.
DIN_AD_COLS = (0, 1, 4)


def hour_of_day(ts: np.ndarray) -> np.ndarray:
    """Hour of day in Beijing time, 0 to 23."""
    return ((np.asarray(ts, dtype=np.int64) + BEIJING_OFFSET_S) // 3600) % N_HOURS


class RankingSpec:
    """Field layout, vocabulary sizes and numerical scaling for the rankers.

    Built once from the encoded data and the training click history. The
    standardisation statistics are computed on (a sample of) the training
    impressions, with each impression's own prior click count.
    """

    def __init__(self, enc: Encoded, hist: ClickHistory, history_len: int = 20,
                 stats_sample: int = 1_000_000, seed: int = 0):
        self.history_len = int(history_len)
        self.n_user_fields = int(enc.user_feat.shape[1])
        self.n_ad_fields = int(enc.ad_feat.shape[1])
        self.field_names = (
            list(enc.user_field_names)
            + ["ad_id", "cate", "campaign", "customer", "brand", "price_bin"][: self.n_ad_fields]
            + ["pid", "hour"]
        )
        self.cat_vocab_sizes = (
            list(enc.user_vocab_sizes) + list(enc.ad_vocab_sizes) + [enc.n_pid + 1, N_HOURS + 1]
        )
        self.n_regular = len(self.cat_vocab_sizes)
        self.n_ads = enc.n_ads
        self.ad_start = self.n_user_fields
        # Positions of the candidate's ad id, cate and brand among the fields.
        self.din_query_positions = [self.ad_start + c for c in DIN_AD_COLS]

        train_idx = np.flatnonzero(enc.split_mask("train"))
        if len(train_idx) > stats_sample:
            rng = np.random.default_rng(seed)
            train_idx = rng.choice(train_idx, size=stats_sample, replace=False)
        lp = np.log1p(enc.ad_price[enc.imp_ad[train_idx]].astype(np.float64))
        lc = np.log1p(hist.counts(enc.imp_user[train_idx], enc.imp_ts[train_idx]).astype(np.float64))
        self.price_mean, self.price_std = float(lp.mean()), float(lp.std() or 1.0)
        self.count_mean, self.count_std = float(lc.mean()), float(lc.std() or 1.0)
        # Price as a per corpus row feature, standardised once.
        self.ad_price_z = ((np.log1p(enc.ad_price.astype(np.float64)) - self.price_mean)
                           / self.price_std).astype(np.float32)

    @property
    def n_numerical(self) -> int:
        return 2

    def meta(self, with_history: bool = False) -> FeatureMeta:
        """FeatureMeta for the rankers.

        With history the L trailing columns are listed with the corpus size
        plus one as their vocabulary, which only fixes n_cat for the trainer.
        DIN does not embed them through that table. It looks each row up in
        the ad feature buffer and reuses the candidate's field embeddings.
        """
        sizes = list(self.cat_vocab_sizes)
        if with_history:
            sizes += [self.n_ads + 1] * self.history_len
        return FeatureMeta(n_numerical=self.n_numerical, cat_vocab_sizes=sizes, cross_vocab_sizes=[])

    def count_z(self, counts: np.ndarray) -> np.ndarray:
        return ((np.log1p(np.asarray(counts, dtype=np.float64)) - self.count_mean)
                / self.count_std).astype(np.float32)


def _assemble(spec: RankingSpec, user_part, ad_rows, enc: Encoded, pid, hour, counts_z,
              history, n: int, label) -> Dataset:
    width = spec.n_regular + (spec.history_len if history is not None else 0)
    cat = np.empty((n, width), dtype=np.int32)
    u = spec.n_user_fields
    a = spec.n_ad_fields
    cat[:, :u] = user_part
    cat[:, u:u + a] = enc.ad_feat[ad_rows]
    cat[:, u + a] = np.asarray(pid, dtype=np.int64) + 1
    cat[:, u + a + 1] = np.asarray(hour, dtype=np.int64) + 1
    if history is not None:
        cat[:, spec.n_regular:] = history
    num = np.empty((n, 2), dtype=np.float32)
    num[:, 0] = spec.ad_price_z[ad_rows]
    num[:, 1] = counts_z
    return Dataset(
        numerical=num,
        categorical=cat,
        cat_freq=np.zeros((n, 0), dtype=np.float32),
        crosses=np.zeros((n, 0), dtype=np.int32),
        label=np.asarray(label, dtype=np.float32),
    )


def rows_dataset(spec: RankingSpec, enc: Encoded, hist: ClickHistory, idx: np.ndarray,
                 frozen: bool, with_history: bool = False) -> Dataset:
    """One Dataset row per impression index in idx.

    frozen=False uses each impression's own time for the history and the prior
    click count (training). frozen=True uses enc.cutoff_ts (the test day).
    The hour field always comes from the impression's own timestamp, because
    the hour a request arrives is known at request time.
    """
    idx = np.asarray(idx, dtype=np.int64)
    users = enc.imp_user[idx]
    ts = enc.imp_ts[idx]
    t = np.full(len(idx), enc.cutoff_ts, dtype=np.int64) if frozen else ts
    counts = hist.counts(users, t)
    history = hist.history(users, t, spec.history_len) if with_history else None
    return _assemble(
        spec, enc.user_feat[users], enc.imp_ad[idx], enc, enc.imp_pid[idx], hour_of_day(ts),
        spec.count_z(counts), history, len(idx), enc.imp_clk[idx],
    )


def candidate_dataset(spec: RankingSpec, enc: Encoded, user_row: int, history_row: Optional[np.ndarray],
                      prior_count: int, pid: int, hour: int, cand_ad_rows: np.ndarray) -> Dataset:
    """One Dataset row per candidate ad for a single request.

    The user's fields, history and context are broadcast across the candidate
    rows. Pass history_row=None for DeepFM and DCN, or the (L,) history for
    DIN. Labels are zeros. Works for K = 500 and for the whole corpus, though a
    caller scoring the whole corpus should chunk cand_ad_rows to bound memory.
    """
    cand = np.asarray(cand_ad_rows, dtype=np.int64)
    n = len(cand)
    history = None
    if history_row is not None:
        history = np.broadcast_to(np.asarray(history_row, dtype=np.int32)[None, :], (n, spec.history_len))
    return _assemble(
        spec, enc.user_feat[int(user_row)][None, :], cand, enc, pid, hour,
        spec.count_z(np.array([prior_count]))[0], history, n, np.zeros(n, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Training and scoring


def fit_ranker(name: str, train_ds: Dataset, val_ds: Dataset, spec: RankingSpec, enc: Encoded,
               overrides: Optional[Dict] = None, device=None) -> BaseModel:
    """Train deepfm, dcn, din or din-mean with the repo's shared trainer.

    DeepFM and DCN use their defaults from src/train/config.py. DIN uses
    DIN_CONFIG from src/models/din.py. overrides replace any key.
    """
    overrides = dict(overrides or {})
    device = device if device is not None else get_device("auto")
    name = name.lower()
    if name in ("deepfm", "dcn"):
        meta = spec.meta(with_history=False)
        cfg = {**get_config(name), **overrides}
        if name == "deepfm":
            module = DeepFMModule(meta, cfg["embed_dim"], cfg["hidden"], cfg["dropout"])
            model = DeepFMModel()
        else:
            module = DCNModule(meta, cfg["embed_dim"], cfg["cross_layers"], cfg["hidden"], cfg["dropout"])
            model = DCNModel()
        model.module = train_torch_model(module, train_ds, val_ds, meta, cfg, device=device)
        model.meta = meta
        model.embed_dim = cfg["embed_dim"]
        model.device = next(model.module.parameters()).device
        return model
    if name in ("din", "din-mean"):
        meta = spec.meta(with_history=True)
        cfg = {**DIN_CONFIG, **overrides}
        if name == "din-mean":
            cfg["pooling"] = "mean"
        model = DINModel(spec.history_len, spec.din_query_positions, enc.ad_feat, DIN_AD_COLS)
        return model.fit(train_ds, val_ds, meta, cfg, device=device)
    raise ValueError(f"unknown ranker {name!r}")


def _tensors(ds: Dataset, device) -> tuple:
    num = torch.from_numpy(np.ascontiguousarray(ds.numerical, dtype=np.float32)).to(device)
    cat = torch.from_numpy(np.ascontiguousarray(ds.categorical)).to(device).long()
    return num, cat


def score_module(module: torch.nn.Module, ds: Dataset, device=None, batch_size: int = 65536) -> np.ndarray:
    """Click probabilities from a raw module, chunked, with no DataLoader.

    The module must already be in eval mode on device. This is the request
    path: one tensor conversion per chunk and one forward call.
    """
    if device is None:
        device = next(module.parameters()).device
    n = len(ds)
    out = np.empty(n, dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            num = torch.from_numpy(np.ascontiguousarray(ds.numerical[s:e], dtype=np.float32)).to(device)
            cat = torch.from_numpy(np.ascontiguousarray(ds.categorical[s:e])).to(device).long()
            out[s:e] = torch.sigmoid(module(num, cat)).float().cpu().numpy()
    return out


def score(model: BaseModel, ds: Dataset, batch_size: int = 65536) -> np.ndarray:
    """Click probabilities for every row of ds from a fitted ranker."""
    model.module.eval()
    return score_module(model.module, ds, model.device, batch_size)
