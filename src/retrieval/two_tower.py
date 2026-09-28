"""The two tower retrieval model: a user tower, an ad tower, one dot product.

This is the standard candidate generation design (Covington et al., RecSys
2016; Yi et al., RecSys 2019). It keeps the scoring rule of the text DSSM in
src/relevance/two_tower.py, an L2 normalised embedding per side and a
temperature scaled cosine, and replaces letter trigram hashing with id
embeddings.

Ad tower. Embeddings of the ad id, category, campaign, customer, brand and
binned price, concatenated, then a small MLP, then L2 normalisation. It only
reads the ad's own attributes, so every corpus ad can be embedded once offline.

User tower. Embeddings of the user id and profile fields, plus the mean of the
user's recent clicked ads, each represented by the same id, category and brand
embeddings the ad tower uses. Sharing those tables is what lets a user who
clicked three shoes land near other shoes. Then a small MLP and normalisation.

Training. Clicked impressions are the positives. Each batch of B clicks gives
a B by B score matrix and every other ad in the batch is a negative (in batch
sampled softmax). Popular ads show up as in batch negatives far more often
than rare ones, which pushes their scores down for everyone. Yi et al. correct
for that by subtracting log q(ad) from each logit, where q is the probability
the ad appears in a batch, estimated online with their streaming frequency
estimator. Both the correction and the estimator are implemented here, and the
correction can be switched off to measure what it buys.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.retrieval.features import ClickHistory, Encoded

# Which ad columns the history pooling shares with the ad tower.
HISTORY_AD_COLS = (0, 1, 4)  # ad id, category, brand


def _emb_dim(vocab: int, big: int, small: int) -> int:
    return big if vocab > 1000 else small


class StreamingFrequencyEstimator:
    """Yi et al. 2019, Algorithm 2: estimate how often each item is sampled.

    For each hashed item it keeps the step it was last seen (A) and a moving
    estimate of the gap between sightings (B). An item seen every B steps
    appears in a batch with probability about 1 / B. The estimate is an
    exponential moving average with rate alpha, so it tracks a stream whose
    popularity drifts.
    """

    def __init__(self, n_buckets: int = 1 << 21, alpha: float = 0.01, init_gap: float = 100.0):
        self.n = n_buckets
        self.alpha = alpha
        self.A = np.zeros(n_buckets, dtype=np.int64)
        self.B = np.full(n_buckets, init_gap, dtype=np.float64)
        self.step = 0

    def _h(self, items: np.ndarray) -> np.ndarray:
        # A fixed multiplicative hash. Items are already dense integers so a
        # modulo alone would do, but hashing keeps the estimator generic.
        return (items.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) >> np.uint64(40)).astype(np.int64) % self.n

    def update(self, items: np.ndarray) -> None:
        self.step += 1
        h = np.unique(self._h(items))
        seen = self.A[h] > 0
        gap = self.step - self.A[h]
        self.B[h] = np.where(seen, (1 - self.alpha) * self.B[h] + self.alpha * gap, self.B[h])
        self.A[h] = self.step

    def log_q(self, items: np.ndarray) -> np.ndarray:
        return -np.log(self.B[self._h(items)])


class _Tower(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class TwoTowerRetriever(nn.Module):
    """User tower and ad tower over integer features held as buffers.

    The feature tables live on the module, so a forward pass takes only row
    indices: user rows and history rows for the user side, corpus rows for the
    ad side. That keeps the request path to one gather per table.
    """

    def __init__(
        self,
        ad_vocab_sizes: List[int],
        user_vocab_sizes: List[int],
        ad_feat: np.ndarray,
        user_feat: np.ndarray,
        embed_dim: int = 64,
        id_dim: int = 32,
        small_dim: int = 8,
        hidden: Optional[List[int]] = None,
        use_history: bool = True,
    ):
        super().__init__()
        hidden = hidden if hidden is not None else [256]
        self.use_history = use_history
        self.register_buffer("ad_feat", torch.as_tensor(ad_feat, dtype=torch.long))
        self.register_buffer("user_feat", torch.as_tensor(user_feat, dtype=torch.long))

        self.ad_emb = nn.ModuleList(
            nn.Embedding(v, _emb_dim(v, id_dim, small_dim)) for v in ad_vocab_sizes
        )
        self.user_emb = nn.ModuleList(
            nn.Embedding(v, _emb_dim(v, id_dim, small_dim)) for v in user_vocab_sizes
        )
        ad_in = sum(e.embedding_dim for e in self.ad_emb)
        hist_in = sum(self.ad_emb[c].embedding_dim for c in HISTORY_AD_COLS) if use_history else 0
        user_in = sum(e.embedding_dim for e in self.user_emb) + hist_in
        self.ad_tower = _Tower(ad_in, hidden, embed_dim)
        self.user_tower = _Tower(user_in, hidden, embed_dim)
        for e in list(self.ad_emb) + list(self.user_emb):
            nn.init.normal_(e.weight, std=0.05)

    # -- ad side ---------------------------------------------------------
    def ad_input(self, ad_rows: torch.Tensor) -> torch.Tensor:
        f = self.ad_feat[ad_rows]
        return torch.cat([emb(f[:, i]) for i, emb in enumerate(self.ad_emb)], dim=-1)

    def encode_ads(self, ad_rows: torch.Tensor) -> torch.Tensor:
        return self.ad_tower(self.ad_input(ad_rows))

    # -- user side -------------------------------------------------------
    def history_vector(self, hist: torch.Tensor) -> torch.Tensor:
        """Mean of the shared embeddings of the clicked ads. hist is rows + 1."""
        mask = (hist > 0).float()
        rows = (hist - 1).clamp(min=0)
        f = self.ad_feat[rows]  # (B, L, 6)
        parts = [self.ad_emb[c](f[..., c]) for c in HISTORY_AD_COLS]
        e = torch.cat(parts, dim=-1) * mask.unsqueeze(-1)
        return e.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)

    def encode_users(self, user_rows: torch.Tensor, hist: Optional[torch.Tensor]) -> torch.Tensor:
        f = self.user_feat[user_rows]
        parts = [emb(f[:, i]) for i, emb in enumerate(self.user_emb)]
        if self.use_history:
            parts.append(self.history_vector(hist))
        return self.user_tower(torch.cat(parts, dim=-1))


@dataclass
class TrainConfig:
    embed_dim: int = 64
    id_dim: int = 32
    small_dim: int = 8
    hidden: List[int] = field(default_factory=lambda: [256])
    history_len: int = 20
    use_history: bool = True
    temperature: float = 0.05
    logq_correction: bool = True
    batch_size: int = 4096
    lr: float = 2e-3
    weight_decay: float = 0.0
    epochs: int = 4
    seed: int = 42
    eval_every_epoch: bool = True
    eval_users: int = 5000
    eval_k: int = 100


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def training_pairs(enc: Encoded, hist: ClickHistory, history_len: int):
    """(user rows, ad rows, histories) for every training click, in time order."""
    m = enc.split_mask("train") & (enc.imp_clk == 1)
    order = np.argsort(enc.imp_ts[m], kind="stable")
    users = enc.imp_user[m][order]
    ads = enc.imp_ad[m][order]
    times = enc.imp_ts[m][order]
    h = hist.history(users, times, history_len)
    return users, ads, h


def test_queries(enc: Encoded, hist: ClickHistory, history_len: int, max_users: Optional[int] = None,
                 seed: int = 0):
    """One query per test day user with at least one click.

    Returns user rows, their frozen histories, and for each user the set of
    corpus rows they clicked on the test day. The history is as of the end of
    the training days for every query.
    """
    m = enc.split_mask("test") & (enc.imp_clk == 1)
    u = enc.imp_user[m]
    a = enc.imp_ad[m]
    users = np.unique(u)
    if max_users is not None and len(users) > max_users:
        rng = np.random.default_rng(seed)
        users = np.sort(rng.choice(users, size=max_users, replace=False))
    keep = np.isin(u, users)
    u, a = u[keep], a[keep]
    order = np.argsort(u, kind="stable")
    u, a = u[order], a[order]
    bounds = np.searchsorted(u, users, side="left"), np.searchsorted(u, users, side="right")
    clicked = [np.unique(a[s:e]) for s, e in zip(*bounds)]
    h = hist.history(users, np.full(len(users), enc.cutoff_ts), history_len)
    return users, h, clicked


@torch.no_grad()
def embed_corpus(model: TwoTowerRetriever, n_ads: int, device, batch: int = 65536) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, n_ads, batch):
        rows = torch.arange(s, min(s + batch, n_ads), device=device)
        out.append(model.encode_ads(rows).float().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


@torch.no_grad()
def embed_users(model: TwoTowerRetriever, users: np.ndarray, hist: np.ndarray, device,
                batch: int = 65536) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(users), batch):
        u = torch.as_tensor(users[s:s + batch], device=device)
        h = torch.as_tensor(hist[s:s + batch], device=device)
        out.append(model.encode_users(u, h).float().cpu().numpy())
    if not out:
        return np.zeros((0, model.ad_tower.net[-1].out_features), dtype=np.float32)
    return np.concatenate(out, axis=0).astype(np.float32)


def exact_hit_rate(user_vecs: np.ndarray, corpus: np.ndarray, clicked: List[np.ndarray],
                   ks=(50, 100, 500)) -> Dict[int, float]:
    """Brute force hit rate, used for the training curve before FAISS exists.

    A hit is a (user, clicked ad) pair whose ad is in that user's top K. The
    rate is over pairs, so a user with three clicks counts three times.
    """
    kmax = max(ks)
    c = torch.from_numpy(corpus)
    hits = {k: 0 for k in ks}
    total = 0
    for s in range(0, len(user_vecs), 1024):
        q = torch.from_numpy(user_vecs[s:s + 1024])
        top = torch.topk(q @ c.T, k=min(kmax, c.shape[0]), dim=1).indices.numpy()
        for i, row in enumerate(top):
            cl = clicked[s + i]
            total += len(cl)
            for k in ks:
                hits[k] += int(np.isin(cl, row[:k]).sum())
    return {k: hits[k] / max(total, 1) for k in ks}


def train(
    enc: Encoded,
    cfg: TrainConfig,
    device: torch.device,
    log=print,
) -> tuple:
    """Train the two tower model. Returns (model, history object, curve)."""
    _set_seed(cfg.seed)
    hist = ClickHistory(enc)
    users, ads, h = training_pairs(enc, hist, cfg.history_len)
    n = len(users)
    log(f"training clicks {n:,}, corpus ads {enc.n_ads:,}, user rows {enc.n_user_rows:,}")

    model = TwoTowerRetriever(
        enc.ad_vocab_sizes, enc.user_vocab_sizes, enc.ad_feat, enc.user_feat,
        embed_dim=cfg.embed_dim, id_dim=cfg.id_dim, small_dim=cfg.small_dim,
        hidden=cfg.hidden, use_history=cfg.use_history,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    est = StreamingFrequencyEstimator()

    eval_set = None
    if cfg.eval_every_epoch:
        eval_set = test_queries(enc, hist, cfg.history_len, max_users=cfg.eval_users, seed=cfg.seed)

    curve = []
    rng = np.random.default_rng(cfg.seed)
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        # Shuffle within the epoch. The estimator sees the same stream order.
        order = rng.permutation(n)
        t0 = time.time()
        tot, cnt = 0.0, 0
        for s in range(0, n - cfg.batch_size + 1, cfg.batch_size):
            idx = order[s:s + cfg.batch_size]
            ub = torch.as_tensor(users[idx], device=device)
            ab_np = ads[idx]
            ab = torch.as_tensor(ab_np, device=device)
            hb = torch.as_tensor(h[idx], device=device)

            est.update(ab_np)
            uv = model.encode_users(ub, hb)
            av = model.encode_ads(ab)
            logits = (uv @ av.T) / cfg.temperature
            if cfg.logq_correction:
                lq = torch.as_tensor(est.log_q(ab_np), dtype=logits.dtype, device=device)
                logits = logits - lq.unsqueeze(0)
            # The same ad can be the positive for several rows in one batch.
            # Those copies are not negatives for each other, so mask them out.
            same = ab.unsqueeze(0) == ab.unsqueeze(1)
            eye = torch.eye(len(idx), dtype=torch.bool, device=device)
            logits = logits.masked_fill(same & ~eye, float("-inf"))
            target = torch.arange(len(idx), device=device)
            loss = F.cross_entropy(logits, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
            cnt += len(idx)
            step += 1
        row = {"epoch": epoch, "train_loss": tot / max(cnt, 1), "seconds": time.time() - t0, "steps": step}
        if eval_set is not None:
            eu, eh, ecl = eval_set
            corpus = embed_corpus(model, enc.n_ads, device)
            uvecs = embed_users(model, eu, eh, device)
            hr = exact_hit_rate(uvecs, corpus, ecl, ks=(cfg.eval_k,))
            row[f"test_hit_rate@{cfg.eval_k}"] = hr[cfg.eval_k]
        log(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        curve.append(row)
    model.eval()
    return model, hist, curve


@torch.no_grad()
def clicked_vs_random(model: TwoTowerRetriever, users: np.ndarray, hist: np.ndarray,
                      clicked: List[np.ndarray], n_ads: int, device, seed: int = 0) -> float:
    """Fraction of (user, clicked ad) pairs scored above a random corpus ad.

    The M1 sanity check. 0.5 means the towers learned nothing.
    """
    rng = np.random.default_rng(seed)
    uv = embed_users(model, users, hist, device)
    pu = np.repeat(np.arange(len(users)), [len(c) for c in clicked])
    pa = np.concatenate(clicked) if clicked else np.zeros(0, dtype=np.int64)
    ra = rng.integers(0, n_ads, size=len(pa))
    ua = torch.from_numpy(uv[pu])
    ca = model.encode_ads(torch.as_tensor(pa, device=device)).float().cpu()
    rr = model.encode_ads(torch.as_tensor(ra, device=device)).float().cpu()
    s_pos = (ua * ca).sum(-1)
    s_neg = (ua * rr).sum(-1)
    return float(((s_pos > s_neg).float() + 0.5 * (s_pos == s_neg).float()).mean())
