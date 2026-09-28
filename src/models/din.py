"""Deep Interest Network style ranker with target attention over click history.

Zhou et al., Deep Interest Network for Click-Through Rate Prediction, KDD 2018.
The idea is that a user's interest is not one fixed vector. Which past clicks
matter depends on the ad being scored, so the history is pooled with weights
that the candidate ad itself decides. A user who clicked shoes and a phone case
looks like a shoe buyer when a shoe is scored and like a phone owner when a
charger is scored.

How the module reads its input. forward(numerical, cat) takes the same two
tensors every ranker in this repo takes, so the shared trainer and
predict_torch work unchanged. cat has n_regular + L columns:

- columns 0 .. n_regular - 1 are ordinary categorical fields, embedded through
  one shared EmbeddingLayer with per field offsets, exactly as DeepFM does.
- the last L columns are the user's click history, as corpus ad rows plus one,
  most recent first, 0 for padding.

Each history item is turned into the same (ad id, category, brand) codes the
candidate carries, through a buffer holding the corpus ad feature table, and
looked up in the same embedding rows the candidate's own fields use. The query
is the candidate's (ad id, category, brand) embedding. So history item and
candidate live in one space and the attention compares like with like.

Attention unit. An MLP over [h, q, h - q, h * q] gives one scalar per history
item. Padding gets weight zero. Following the paper there is no softmax by
default: the weights are not normalised, so the pooled vector keeps a sense of
how much relevant history there is, not only its direction. attention_norm
"softmax" is available for comparison. pooling "mean" turns attention off and
averages the history, which is the ablation that asks whether target attention
beats a plain mean.

The pooled interest vector, every field embedding flattened, and the numerical
features go into an MLP and a final logit. The paper's Dice activation and
mini batch aware regularisation are not reproduced. ReLU and weight decay are
used instead, and DESIGN.md says so.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.models.base import BaseModel, MLP, EmbeddingLayer
from src.schema import Dataset, FeatureMeta
from src.train.trainer import get_device, predict_torch, train_torch_model

DIN_CONFIG: Dict = {
    "lr": 1e-3,
    "batch_size": 4096,
    "max_epochs": 20,
    "epochs": 15,
    "patience": 3,
    "weight_decay": 1e-5,
    "embed_dim": 16,
    "hidden": [256, 128, 64],
    "attention_hidden": [64, 32],
    "dropout": 0.3,
    "pooling": "attention",  # or "mean"
    "attention_norm": "none",  # or "softmax"
}


class DINModule(nn.Module):
    """Target attention over history plus a deep tower. Returns raw logits."""

    def __init__(
        self,
        meta: FeatureMeta,
        n_history: int,
        query_positions: Sequence[int],
        ad_feat: np.ndarray,
        ad_feat_cols: Sequence[int],
        embed_dim: int = 16,
        hidden: Sequence[int] = (256, 128, 64),
        attention_hidden: Sequence[int] = (64, 32),
        dropout: float = 0.3,
        pooling: str = "attention",
        attention_norm: str = "none",
    ):
        super().__init__()
        if pooling not in ("attention", "mean"):
            raise ValueError(pooling)
        if attention_norm not in ("none", "softmax"):
            raise ValueError(attention_norm)
        self.n_history = int(n_history)
        self.n_regular = meta.n_embed_fields - self.n_history
        self.embed_dim = embed_dim
        self.pooling = pooling
        self.attention_norm = attention_norm
        vocab = meta.embed_vocab_sizes()[: self.n_regular]
        self.embedding = EmbeddingLayer(vocab, embed_dim)
        self.query_positions = list(query_positions)
        self.register_buffer("query_pos", torch.tensor(self.query_positions, dtype=torch.long))
        self.register_buffer(
            "hist_feat", torch.as_tensor(np.asarray(ad_feat)[:, list(ad_feat_cols)], dtype=torch.long)
        )
        q_dim = len(self.query_positions) * embed_dim
        att_layers = []
        prev = 4 * q_dim
        for h in attention_hidden:
            att_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        att_layers.append(nn.Linear(prev, 1))
        self.attention = nn.Sequential(*att_layers)
        deep_in = self.n_regular * embed_dim + q_dim + meta.n_numerical
        self.mlp = MLP(deep_in, list(hidden), dropout=dropout)
        self.head = nn.Linear(self.mlp.out_dim, 1)

    def history_embeddings(self, hist: torch.Tensor) -> tuple:
        """(B, L) rows + 1 -> ((B, L, q_dim) embeddings, (B, L) float mask)."""
        mask = (hist > 0).to(torch.float32)
        rows = (hist - 1).clamp(min=0)
        codes = self.hist_feat[rows]  # (B, L, 3)
        idx = codes + self.embedding.offsets[self.query_pos].view(1, 1, -1)
        e = self.embedding.embedding(idx)  # (B, L, 3, d)
        return e.reshape(e.size(0), e.size(1), -1), mask

    def forward(self, numerical: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        reg = cat[:, : self.n_regular]
        hist = cat[:, self.n_regular:]
        emb, _ = self.embedding(reg)  # (B, F, d)
        q = emb[:, self.query_pos, :].reshape(emb.size(0), -1)  # (B, q_dim)

        if self.n_history > 0:
            h, mask = self.history_embeddings(hist)
            if self.pooling == "mean":
                w = mask / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            else:
                qe = q.unsqueeze(1).expand_as(h)
                a = self.attention(torch.cat([h, qe, h - qe, h * qe], dim=-1)).squeeze(-1)
                if self.attention_norm == "softmax":
                    a = a.masked_fill(mask == 0, float("-inf"))
                    w = torch.softmax(a, dim=1)
                    w = torch.nan_to_num(w, nan=0.0)  # rows with no history at all
                else:
                    w = a * mask
            interest = (w.unsqueeze(-1) * h).sum(dim=1)
        else:
            interest = torch.zeros_like(q)

        flat = emb.reshape(emb.size(0), -1)
        x = torch.cat([flat, interest, numerical], dim=1)
        return self.head(self.mlp(x)).squeeze(-1)


class DINModel(BaseModel):
    """BaseModel wrapper that trains a DINModule through the shared trainer.

    The history layout and the ad feature table are fixed at construction,
    because they describe the dataset rather than a hyperparameter.
    """

    def __init__(self, n_history: int, query_positions: Sequence[int], ad_feat: np.ndarray,
                 ad_feat_cols: Sequence[int]):
        self.n_history = n_history
        self.query_positions = list(query_positions)
        self.ad_feat = ad_feat
        self.ad_feat_cols = list(ad_feat_cols)
        self.module = None
        self.meta = None
        self.device = None
        self.name = "DIN"

    def build(self, meta: FeatureMeta, config: dict) -> DINModule:
        cfg = {**DIN_CONFIG, **config}
        if cfg["pooling"] == "mean":
            self.name = "DIN-mean"
        return DINModule(
            meta, self.n_history, self.query_positions, self.ad_feat, self.ad_feat_cols,
            embed_dim=cfg["embed_dim"], hidden=cfg["hidden"],
            attention_hidden=cfg["attention_hidden"], dropout=cfg["dropout"],
            pooling=cfg["pooling"], attention_norm=cfg["attention_norm"],
        )

    def fit(self, train: Dataset, val: Dataset, meta: FeatureMeta, config: dict,
            device=None) -> "DINModel":
        self.meta = meta
        cfg = {**DIN_CONFIG, **config}
        self.module = self.build(meta, cfg)
        device = device if device is not None else get_device("auto")
        self.module = train_torch_model(self.module, train, val, meta, cfg, device=device)
        self.device = next(self.module.parameters()).device
        return self

    def predict_proba(self, data: Dataset) -> np.ndarray:
        return predict_torch(self.module, data, self.device)

    def get_name(self) -> str:
        return self.name
