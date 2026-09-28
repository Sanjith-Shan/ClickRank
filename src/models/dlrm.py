"""DLRM built from TorchRec modules.

Naumov et al., *Deep Learning Recommendation Model for Personalization and
Recommendation Systems*, 2019. Dense features go through a bottom MLP down to the
embedding width, every categorical field is looked up in its own table through a
TorchRec ``EmbeddingBagCollection``, the pairwise dot products of all those vectors
form the interaction layer, and a top MLP turns them into a logit. The model is
``torchrec.models.dlrm.DLRM`` itself, not a reimplementation.

The module keeps the benchmark's ``forward(numerical, cat)`` signature, so the
shared trainer, early stopping, held out rows and metrics are the same ones the
other five architectures use. Inside forward the (B, F) categorical ids become a
``KeyedJaggedTensor`` with one id per field, which is the input TorchRec expects.

TorchRec needs Linux and its fbgemm_gpu kernels, so it is imported lazily and the
model is only available where TorchRec is installed.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.models.base import BaseModel
from src.schema import Dataset, FeatureMeta
from src.train.trainer import predict_torch, train_torch_model


def torchrec_available() -> bool:
    try:
        import torchrec  # noqa: F401
    except Exception:  # noqa: BLE001 any import failure means unavailable
        return False
    return True


def make_tables(vocab_sizes: List[int], embed_dim: int):
    """One EmbeddingBagConfig per categorical field, keyed f0, f1, ..."""
    from torchrec import EmbeddingBagConfig

    return [
        EmbeddingBagConfig(
            name=f"t{i}",
            embedding_dim=embed_dim,
            num_embeddings=int(v),
            feature_names=[f"f{i}"],
        )
        for i, v in enumerate(vocab_sizes)
    ]


def to_kjt(cat: torch.Tensor, keys: List[str]):
    """(B, F) long ids to a KeyedJaggedTensor with exactly one id per field.

    KJT stores values key major, so the ids of field 0 for the whole batch come
    first, then field 1, and so on. Every length is 1.
    """
    from torchrec import KeyedJaggedTensor

    b, f = cat.shape
    return KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=cat.t().reshape(-1).long(),
        lengths=torch.ones(b * f, dtype=torch.int32, device=cat.device),
    )


class DLRMModule(nn.Module):
    """TorchRec DLRM behind the benchmark's forward(numerical, cat) interface."""

    def __init__(self, meta: FeatureMeta, embed_dim: int, bottom: List[int], top: List[int]):
        super().__init__()
        from torchrec import EmbeddingBagCollection
        from torchrec.models.dlrm import DLRM

        self.keys = [f"f{i}" for i in range(meta.n_embed_fields)]
        ebc = EmbeddingBagCollection(tables=make_tables(meta.embed_vocab_sizes(), embed_dim))
        self.dlrm = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=meta.n_numerical,
            dense_arch_layer_sizes=list(bottom) + [embed_dim],
            over_arch_layer_sizes=list(top) + [1],
        )

    def forward(self, numerical: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        return self.dlrm(numerical, to_kjt(cat, self.keys)).squeeze(-1)


class DLRMModel(BaseModel):
    """BaseModel wrapper that trains the TorchRec DLRM through the shared trainer."""

    def __init__(self) -> None:
        self.module = None
        self.device = None

    def fit(self, train: Dataset, val: Dataset, meta: FeatureMeta, config: dict) -> "DLRMModel":
        if not torchrec_available():
            raise RuntimeError("DLRM needs TorchRec, which needs Linux and fbgemm_gpu")
        self.module = DLRMModule(meta, config["embed_dim"], config["bottom"], config["top"])
        self.module = train_torch_model(self.module, train, val, meta, config)
        self.device = next(self.module.parameters()).device
        return self

    def predict_proba(self, data: Dataset) -> np.ndarray:
        return predict_torch(self.module, data, self.device)

    def get_name(self) -> str:
        return "DLRM"
