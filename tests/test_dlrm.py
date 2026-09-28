"""DLRM through TorchRec. Skipped where TorchRec is not installed (it needs Linux)."""

import numpy as np
import pytest

torchrec = pytest.importorskip("torchrec")

import torch  # noqa: E402

from src.models.dlrm import DLRMModule, to_kjt  # noqa: E402


class _Meta:
    n_numerical = 13
    n_embed_fields = 4

    def embed_vocab_sizes(self):
        return [10, 20, 30, 40]


def test_kjt_is_key_major_with_one_id_per_field():
    cat = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    kjt = to_kjt(cat, ["f0", "f1", "f2", "f3"])
    assert kjt.values().tolist() == [1, 5, 2, 6, 3, 7, 4, 8]
    assert kjt.lengths().tolist() == [1] * 8


def test_forward_shape_and_backward():
    m = DLRMModule(_Meta(), 8, [16], [16])
    num = torch.randn(5, 13)
    cat = torch.from_numpy(np.random.default_rng(0).integers(0, 10, size=(5, 4)))
    out = m(num, cat)
    assert out.shape == (5,)
    out.sum().backward()
