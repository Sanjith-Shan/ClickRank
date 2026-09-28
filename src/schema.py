"""Shared data contracts for the ClickRank pipeline.

This module is intentionally dependency light. It defines the column schema of
the Criteo dataset and the two dataclasses that every downstream module agrees
on. Keeping these here avoids circular imports between data, features, models,
and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Criteo Display Advertising Challenge schema.
# 1 label, 13 dense integer features (I1..I13), 26 sparse categorical features (C1..C26).
LABEL_COL: str = "label"
NUM_COLS: List[str] = [f"I{i}" for i in range(1, 14)]
CAT_COLS: List[str] = [f"C{i}" for i in range(1, 27)]
ALL_COLS: List[str] = [LABEL_COL] + NUM_COLS + CAT_COLS


@dataclass
class Dataset:
    """A fully featurized split of the data.

    All arrays share the same first dimension N (number of rows). This is the
    single container passed between the feature pipeline, the trainer, and the
    evaluation code.

    Fields
    ------
    numerical : (N, n_numerical) float32
        Log transformed and standardized dense features plus binary is_missing
        indicators.
    categorical : (N, n_cat) int64
        Hash encoded category indices, one column per sparse field. Each column
        is a valid index into that field's embedding table (0 .. vocab_size - 1).
    cat_freq : (N, n_cat) float32
        Frequency encodings of the same sparse fields, normalized to [0, 1].
        Used by the logistic regression baseline.
    crosses : (N, n_cross) int64
        Hash encoded 2nd order feature crosses, one column per cross.
    label : (N,) float32
        Binary click label in {0.0, 1.0}.
    """

    numerical: np.ndarray
    categorical: np.ndarray
    cat_freq: np.ndarray
    crosses: np.ndarray
    label: np.ndarray

    def __len__(self) -> int:
        return int(self.label.shape[0])


@dataclass
class FeatureMeta:
    """Metadata describing the featurized space.

    Models use this to size their embedding tables and input layers. The deep
    models embed the concatenation of categorical fields and cross fields, so
    the helpers below expose the combined view.
    """

    n_numerical: int
    cat_vocab_sizes: List[int] = field(default_factory=list)
    cross_vocab_sizes: List[int] = field(default_factory=list)

    @property
    def n_cat(self) -> int:
        return len(self.cat_vocab_sizes)

    @property
    def n_cross(self) -> int:
        return len(self.cross_vocab_sizes)

    def embed_vocab_sizes(self) -> List[int]:
        """Vocab sizes for every embedded field, categorical then cross."""
        return list(self.cat_vocab_sizes) + list(self.cross_vocab_sizes)

    @property
    def n_embed_fields(self) -> int:
        return self.n_cat + self.n_cross
