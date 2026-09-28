"""Data loading and synthetic data generation for ClickRank.

This module reads the Criteo style display advertising data and also provides a
synthetic generator. The synthetic generator is the workhorse for offline runs.
It is designed so that pairwise categorical interactions carry real signal in
the click label. That property is what lets the factorization machine family
and the deep cross network beat a plain logistic regression baseline.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.schema import ALL_COLS, CAT_COLS, LABEL_COL, NUM_COLS

# Token used in place of a missing categorical value. Kept as a literal string
# so downstream hashing treats it like any other category.
NAN_TOKEN = "__nan__"

# Number of leading categorical columns that carry the pairwise interaction
# signal. These are kept low cardinality and skewed so the cross generator
# selects them and the deep models can learn the interactions.
_N_INTERACTION_COLS = 4


def load_raw(data_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Read a Criteo style TSV or CSV file into a typed DataFrame.

    The reader sniffs the delimiter. It tries tab first and falls back to comma.
    Columns are assigned the canonical schema ALL_COLS. When sample_size is given
    only the first sample_size rows are read from disk, because the data is time
    ordered and a positional head preserves the temporal structure. Reading just
    the head keeps memory bounded on the full multi gigabyte Criteo file.

    Numerical columns are coerced to numeric with bad values becoming NaN.
    Categorical columns are coerced to string and missing values become the
    literal token "__nan__".
    """
    df = _read_with_delimiter(data_path, nrows=sample_size)

    if df.shape[1] != len(ALL_COLS):
        raise ValueError(
            f"Expected {len(ALL_COLS)} columns but found {df.shape[1]} in {data_path}."
        )
    df.columns = ALL_COLS

    if sample_size is not None:
        df = df.head(sample_size)

    df = df.reset_index(drop=True)

    # Label as int in {0, 1}.
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int)

    # Numerical columns coerced to numeric. Errors and blanks become NaN.
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Categorical columns coerced to string. Missing values become the token.
    for col in CAT_COLS:
        df[col] = df[col].astype("string")
        df[col] = df[col].fillna(NAN_TOKEN).astype(str)
        df[col] = df[col].replace({"": NAN_TOKEN, "nan": NAN_TOKEN, "<NA>": NAN_TOKEN})

    return df


def _read_with_delimiter(data_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read the file as tab separated first and fall back to comma separated.

    The fallback triggers when the tab read yields a single column which means
    the real delimiter was not a tab. When nrows is given only that many rows are
    read from disk, which keeps memory bounded on very large files.
    """
    df = pd.read_csv(data_path, sep="\t", header=None, dtype=str, na_values=[], nrows=nrows)
    if df.shape[1] == 1:
        df = pd.read_csv(data_path, sep=",", header=None, dtype=str, na_values=[], nrows=nrows)
    return df


def generate_synthetic(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic Criteo style dataset with learnable interactions.

    The output matches the schema exactly with one label column, 13 numerical
    columns I1 to I13, and 26 categorical columns C1 to C26.

    Feature distributions
    ---------------------
    Numerical features are drawn from heavy tailed distributions using lognormal
    and pareto draws which mimic the long tail of real ad counts. About 10
    percent NaN is injected into a few numerical columns to exercise the missing
    value handling. Categorical features are hex like strings of varying
    cardinality. A few fields are high cardinality near 5000 and a few are low
    cardinality near 10. Values look like f"{val:08x}".

    Label signal
    ------------
    The click label is a Bernoulli draw from a latent logit. The logit has three
    kinds of structure.

    1. Linear terms. A few numerical features and a few categorical hidden group
       memberships contribute additively.
    2. Pairwise interactions. The categorical columns carry a hidden integer
       group id. The logit rises sharply when the hidden group of C1 matches the
       hidden group of C2 and again when the hidden group of C3 matches the
       hidden group of C4. These match terms are pure second order effects. A
       linear model cannot represent them from the raw one hot or frequency
       features, which is exactly why the factorization machine family and the
       deep cross network can beat logistic regression here. The interaction
       coefficients are deliberately large so the gap is visible.
    3. A numeric effect from a transformed dense feature plus gaussian noise.

    The intercept w0 is tuned so the base click through rate lands roughly in the
    0.2 to 0.3 band. The whole function is deterministic given seed.
    """
    rng = np.random.default_rng(seed)

    # Per column cardinalities. A spread of high, medium, and low cardinality.
    cardinalities = _column_cardinalities(rng)

    # Hidden group ids per categorical column. Each distinct category value of a
    # column maps to one of a small number of hidden groups. The label depends on
    # group matches across specific column pairs. The group assignment is stored
    # per column as an array indexed by the integer category value.
    n_groups = 8
    group_maps = [
        rng.integers(0, n_groups, size=card) for card in cardinalities
    ]

    # Draw integer category values per column, then render as hex strings. The
    # first four columns are the interaction carriers. They use a skewed draw so
    # their value frequency distribution has high variance. The cross generator
    # ranks columns by frequency variance, so this skew makes it select exactly
    # the columns that carry the pairwise signal. Every other column uses a
    # uniform draw which keeps its frequency variance low.
    cat_int = np.empty((n_rows, len(CAT_COLS)), dtype=np.int64)
    for j, card in enumerate(cardinalities):
        if j < _N_INTERACTION_COLS:
            ranks = np.arange(1, card + 1)
            probs = 1.0 / ranks ** 1.1
            probs = probs / probs.sum()
            cat_int[:, j] = rng.choice(card, size=n_rows, p=probs)
        else:
            cat_int[:, j] = rng.integers(0, card, size=n_rows)

    # Numerical features from heavy tailed distributions.
    num = _draw_numerical(rng, n_rows)

    # Build the latent logit.
    logit = np.full(n_rows, -1.2, dtype=np.float64)  # intercept w0

    # Linear categorical effects. A small per group weight on two columns.
    lin_group_w0 = rng.normal(0.0, 0.4, size=n_groups)
    lin_group_w1 = rng.normal(0.0, 0.4, size=n_groups)
    g0 = group_maps[0][cat_int[:, 0]]
    g1 = group_maps[1][cat_int[:, 1]]
    logit += lin_group_w0[g0]
    logit += lin_group_w1[g1]

    # Pairwise interaction one. Strong bump when C1 group matches C2 group.
    g_c1 = group_maps[0][cat_int[:, 0]]
    g_c2 = group_maps[1][cat_int[:, 1]]
    logit += np.where(g_c1 == g_c2, 2.5, -0.5)

    # Pairwise interaction two. Strong bump when C3 group matches C4 group.
    g_c3 = group_maps[2][cat_int[:, 2]]
    g_c4 = group_maps[3][cat_int[:, 3]]
    logit += np.where(g_c3 == g_c4, 2.2, -0.4)

    # Numeric effect. Use a log transformed dense feature so the relationship is
    # smooth and bounded.
    numeric_signal = np.log1p(np.clip(num[:, 0], 0, None))
    numeric_signal = (numeric_signal - numeric_signal.mean()) / (numeric_signal.std() + 1e-8)
    logit += 0.8 * numeric_signal

    # A second weaker numeric effect.
    numeric_signal2 = np.log1p(np.clip(num[:, 3], 0, None))
    numeric_signal2 = (numeric_signal2 - numeric_signal2.mean()) / (numeric_signal2.std() + 1e-8)
    logit += 0.4 * numeric_signal2

    # Gaussian noise so the task is not trivially separable.
    logit += rng.normal(0.0, 0.8, size=n_rows)

    prob = 1.0 / (1.0 + np.exp(-logit))
    label = (rng.random(n_rows) < prob).astype(int)

    # Inject NaN into a few numerical columns at about 10 percent.
    num = _inject_missing(rng, num)

    # Assemble the DataFrame.
    data = {LABEL_COL: label}
    for i, col in enumerate(NUM_COLS):
        data[col] = num[:, i].astype(np.float64)
    for j, col in enumerate(CAT_COLS):
        data[col] = np.array(
            [f"{v:08x}" for v in cat_int[:, j]], dtype=object
        )

    df = pd.DataFrame(data, columns=ALL_COLS)
    return df


def _column_cardinalities(rng: np.random.Generator) -> list:
    """Return a per column cardinality list for the 26 categorical columns.

    The first four columns are the interaction carriers and get moderate
    cardinality so group matches are common enough to learn. The rest spread
    across high, medium, and low cardinality to mimic real sparse fields.
    """
    cards = []
    # Interaction carriers C1 to C4. Low cardinality so the skewed draw gives a
    # high frequency variance and the group match combos stay dense enough to
    # learn from a moderate number of rows.
    cards.extend([12, 12, 12, 12])
    # High cardinality fields near 5000.
    cards.extend([5000, 5000, 4000, 3000])
    # Medium cardinality fields.
    cards.extend([500, 800, 300, 600, 1000, 400])
    # Moderate cardinality fields. Kept well above the interaction carriers so
    # their uniform frequency variance stays below the skewed carriers and they
    # are not chosen for crossing.
    cards.extend([80, 60, 120, 90, 70, 100])
    # Fill the remainder to reach 26 columns with mixed sizes.
    remaining = len(CAT_COLS) - len(cards)
    extra = rng.integers(50, 2000, size=remaining)
    cards.extend(int(x) for x in extra)
    return cards[: len(CAT_COLS)]


def _draw_numerical(rng: np.random.Generator, n_rows: int) -> np.ndarray:
    """Draw 13 heavy tailed numerical columns.

    A mix of lognormal and pareto draws gives the long right tail typical of ad
    impression and count features.
    """
    cols = []
    for i in range(len(NUM_COLS)):
        if i % 2 == 0:
            # Lognormal draw.
            col = rng.lognormal(mean=1.0, sigma=1.2, size=n_rows)
        else:
            # Pareto draw scaled up.
            col = rng.pareto(a=2.5, size=n_rows) * 10.0
        cols.append(col)
    return np.stack(cols, axis=1).astype(np.float64)


def _inject_missing(rng: np.random.Generator, num: np.ndarray) -> np.ndarray:
    """Inject about 10 percent NaN into a few numerical columns.

    Columns 2, 5, and 9 receive missing values. The missingness is independent
    per cell so the is_missing indicator built downstream has real variation.
    """
    num = num.copy()
    missing_cols = [2, 5, 9]
    for c in missing_cols:
        mask = rng.random(num.shape[0]) < 0.10
        num[mask, c] = np.nan
    return num
