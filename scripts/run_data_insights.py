#!/usr/bin/env python
"""SQL analytics over the impression data, run on DuckDB.

The rest of ClickRank makes feature engineering decisions and explains them in
prose. Hash encoding because the fields are high cardinality. Missing indicators
because missingness is signal. Crosses because interactions matter. A temporal
split because the data drifts. Those are all defensible, and until now none of
them were measured on this data.

This script measures them. Each analysis is a real SQL query in `src/sql/`,
executed by DuckDB against the Criteo Parquet or CSV, and each one is the
empirical case for a decision the project already made.

- Long tail cardinality per sparse field, which is the case for hashing.
- Missingness against the label per dense field, which is the case for the
  is_missing indicators.
- Interaction lift over the marginals per crossed pair, which is the case for
  the crosses, and it also checks whether the fields the pipeline picks to cross
  are the fields that actually carry interaction.
- CTR drift across the file order against random folds of the same size, which
  is the case for the temporal split.
- CTR by slice with Wilson score intervals, which is the guard that stops a high
  CTR slice with twelve impressions from being read as a finding.

Every table carries the sample size the number came from. DuckDB needs no JVM,
so this lane runs anywhere.

Run from the repository root.
    python scripts/run_data_insights.py --synthetic
    python scripts/run_data_insights.py --sample-size 500000
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Insert the repository root onto sys.path so that "import src" works when this
# script is run directly from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

# Select a non interactive backend before importing pyplot so the plots work on a
# headless machine.
matplotlib.use("Agg")

import duckdb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.loader import NAN_TOKEN, generate_synthetic  # noqa: E402
from src.schema import ALL_COLS, CAT_COLS, LABEL_COL, NUM_COLS  # noqa: E402
from src.sql import load_query  # noqa: E402

SEED = 42
ROW_INDEX_COL = "row_id"
DEFAULT_DATA_PATH = os.path.join(_REPO_ROOT, "data", "criteo.csv")
DEFAULT_OUTPUT_DIR = os.path.join(_REPO_ROOT, "results", "insights")
REPORT_NAME = "insights_report.md"

# Two sided 95 percent normal quantile, used for the Wilson score intervals.
Z_95 = 1.959963984540054

# Guards the log ratio in the interaction analysis away from zero and one.
EPS = 1e-6

# Number of sparse fields the cross generator selects, matching the default in
# `src.data.preprocess.FeaturePipeline`.
N_CROSS_FEATURES = 5

# Suffix marking the null twin of a pair in the interaction analysis. Every real
# pair is measured alongside a copy of itself whose labels were redrawn at the
# baseline rate, which gives the same cells at the same volumes with the
# interaction removed by construction. That is the control. Without it a weighted
# log lift is a number with nothing to be large relative to.
NULL_TWIN_SUFFIX = " [labels redrawn]"

# Odd 64 bit multiplier that decorrelates the null twin's labels from the row
# index before hashing. Any odd constant works.
NULL_TWIN_MIXER = 2862933555777941757

# Resolution of the integer comparison that stands in for a Bernoulli draw.
NULL_TWIN_SCALE = 1_000_000


def parse_args() -> argparse.Namespace:
    """Parse the command line flags for the analytics run."""
    parser = argparse.ArgumentParser(
        description="Run the SQL analytics layer over the impression data."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use the synthetic generator instead of a Criteo file on disk.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to a Criteo style TSV, CSV, or Parquet file.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200_000,
        help="Number of rows to analyse.",
    )
    parser.add_argument(
        "--min-impressions",
        type=int,
        default=200,
        help="Volume floor below which a categorical slice is not reported.",
    )
    parser.add_argument(
        "--min-cell",
        type=int,
        default=100,
        help="Volume floor for a joint cell in the interaction analysis.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=10,
        help="Rare token cutoff, matching the feature pipeline default.",
    )
    parser.add_argument(
        "--hash-buckets",
        type=int,
        default=10_000,
        help="Hash space per sparse field, matching the feature pipeline default.",
    )
    parser.add_argument(
        "--cross-buckets",
        type=int,
        default=100_000,
        help="Hash space per cross, matching the feature pipeline default.",
    )
    parser.add_argument(
        "--n-buckets",
        type=int,
        default=10,
        help="Number of equal sized blocks used for the drift analysis.",
    )
    parser.add_argument(
        "--slice-limit",
        type=int,
        default=25,
        help="How many CTR slices to report.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the report and the charts.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="DuckDB thread count. 0 leaves the DuckDB default.",
    )
    return parser.parse_args()


def hardware_label() -> str:
    """A short description of the machine every reported number came from."""
    cores = os.cpu_count() or 0
    return (
        f"{platform.system()} {platform.machine()}, {cores} logical cores, "
        f"Python {platform.python_version()}, DuckDB {duckdb.__version__}"
    )


def load_label() -> str:
    """The one minute load average against the core count.

    A wall time on a shared laptop only means something next to what else the
    machine was doing, so it is recorded rather than left for a reader to guess.
    """
    cores = os.cpu_count() or 1
    try:
        one_minute = os.getloadavg()[0]
    except (OSError, AttributeError):
        return "load average unavailable on this platform"
    state = "contended" if one_minute > cores else "quiet"
    return f"one minute load average {one_minute:.1f} against {cores} cores, {state}"


def column_list(columns: Sequence[str]) -> str:
    """Render a column list for an UNPIVOT clause."""
    return ", ".join(columns)


def _sniff_delimiter(data_path: str) -> str:
    """Return the delimiter of a Criteo style file by looking at the first line."""
    with open(data_path, "r", encoding="utf-8", errors="replace") as handle:
        first = handle.readline()
    return "\t" if first.count("\t") >= first.count(",") else ","


def build_source(
    connection: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
    scratch_dir: str,
) -> Tuple[str, int]:
    """Materialize the `impressions` table and say where the rows came from.

    Two ingestion paths land in the same table so every query downstream is
    written against one shape. Dense columns become DOUBLE with missing values as
    SQL NULL, since a NaN read out of a Parquet float column is not a NULL and
    would slip past every `IS NULL` test in the missingness analysis. Sparse
    columns become VARCHAR with missing values folded into the same `__nan__`
    token the pandas loader uses, so the cardinality counts here match the
    vocabulary the feature pipeline actually builds.

    The row index is materialized rather than derived at query time, because it
    is the file order and therefore the time order, and the temporal analysis
    rests on it entirely.
    """
    use_real = not args.synthetic and os.path.exists(args.data_path)

    if use_real:
        source = f"real Criteo file at {args.data_path}"
        if args.data_path.endswith(".parquet"):
            reader = f"read_parquet('{args.data_path}')"
        else:
            delimiter = _sniff_delimiter(args.data_path)
            column_spec = ", ".join(
                [f"'{LABEL_COL}': 'INTEGER'"]
                + [f"'{col}': 'DOUBLE'" for col in NUM_COLS]
                + [f"'{col}': 'VARCHAR'" for col in CAT_COLS]
            )
            reader = (
                f"read_csv('{args.data_path}', delim='{delimiter}', header=false, "
                f"columns={{{column_spec}}}, nullstr='', ignore_errors=true)"
            )
        raw_select = f"SELECT * FROM {reader} LIMIT {int(args.sample_size)}"
    else:
        if not args.synthetic:
            print(
                f"no data file at {args.data_path}, falling back to the "
                "synthetic generator."
            )
        source = "synthetic generator"
        frame = generate_synthetic(args.sample_size, seed=SEED)
        parquet_path = os.path.join(scratch_dir, "impressions.parquet")
        frame[ALL_COLS].to_parquet(parquet_path, index=False)
        raw_select = f"SELECT * FROM read_parquet('{parquet_path}')"

    dense_projection = ",\n    ".join(
        f"CASE WHEN {col} IS NULL OR isnan(CAST({col} AS DOUBLE)) THEN NULL "
        f"ELSE CAST({col} AS DOUBLE) END AS {col}"
        for col in NUM_COLS
    )
    sparse_projection = ",\n    ".join(
        f"CASE WHEN {col} IS NULL OR {col} IN ('', 'nan', '<NA>') "
        f"THEN '{NAN_TOKEN}' ELSE CAST({col} AS VARCHAR) END AS {col}"
        for col in CAT_COLS
    )

    connection.execute("DROP TABLE IF EXISTS impressions")
    connection.execute(
        f"""
        CREATE TABLE impressions AS
        SELECT
            (row_number() OVER () - 1)::BIGINT AS {ROW_INDEX_COL},
            coalesce(CAST({LABEL_COL} AS INTEGER), 0) AS {LABEL_COL},
            {dense_projection},
            {sparse_projection}
        FROM ({raw_select})
        """
    )
    n_rows = int(connection.execute("SELECT count(*) FROM impressions").fetchone()[0])
    return source, n_rows


def build_pair_union(
    pairs: Sequence[Tuple[str, str]],
    baseline_ctr: Optional[float] = None,
) -> str:
    """Build the UNION ALL body the interaction queries expand into.

    When a baseline CTR is given, every pair is emitted twice. The first copy is
    the real data. The second is the null twin, identical in its two value
    columns and therefore identical in its cell structure and its cell volumes,
    but with the label redrawn at the baseline rate from a hash of the row index.
    The twin has no interaction in it by construction, so whatever weighted log
    lift the statistic returns on it is what sampling noise alone produces at
    these cell sizes. That is the number the real pairs have to beat.

    A null twin is the right control here and a pair of low ranked fields is not.
    The fields the pipeline crosses are the low cardinality skewed ones, so their
    joint cells are dense. A pair of high cardinality fields has cells too thin to
    clear any honest volume floor, so comparing against one would confound the
    thing under test with cell volume. The twin holds volume fixed exactly.

    The pair list comes from the schema and from the frequency variance ranking,
    never from user input, so formatting it into the query text is schema
    substitution rather than string interpolation of data.
    """
    parts = []
    for column_a, column_b in pairs:
        parts.append(
            f"    SELECT '{column_a} x {column_b}' AS pair, "
            f"{column_a} AS a_value, {column_b} AS b_value, label FROM impressions"
        )
    if baseline_ctr is not None:
        threshold = int(round(baseline_ctr * NULL_TWIN_SCALE))
        for column_a, column_b in pairs:
            parts.append(
                f"    SELECT '{column_a} x {column_b}{NULL_TWIN_SUFFIX}' AS pair, "
                f"{column_a} AS a_value, {column_b} AS b_value, "
                f"CASE WHEN (hash({ROW_INDEX_COL} * {NULL_TWIN_MIXER}::UBIGINT) "
                f"% {NULL_TWIN_SCALE}) < {threshold} THEN 1 ELSE 0 END AS label "
                f"FROM impressions"
            )
    return "\n    UNION ALL\n".join(parts)


def pair_against_null(interactions: pd.DataFrame) -> pd.DataFrame:
    """Join each real pair to its own null twin and score the ratio.

    Comparing a pair against the loudest null anywhere in the run would be the
    wrong test, because the nulls differ in cell volume and a pair should only
    ever be judged against noise at its own cell sizes. This joins each pair to
    the twin built from its own two columns, so cells, volumes, and marginal
    structure are all held fixed and the only thing that varies is whether the
    label carries the interaction.
    """
    real = interactions[interactions["group"] == "crossed by the pipeline"].copy()
    null = interactions[interactions["group"] == "labels redrawn"].copy()
    null["pair"] = null["pair"].str.replace(NULL_TWIN_SUFFIX, "", regex=False)
    null = null[["pair", "weighted_abs_log_lift"]].rename(
        columns={"weighted_abs_log_lift": "null_abs_log_lift"}
    )
    merged = real.merge(null, on="pair", how="left")
    merged["lift_over_null"] = merged["weighted_abs_log_lift"] / merged[
        "null_abs_log_lift"
    ].clip(lower=1e-12)
    return merged.sort_values("lift_over_null", ascending=False)


def run_query(
    connection: duckdb.DuckDBPyConnection, name: str, **params: Any
) -> pd.DataFrame:
    """Load a query file, substitute its placeholders, and return the result."""
    return connection.execute(load_query(name, **params)).fetchdf()


def analyse(
    connection: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
    n_rows: int,
) -> Dict[str, Any]:
    """Run every analysis and return the results plus the derived context."""
    cat_columns = column_list(CAT_COLS)
    dense_columns = column_list(NUM_COLS)
    n_train = int(n_rows * 0.8)
    n_val = int(n_rows * 0.1)

    baseline_ctr, total_clicks = connection.execute(
        "SELECT avg(label)::DOUBLE, sum(label) FROM impressions"
    ).fetchone()

    print("ranking sparse fields by training frequency variance.")
    ranking = run_query(
        connection,
        "cross_field_ranking",
        cat_columns=cat_columns,
        n_train=max(n_train, 1),
    )
    selected_fields = sorted(ranking["field"].head(N_CROSS_FEATURES).tolist())
    selected_pairs = list(combinations(selected_fields, 2))
    # The lift query measures the real pairs against their null twins. The cell
    # query only ever looks at real data, so it gets the union without them.
    pair_union = build_pair_union(selected_pairs, baseline_ctr=float(baseline_ctr))
    real_pair_union = build_pair_union(selected_pairs)

    print("measuring long tail cardinality per sparse field.")
    cardinality = run_query(
        connection,
        "cardinality_tail",
        cat_columns=cat_columns,
        min_count=args.min_count,
    )

    print("measuring missingness against the label per dense field.")
    missingness = run_query(
        connection, "missingness", dense_columns=dense_columns
    )

    print("measuring CTR by slice with Wilson score intervals.")
    slices = run_query(
        connection,
        "ctr_by_slice",
        cat_columns=cat_columns,
        min_impressions=args.min_impressions,
        z=Z_95,
        limit=args.slice_limit,
    )
    field_power = run_query(
        connection,
        "ctr_field_power",
        cat_columns=cat_columns,
        min_impressions=args.min_impressions,
    )

    print("measuring interaction lift over the marginals per crossed pair.")
    interactions = run_query(
        connection,
        "interaction_lift",
        pair_union=pair_union,
        min_cell=args.min_cell,
        eps=EPS,
    )
    interaction_cells = run_query(
        connection,
        "interaction_cells",
        pair_union=real_pair_union,
        min_cell=args.min_cell,
        eps=EPS,
        limit=12,
    )

    print("measuring CTR drift across the temporal split.")
    drift = run_query(
        connection,
        "temporal_drift",
        n_buckets=args.n_buckets,
        n_train=n_train,
        n_train_val=n_train + n_val,
    )

    interactions["group"] = [
        "labels redrawn" if pair.endswith(NULL_TWIN_SUFFIX) else "crossed by the pipeline"
        for pair in interactions["pair"]
    ]
    interaction_summary = pair_against_null(interactions)

    return {
        "baseline_ctr": float(baseline_ctr),
        "total_clicks": int(total_clicks),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_rows - n_train - n_val,
        "ranking": ranking,
        "selected_fields": selected_fields,
        "selected_pairs": selected_pairs,
        "cardinality": cardinality,
        "missingness": missingness,
        "slices": slices,
        "field_power": field_power,
        "interactions": interactions,
        "interaction_summary": interaction_summary,
        "interaction_cells": interaction_cells,
        "drift": drift,
    }


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------

def plot_cardinality(cardinality: pd.DataFrame, path: str) -> None:
    """Head concentration for the highest cardinality sparse fields."""
    top = cardinality.head(12).iloc[::-1]
    positions = range(len(top))
    figure, axis = plt.subplots(figsize=(9, 6))
    height = 0.26
    axis.barh(
        [p + height for p in positions],
        top["top1000_coverage"],
        height=height,
        label="top 1000 values",
        color="#b8d4e8",
    )
    axis.barh(
        list(positions),
        top["top100_coverage"],
        height=height,
        label="top 100 values",
        color="#5b9bd5",
    )
    axis.barh(
        [p - height for p in positions],
        top["top10_coverage"],
        height=height,
        label="top 10 values",
        color="#1f4e79",
    )
    axis.set_yticks(list(positions))
    axis.set_yticklabels(
        [f"{row.field} ({int(row.distinct_values)} values)" for row in top.itertuples()]
    )
    axis.set_xlabel("share of impressions covered")
    axis.set_xlim(0, 1)
    axis.set_title("Traffic concentration in the head of each sparse field")
    axis.legend(loc="lower right")
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_missingness(missingness: pd.DataFrame, baseline: float, path: str) -> None:
    """CTR when a dense feature is missing against when it is present."""
    present = missingness[missingness["n_missing"] > 0].copy()
    if present.empty:
        present = missingness.copy()
    present = present.sort_values("ctr_delta", key=abs, ascending=False)
    positions = range(len(present))
    figure, axis = plt.subplots(figsize=(9, 5))
    width = 0.38
    axis.bar(
        [p - width / 2 for p in positions],
        present["ctr_missing"].fillna(0.0),
        width=width,
        label="CTR when missing",
        color="#c0504d",
    )
    axis.bar(
        [p + width / 2 for p in positions],
        present["ctr_present"].fillna(0.0),
        width=width,
        label="CTR when present",
        color="#4f81bd",
    )
    axis.axhline(baseline, color="black", linestyle="--", linewidth=1, label="baseline CTR")
    axis.set_xticks(list(positions))
    axis.set_xticklabels(
        [
            f"{row.field}\n{row.missing_rate:.0%} missing"
            for row in present.itertuples()
        ],
        fontsize=8,
    )
    axis.set_ylabel("click through rate")
    axis.set_title("Missingness is not missing at random")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_drift(drift: pd.DataFrame, baseline: float, path: str) -> None:
    """CTR by ordered block against CTR by random fold of the same size."""
    blocks = drift[drift["grain"] == "time block"].sort_values("bucket_order")
    folds = drift[drift["grain"] == "random fold"].sort_values("bucket_order")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        blocks["bucket_order"],
        blocks["ctr"],
        marker="o",
        color="#c0504d",
        label="ordered blocks, in file order",
    )
    axis.plot(
        folds["bucket_order"],
        folds["ctr"],
        marker="s",
        color="#4f81bd",
        label="random folds of the same size",
    )
    axis.axhline(baseline, color="black", linestyle="--", linewidth=1, label="baseline CTR")
    axis.set_xlabel("block index, earliest to latest")
    axis.set_ylabel("click through rate")
    axis.set_title("CTR drift over the file order against sampling noise")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_interactions(summary: pd.DataFrame, path: str) -> None:
    """Interaction strength per pair against its own null twin.

    Plotting the null beside each bar rather than as a single reference line is
    deliberate. The null moves with cell volume, so one line would understate the
    noise on the thin pairs and overstate it on the dense ones.
    """
    ordered = summary.sort_values("weighted_abs_log_lift")
    positions = list(range(len(ordered)))
    height = 0.38
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        [p + height / 2 for p in positions],
        ordered["weighted_abs_log_lift"],
        height=height,
        color="#1f4e79",
        label="real data",
    )
    axis.barh(
        [p - height / 2 for p in positions],
        ordered["null_abs_log_lift"],
        height=height,
        color="#bfbfbf",
        label="same cells, labels redrawn",
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(
        [
            f"{row.pair}  (n={int(row.impressions_covered):,})"
            for row in ordered.itertuples()
        ],
        fontsize=9,
    )
    axis.set_xlabel("impression weighted mean absolute log lift over the marginals")
    axis.set_title("Which crossed pairs carry interaction the marginals do not")
    axis.legend(loc="lower right")
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def markdown_table(frame: pd.DataFrame, columns: Dict[str, str], formats: Dict[str, str]) -> List[str]:
    """Render a DataFrame as a markdown table with explicit column formats."""
    headers = list(columns.values())
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in frame.itertuples(index=False):
        values = []
        for key in columns:
            value = getattr(row, key)
            spec = formats.get(key, "{}")
            if value is None or (isinstance(value, float) and pd.isna(value)):
                values.append("n/a")
            else:
                values.append(spec.format(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    results: Dict[str, Any],
    args: argparse.Namespace,
    source: str,
    n_rows: int,
    elapsed: float,
    output_dir: str,
) -> str:
    """Write the insights report, with the sample size beside every finding."""
    baseline = results["baseline_ctr"]
    cardinality = results["cardinality"]
    missingness = results["missingness"]
    interactions = results["interactions"]
    drift = results["drift"]

    lines: List[str] = []
    lines.append("# Data Insights")
    lines.append("")
    lines.append(
        "Every figure in this report was produced by a SQL query in `src/sql/`, "
        "executed by DuckDB against the dataset named below, and nothing here is "
        "carried over from another run. Sample sizes sit next to the numbers "
        "because a rate without its volume is not a measurement."
    )
    lines.append("")
    lines.append(f"- Source. {source}.")
    lines.append(f"- Rows analysed. {n_rows:,}.")
    lines.append(f"- Baseline CTR. {baseline:.4f} over {n_rows:,} impressions, "
                 f"{results['total_clicks']:,} clicks.")
    lines.append(
        f"- Temporal split. train {results['n_train']:,}, val {results['n_val']:,}, "
        f"test {results['n_test']:,}."
    )
    lines.append(f"- Hardware. {hardware_label()}.")
    lines.append(f"- Machine state. {load_label()}.")
    lines.append(f"- Total query wall time. {elapsed:.2f} s.")
    lines.append("")
    lines.append(
        "The through line is that each feature engineering choice ClickRank "
        "already makes is stated here as a measurement rather than a convention. "
        "Where a measurement fails to support the choice, that is said plainly."
    )
    lines.append("")

    # ---- Cardinality ----
    heaviest = cardinality.iloc[0]
    args_hash_note = (
        f"{len(CAT_COLS) * args.hash_buckets:,} hash buckets in total, "
        f"{args.hash_buckets:,} per field,"
    )
    lines.append("## Long Tail Cardinality, or Why Hashing")
    lines.append("")
    lines.append(
        "One hot encoding a sparse field costs one column per distinct value, and "
        "an embedding table costs one row. Both are only sane when the distinct "
        "value count is small and the values recur often enough to learn from. "
        "The table below is the measurement of whether that holds. "
        "`values for 90 pct` is the number of distinct values, ranked by volume, "
        "needed to cover 90 percent of impressions, so the gap between it and the "
        "distinct value count is the size of the untrainable tail. "
        "`rare traffic` is the share of impressions whose value falls below the "
        f"pipeline's rare token cutoff of {args.min_count} occurrences."
    )
    lines.append("")
    lines += markdown_table(
        cardinality.head(15),
        {
            "field": "Field",
            "distinct_values": "Distinct values",
            "top10_coverage": "Top 10 cover",
            "top100_coverage": "Top 100 cover",
            "values_for_90pct": "Values for 90 pct",
            "singleton_values": "Values seen once",
            "rare_traffic_share": "Rare traffic",
        },
        {
            "distinct_values": "{:,.0f}",
            "top10_coverage": "{:.1%}",
            "top100_coverage": "{:.1%}",
            "values_for_90pct": "{:,.0f}",
            "singleton_values": "{:,.0f}",
            "rare_traffic_share": "{:.2%}",
        },
    )
    lines.append("")
    lines.append(
        f"The widest field is {heaviest['field']} with "
        f"{int(heaviest['distinct_values']):,} distinct values over {n_rows:,} "
        f"impressions, of which {int(heaviest['singleton_values']):,} were seen "
        f"exactly once. Its top 100 values cover "
        f"{heaviest['top100_coverage']:.1%} of traffic, and it takes "
        f"{int(heaviest['values_for_90pct']):,} distinct values to reach 90 "
        "percent. A one hot encoding of this one field would be "
        f"{int(heaviest['distinct_values']) / (2 * len(NUM_COLS)):.0f} times wider "
        "than the entire dense feature block, and almost every column in it would "
        f"be zero on almost every row. Across all {len(cardinality)} sparse fields "
        f"the total distinct value count is "
        f"{int(cardinality['distinct_values'].sum()):,}, against the "
        f"{args_hash_note} the pipeline actually allocates. The collisions that "
        "bounded space accepts fall overwhelmingly on tail values that could not "
        "have been learned separately anyway."
    )
    lines.append("")
    lines.append("![Traffic concentration by field](insights_cardinality.png)")
    lines.append("")

    # ---- Missingness ----
    lines.append("## Missingness, or Why the is_missing Indicators")
    lines.append("")
    lines.append(
        "The dense pipeline fills missing values with zero and appends a binary "
        "indicator per column, which doubles the dense width from 13 to 26. That "
        "is only worth the width if missingness predicts the click. If a column is "
        "missing at random with respect to the label, its indicator is noise. The "
        "z statistic below is a two proportion test using the pooled rate under "
        "the null, so it says how far the gap is from what the volume alone would "
        "produce."
    )
    lines.append("")
    lines += markdown_table(
        missingness,
        {
            "field": "Field",
            "n_missing": "Rows missing",
            "n_present": "Rows present",
            "missing_rate": "Missing rate",
            "ctr_missing": "CTR missing",
            "ctr_present": "CTR present",
            "ctr_delta": "Delta",
            "z_statistic": "z",
        },
        {
            "n_missing": "{:,.0f}",
            "n_present": "{:,.0f}",
            "missing_rate": "{:.1%}",
            "ctr_missing": "{:.4f}",
            "ctr_present": "{:.4f}",
            "ctr_delta": "{:+.4f}",
            "z_statistic": "{:+.1f}",
        },
    )
    lines.append("")
    with_missing = missingness[missingness["n_missing"] > 0]
    if with_missing.empty:
        lines.append(
            "No dense column in this sample carried any missing values, so the "
            "indicator block is thirteen constant columns here and the case for it "
            "cannot be made on this data. That is a property of the sample, not a "
            "verdict on the feature."
        )
    else:
        strongest = with_missing.iloc[0]
        significant = with_missing[with_missing["z_statistic"].abs() >= 3.0]
        lines.append(
            f"{len(with_missing)} of the {len(NUM_COLS)} dense columns carry "
            f"missing values in this sample, and {len(significant)} of those show "
            "a CTR gap with an absolute z above 3. The largest gap is on "
            f"{strongest['field']}, where the "
            f"{int(strongest['n_missing']):,} rows with the value missing click at "
            f"{strongest['ctr_missing']:.4f} against {strongest['ctr_present']:.4f} "
            f"on the {int(strongest['n_present']):,} rows where it is present, a "
            f"gap of {strongest['ctr_delta']:+.4f} at z = "
            f"{strongest['z_statistic']:+.1f}."
        )
        lines.append("")
        if len(significant) > 0:
            lines.append(
                "That is the case for the indicator block. A zero fill alone would "
                "erase the distinction, because a filled zero and a genuine zero "
                "become the same number after log1p, and the indicator is the only "
                "thing that keeps them apart."
            )
        else:
            lines.append(
                "No column clears that bar on this sample, so this data does not "
                "make the case for the indicator block. That is the expected "
                "result here and it is worth stating rather than glossing. The "
                "synthetic generator injects missingness independently of the "
                "latent logit, so its missingness is missing at random by "
                "construction and there is no signal in it to find. The real "
                "Criteo file is where this measurement carries weight, since "
                "several of its dense columns are missing in more than 70 percent "
                "of rows and that missingness is a property of the impression "
                "rather than a coin flip. Rerun this against `data/criteo.csv` to "
                "get the number that decides it."
            )
    lines.append("")
    lines.append("![Missingness against the label](insights_missingness.png)")
    lines.append("")

    # ---- Interactions ----
    summary = results["interaction_summary"]
    lines.append("## Interaction Lift, or Why the Crosses")
    lines.append("")
    lines.append(
        "The cross generator ranks sparse fields by the variance of their training "
        "value frequency distribution and crosses the top "
        f"{N_CROSS_FEATURES} pairwise. On this data that rule selects "
        f"{', '.join(results['selected_fields'])}. The question this section "
        "answers is whether those pairs actually carry interaction, or whether the "
        "ranking rule is picking fields for a reason unrelated to the label."
    )
    lines.append("")
    lines.append(
        "The null model is multiplicative in the lift. If value a lifts CTR by 1.3 "
        "and value b lifts it by 0.8, then with no interaction the joint cell "
        "should sit at 1.04 times baseline. The score is the impression weighted "
        "mean absolute log ratio between the observed cell CTR and that "
        f"prediction, over cells with at least {args.min_cell} impressions."
    )
    lines.append("")
    lines.append(
        "Every pair is measured twice. Once on the real data, and once on a null "
        "twin holding the same two value columns, and therefore the same cells at "
        "the same volumes, with the label redrawn at the baseline rate. The twin "
        "has no interaction in it by construction, so its score is what this "
        "statistic returns from sampling noise alone at these cell sizes. That is "
        "what the real numbers have to be read against."
    )
    lines.append("")
    lines += markdown_table(
        summary,
        {
            "pair": "Pair",
            "measurable_cells": "Cells",
            "impressions_covered": "Impressions",
            "weighted_abs_log_lift": "Real",
            "null_abs_log_lift": "Null twin",
            "lift_over_null": "Real over null",
            "share_beyond_1_5x": "Traffic beyond 1.5x",
        },
        {
            "measurable_cells": "{:,.0f}",
            "impressions_covered": "{:,.0f}",
            "weighted_abs_log_lift": "{:.4f}",
            "null_abs_log_lift": "{:.4f}",
            "lift_over_null": "{:.2f}x",
            "share_beyond_1_5x": "{:.1%}",
        },
    )
    lines.append("")
    crossed = interactions[interactions["group"] == "crossed by the pipeline"]
    control = interactions[interactions["group"] == "labels redrawn"]
    if crossed.empty or control.empty:
        lines.append(
            "One of the two groups produced no cell above the volume floor of "
            f"{args.min_cell} impressions, so the comparison cannot be made on "
            "this sample. Lower `--min-cell` or raise `--sample-size` to bracket it."
        )
    else:
        clear = summary[summary["lift_over_null"] >= 1.5]
        marginal = summary[summary["lift_over_null"] < 1.2]
        lines.append(
            f"{len(clear)} of the {len(summary)} pairs beat their own null twin by "
            "1.5 times or more, which is the reading that matters, since a pair "
            "judged against noise at its own cell sizes is the only fair test. "
            f"The strongest is {summary.iloc[0]['pair']} at "
            f"{summary.iloc[0]['lift_over_null']:.1f} times its null over "
            f"{int(summary.iloc[0]['impressions_covered']):,} impressions in "
            f"{int(summary.iloc[0]['measurable_cells'])} cells. Those pairs are "
            "encoding something the two marginals do not already carry, which is "
            "the justification for spending bucket space on a cross."
        )
        lines.append("")
        if len(marginal) > 0:
            marginal_names = ", ".join(marginal["pair"].tolist())
            lines.append(
                f"The other side of that result is worth stating plainly. "
                f"{len(marginal)} pairs sit within 1.2 times their null twin, "
                f"which is {marginal_names}. Their apparent lift is what this "
                "statistic returns from sampling noise alone, and they are paying "
                f"for a {args.cross_buckets:,} bucket embedding table to encode "
                "nothing. On this data the top k rule crosses more pairs than the "
                "measurement supports, and a rule that ranked pairs by lift over "
                "their null would keep the same signal at a fraction of the "
                "parameter cost."
            )
        lines.append("")
        lines.append(
            "The frequency variance rule that selects these fields never looks at "
            "the label, so this is not circular. It selects fields whose value "
            "distributions are concentrated, and concentration is what makes joint "
            "cells dense enough for an interaction to be measurable at all. The "
            "high cardinality fields the rule passes over would not produce a "
            "single cell above the volume floor at this sample size, which is a "
            "second and independent reason not to cross them."
        )
    lines.append("")
    lines.append("The individual cells driving that, each with its impression count.")
    lines.append("")
    lines += markdown_table(
        results["interaction_cells"],
        {
            "pair": "Pair",
            "a_value": "Value a",
            "b_value": "Value b",
            "impressions": "Impressions",
            "observed_ctr": "Observed CTR",
            "expected_ctr": "Expected from marginals",
            "interaction_lift": "Lift",
        },
        {
            "impressions": "{:,.0f}",
            "observed_ctr": "{:.4f}",
            "expected_ctr": "{:.4f}",
            "interaction_lift": "{:.2f}x",
        },
    )
    lines.append("")
    lines.append("![Interaction lift by pair](insights_interactions.png)")
    lines.append("")

    # ---- Drift ----
    lines.append("## CTR Drift, or Why a Temporal Split")
    lines.append("")
    lines.append(
        "A temporal split costs accuracy on paper compared to a random one. It is "
        "worth paying only if the data is non stationary, because a random split "
        "over stationary data leaks nothing that matters. This section measures "
        f"the non stationarity. The file is cut into {args.n_buckets} equal blocks "
        "in order, and into the same number of equal random folds. The random folds "
        "are the control, since their spread is what pure sampling noise looks like "
        "at this volume."
    )
    lines.append("")
    split_rows = drift[drift["grain"] == "temporal split"]
    lines += markdown_table(
        split_rows,
        {
            "bucket": "Split",
            "impressions": "Impressions",
            "clicks": "Clicks",
            "ctr": "CTR",
        },
        {"impressions": "{:,.0f}", "clicks": "{:,.0f}", "ctr": "{:.4f}"},
    )
    lines.append("")
    blocks = drift[drift["grain"] == "time block"]
    folds = drift[drift["grain"] == "random fold"]
    # The spread statistic is a standard deviation rather than a range. A range
    # is an order statistic over ten numbers and it moves a lot on its own, so
    # comparing two ranges would read noise as a finding at exactly the moment
    # this section is trying not to.
    block_sd = float(blocks["ctr"].std(ddof=0))
    fold_sd = float(folds["ctr"].std(ddof=0))
    ratio = block_sd / fold_sd if fold_sd > 0 else float("inf")
    per_block = int(blocks["impressions"].mean())
    lines.append(
        f"CTR across the {args.n_buckets} ordered blocks spans "
        f"{blocks['ctr'].min():.4f} to {blocks['ctr'].max():.4f} with a standard "
        f"deviation of {block_sd:.4f}. Across {args.n_buckets} random folds of "
        f"the same size it spans {folds['ctr'].min():.4f} to "
        f"{folds['ctr'].max():.4f} with a standard deviation of {fold_sd:.4f}. "
        f"The ordered spread is {ratio:.1f} times the random spread, on "
        f"{per_block:,} impressions per block."
    )
    lines.append("")
    synthetic_source = "synthetic" in source
    if synthetic_source:
        lines.append(
            "On synthetic data this comparison has a known right answer and it is "
            "worth checking against. The generator draws every row independently, "
            "so the file has no time structure in it at all and the honest "
            "expectation is a ratio near one. Anything the ordered blocks show "
            "here is sampling noise wearing the shape of drift, which is exactly "
            f"why the random folds are computed alongside them. The {ratio:.1f} "
            "observed is the size of that noise, not a measurement of drift, and "
            "it is the reason a range based version of this statistic would have "
            "been misleading. Rerun against `data/criteo.csv`, where the rows are "
            "in genuine time order, for the number that decides the split."
        )
    elif ratio >= 2.0:
        lines.append(
            "That is drift, and it is well outside what folds of this size "
            "produce by chance. It means a model validated on a random slice "
            "would be graded on rows drawn from the same period it trained on, "
            "and the grade would not survive contact with the next period. The "
            "temporal split is what makes the test set a forward looking one."
        )
    else:
        lines.append(
            "That is not enough separation to call drift. The ordered blocks are "
            "within the range that random folds of the same size produce on their "
            "own, so on this sample the click rate is close to stationary and the "
            "temporal split is not buying anything on the base rate alone. It "
            "still prevents the leakage this statistic does not measure, which is "
            "the frequency encodings, the rare token cutoff, and the "
            "standardization statistics all being fit on rows that a random split "
            "would have drawn from the future."
        )
    lines.append("")
    lines.append("![CTR drift across the split](insights_drift.png)")
    lines.append("")

    # ---- Slices ----
    lines.append("## CTR by Slice, With Intervals")
    lines.append("")
    lines.append(
        "The slice table is where a volume floor and a confidence interval stop a "
        "list of numbers from becoming a list of mistakes. Every slice below "
        f"cleared a floor of {args.min_impressions} impressions, and each carries a "
        "Wilson score interval at 95 percent. A slice is only reported as separated "
        "from the baseline when that interval excludes the baseline CTR entirely. "
        "The Wilson interval is used rather than the normal approximation because "
        "it does not collapse to zero width when a slice has no clicks."
    )
    lines.append("")
    lines += markdown_table(
        results["slices"],
        {
            "field": "Field",
            "value": "Value",
            "impressions": "Impressions",
            "clicks": "Clicks",
            "ctr": "CTR",
            "ctr_low": "CI low",
            "ctr_high": "CI high",
            "ctr_lift": "Lift",
            "separated_from_baseline": "Separated",
        },
        {
            "impressions": "{:,.0f}",
            "clicks": "{:,.0f}",
            "ctr": "{:.4f}",
            "ctr_low": "{:.4f}",
            "ctr_high": "{:.4f}",
            "ctr_lift": "{:.2f}x",
        },
    )
    lines.append("")
    separated = int(results["slices"]["separated_from_baseline"].sum())
    lines.append(
        f"{separated} of the {len(results['slices'])} reported slices have an "
        "interval that excludes the baseline. Rolled up per field, the impression "
        "weighted CTR deviation says which fields are worth modelling at all."
    )
    lines.append("")
    lines += markdown_table(
        results["field_power"].head(12),
        {
            "field": "Field",
            "distinct_values": "Distinct values",
            "measurable_slices": "Slices above floor",
            "measurable_impressions": "Impressions covered",
            "covered_share": "Share of traffic",
            "weighted_ctr_deviation": "Weighted CTR deviation",
        },
        {
            "distinct_values": "{:,.0f}",
            "measurable_slices": "{:,.0f}",
            "measurable_impressions": "{:,.0f}",
            "covered_share": "{:.1%}",
            "weighted_ctr_deviation": "{:.4f}",
        },
    )
    lines.append("")
    lines.append("## Reproducing This")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/run_data_insights.py --synthetic --sample-size "
                 f"{args.sample_size}")
    lines.append("```")
    lines.append("")
    lines.append(
        "The queries live in `src/sql/` and run against a DuckDB table named "
        "`impressions`. They are plain SQL and can be pointed at any Parquet or "
        "CSV with the Criteo schema without going through this script."
    )
    lines.append("")

    report_path = os.path.join(output_dir, REPORT_NAME)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return report_path


def main() -> int:
    """Run every analysis, write the charts, and write the report."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    connection = duckdb.connect()
    if args.threads > 0:
        connection.execute(f"PRAGMA threads={int(args.threads)}")

    scratch_dir = tempfile.mkdtemp(prefix="clickrank_insights_")
    try:
        source, n_rows = build_source(connection, args, scratch_dir)
        print(f"analysing {n_rows:,} rows from the {source}.")
        if n_rows == 0:
            print("no rows to analyse. nothing was written.")
            return 0

        start = time.perf_counter()
        results = analyse(connection, args, n_rows)
        elapsed = time.perf_counter() - start

        plot_cardinality(
            results["cardinality"],
            os.path.join(args.output_dir, "insights_cardinality.png"),
        )
        plot_missingness(
            results["missingness"],
            results["baseline_ctr"],
            os.path.join(args.output_dir, "insights_missingness.png"),
        )
        plot_interactions(
            results["interaction_summary"],
            os.path.join(args.output_dir, "insights_interactions.png"),
        )
        plot_drift(
            results["drift"],
            results["baseline_ctr"],
            os.path.join(args.output_dir, "insights_drift.png"),
        )

        report_path = write_report(
            results, args, source, n_rows, elapsed, args.output_dir
        )

        summary = {
            "source": source,
            "rows": n_rows,
            "baseline_ctr": results["baseline_ctr"],
            "hardware": hardware_label(),
            "query_seconds": elapsed,
            "selected_fields": results["selected_fields"],
        }
        with open(
            os.path.join(args.output_dir, "insights_summary.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle, indent=2)

        print("")
        print(f"queries finished in {elapsed:.2f} s.")
        print(f"wrote the report to {report_path}")
        print(f"wrote four charts to {args.output_dir}")
    finally:
        connection.close()
        import shutil

        shutil.rmtree(scratch_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
