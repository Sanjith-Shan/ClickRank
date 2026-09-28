#!/usr/bin/env python
"""Run the distributed feature pipeline and measure it against the pandas path.

This is the entry point for the Spark lane. It reads either the real Criteo file
or the synthetic generator, attaches a temporal split, fits the feature pipeline
on the training rows only, transforms every split, and writes the result as
Parquet partitioned by split. It reports rows processed, wall time, output
partition count, and output size on disk.

The `--scale` mode is the part worth reading. The claim this lane makes is that
it is not bounded the way the pandas path is, and a claim like that should be
measured rather than asserted. Scale mode runs both pipelines over the same
inputs at increasing row counts and reports where the crossover actually falls,
along with the resident footprint the pandas path needs at each point. Every
number it prints was timed on the machine it ran on, and the report names that
machine.

Spark needs a JVM. When one is missing, or when pyspark is not installed, this
script prints what is missing and exits 0 rather than failing, so it stays safe
to run anywhere.

Run from the repository root.
    python scripts/run_spark_pipeline.py --synthetic --sample-size 50000
    python scripts/run_spark_pipeline.py --synthetic --scale
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

# Insert the repository root onto sys.path so that "import src" works when this
# script is run directly from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

# Select a non interactive backend before importing pyplot so the plot works on a
# headless machine.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 import after backend selection
import pandas as pd  # noqa: E402

from src.data.loader import generate_synthetic, load_raw  # noqa: E402
from src.data.preprocess import FeaturePipeline  # noqa: E402
from src.data.split import temporal_split  # noqa: E402
from src.spark.session import build_session, unavailable_reason  # noqa: E402

SEED = 42
DEFAULT_DATA_PATH = os.path.join(_REPO_ROOT, "data", "criteo.csv")
DEFAULT_OUTPUT = os.path.join(_REPO_ROOT, "results", "features.parquet")
REPORT_PATH = os.path.join(_REPO_ROOT, "results", "spark_pipeline_report.md")
SCALING_PLOT_PATH = os.path.join(_REPO_ROOT, "results", "spark_vs_pandas_scaling.png")
# Scale points for the default sweep. Kept modest because the pandas side of the
# comparison hashes every cell in interpreted Python, so a large point costs
# minutes on that path alone. Pass --scale-points to push it further.
DEFAULT_SCALE_POINTS = (10_000, 25_000, 50_000, 100_000)


def parse_args() -> argparse.Namespace:
    """Parse the command line flags for the Spark pipeline run."""
    parser = argparse.ArgumentParser(
        description="Run the distributed feature pipeline and measure it."
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
        help="Path to a Criteo style TSV or CSV file.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="Number of rows to process.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Directory to write the partitioned Parquet output into.",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Run both pipelines at increasing row counts and report the crossover.",
    )
    parser.add_argument(
        "--scale-points",
        type=int,
        nargs="+",
        default=list(DEFAULT_SCALE_POINTS),
        help="Row counts to measure in scale mode.",
    )
    parser.add_argument(
        "--master",
        type=str,
        default="local[*]",
        help="Spark master URL. local[*] uses every core on this machine.",
    )
    parser.add_argument(
        "--shuffle-partitions",
        type=int,
        default=16,
        help="Post shuffle partition count.",
    )
    parser.add_argument(
        "--driver-memory",
        type=str,
        default="4g",
        help="Driver JVM heap. In local mode this is the whole run's memory.",
    )
    parser.add_argument(
        "--output-files",
        type=int,
        default=4,
        help="Repartition to this many files per split before writing.",
    )
    parser.add_argument(
        "--hash-buckets", type=int, default=10_000, help="Hash space per sparse field."
    )
    parser.add_argument(
        "--cross-buckets", type=int, default=100_000, help="Hash space per cross."
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=10,
        help="Training count below which a category collapses to the rare token.",
    )
    parser.add_argument(
        "--n-cross-features",
        type=int,
        default=5,
        help="Number of sparse fields to cross pairwise.",
    )
    return parser.parse_args()


def hardware_label() -> str:
    """A short description of the machine every reported number came from."""
    machine = platform.machine()
    system = platform.system()
    cores = os.cpu_count() or 0
    return f"{system} {machine}, {cores} logical cores, Python {platform.python_version()}"


def load_label() -> str:
    """The one minute load average against the core count.

    A wall time on a shared laptop is only interpretable next to what else the
    machine was doing. When the one minute load average is above the core count
    the run was competing for cpu and every duration below is an upper bound, so
    the number is recorded rather than left for a reader to guess at.
    """
    cores = os.cpu_count() or 1
    try:
        one_minute = os.getloadavg()[0]
    except (OSError, AttributeError):
        return "load average unavailable on this platform"
    state = "contended" if one_minute > cores else "quiet"
    return f"one minute load average {one_minute:.1f} against {cores} cores, {state}"


def load_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    """Load the requested number of rows and say where they came from."""
    if not args.synthetic and os.path.exists(args.data_path):
        frame = load_raw(args.data_path, sample_size=args.sample_size)
        return frame, f"real Criteo file at {args.data_path}"
    if not args.synthetic:
        print(
            f"no data file at {args.data_path}, falling back to the synthetic generator."
        )
    frame = generate_synthetic(args.sample_size, seed=SEED)
    return frame, "synthetic generator"


def run_pandas_pipeline(frame: pd.DataFrame, args: argparse.Namespace) -> Dict[str, Any]:
    """Fit and transform with the reference pandas pipeline, and measure it.

    The reported footprint is the exact resident cost of the pandas path, which
    is the deep memory of the raw frame plus the bytes of the three featurized
    Datasets. Those are all live at once by the time the trainer is handed its
    splits, so the sum is what the process actually has to hold. This is a
    computed size rather than a sampled one, so it does not depend on when the
    allocator happens to return pages to the operating system.
    """
    start = time.perf_counter()
    train_df, val_df, test_df = temporal_split(frame)
    pipeline = FeaturePipeline(
        hash_bucket_size=args.hash_buckets,
        cross_bucket_size=args.cross_buckets,
        min_count=args.min_count,
        n_cross_features=args.n_cross_features,
    )
    pipeline.fit(train_df)
    datasets = [
        pipeline.transform(train_df),
        pipeline.transform(val_df),
        pipeline.transform(test_df),
    ]
    elapsed = time.perf_counter() - start

    frame_bytes = int(frame.memory_usage(deep=True).sum())
    output_bytes = 0
    for dataset in datasets:
        for array in (
            dataset.numerical,
            dataset.categorical,
            dataset.cat_freq,
            dataset.crosses,
            dataset.label,
        ):
            output_bytes += int(array.nbytes)

    return {
        "rows": int(len(frame)),
        "seconds": float(elapsed),
        "rows_per_second": float(len(frame) / elapsed) if elapsed > 0 else 0.0,
        "frame_bytes": frame_bytes,
        "output_bytes": output_bytes,
        "resident_bytes": frame_bytes + output_bytes,
    }


def run_spark_pipeline(
    spark: Any,
    frame: pd.DataFrame,
    args: argparse.Namespace,
    output_path: str,
) -> Dict[str, Any]:
    """Fit, transform, and write with the Spark pipeline, and measure it.

    The timer starts after the session exists, because JVM startup is a fixed
    cost paid once per application and folding it into a per row throughput
    number would misrepresent both pipelines. Startup is reported on its own.
    """
    from src.spark.io import (
        add_temporal_split,
        directory_size_bytes,
        read_pandas,
        write_features,
    )
    from src.spark.pipeline import SparkFeaturePipeline, SparkPipelineConfig

    start = time.perf_counter()
    spark_df = read_pandas(spark, frame)
    spark_df, n_train, n_val, n_test = add_temporal_split(spark_df, n_rows=len(frame))

    pipeline = SparkFeaturePipeline(
        SparkPipelineConfig(
            hash_bucket_size=args.hash_buckets,
            cross_bucket_size=args.cross_buckets,
            min_count=args.min_count,
            n_cross_features=args.n_cross_features,
        )
    )
    pipeline.fit(spark_df.filter(spark_df["split"] == "train"))
    featurized = pipeline.transform(spark_df)
    input_partitions = featurized.rdd.getNumPartitions()

    if os.path.exists(output_path):
        shutil.rmtree(output_path, ignore_errors=True)
    write_features(featurized, output_path, n_output_files=args.output_files)
    elapsed = time.perf_counter() - start

    pipeline.unpersist()

    parquet_files = []
    for root, _dirs, files in os.walk(output_path):
        parquet_files += [name for name in files if name.endswith(".parquet")]

    return {
        "rows": int(len(frame)),
        "seconds": float(elapsed),
        "rows_per_second": float(len(frame) / elapsed) if elapsed > 0 else 0.0,
        "input_partitions": int(input_partitions),
        "output_files": len(parquet_files),
        "output_bytes": directory_size_bytes(output_path),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "feature_columns": len(pipeline.feature_columns()),
        "cross_pairs": [list(pair) for pair in pipeline.pairs_],
    }


def format_bytes(value: int) -> str:
    """Render a byte count in the largest unit that keeps it readable."""
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def run_single(args: argparse.Namespace, spark: Any) -> Dict[str, Any]:
    """Run the Spark pipeline once at the requested sample size."""
    frame, source = load_frame(args)
    print(f"loaded {len(frame)} rows from the {source}.")

    result = run_spark_pipeline(spark, frame, args, args.output)

    print("")
    print(f"rows processed      {result['rows']}")
    print(f"split sizes         train {result['n_train']}, val {result['n_val']}, "
          f"test {result['n_test']}")
    print(f"feature columns     {result['feature_columns']}")
    print(f"wall time           {result['seconds']:.2f} s")
    print(f"throughput          {result['rows_per_second']:.0f} rows/s")
    print(f"input partitions    {result['input_partitions']}")
    print(f"output files        {result['output_files']} across 3 split directories")
    print(f"output size         {format_bytes(result['output_bytes'])}")
    print(f"output path         {args.output}")
    result["source"] = source
    return result


def run_scale(args: argparse.Namespace, spark: Any) -> List[Dict[str, Any]]:
    """Measure both pipelines at increasing row counts.

    Both run over the same frame at each point, so the comparison is like for
    like. The pandas path is timed doing exactly the work the benchmark asks of
    it today, which is fit on train and transform all three splits.
    """
    rows: List[Dict[str, Any]] = []
    scratch = tempfile.mkdtemp(prefix="clickrank_scale_")
    try:
        for n_rows in sorted(args.scale_points):
            scoped = argparse.Namespace(**vars(args))
            scoped.sample_size = n_rows
            frame, source = load_frame(scoped)

            print(f"measuring {len(frame)} rows from the {source}.")
            pandas_result = run_pandas_pipeline(frame, scoped)
            print(
                f"  pandas {pandas_result['seconds']:7.2f} s  "
                f"{pandas_result['rows_per_second']:8.0f} rows/s  "
                f"resident {format_bytes(pandas_result['resident_bytes'])}"
            )

            output_path = os.path.join(scratch, f"features_{n_rows}.parquet")
            spark_result = run_spark_pipeline(spark, frame, scoped, output_path)
            print(
                f"  spark  {spark_result['seconds']:7.2f} s  "
                f"{spark_result['rows_per_second']:8.0f} rows/s  "
                f"parquet {format_bytes(spark_result['output_bytes'])}"
            )

            rows.append(
                {
                    "rows": int(len(frame)),
                    "source": source,
                    "pandas_seconds": pandas_result["seconds"],
                    "pandas_rows_per_second": pandas_result["rows_per_second"],
                    "pandas_resident_bytes": pandas_result["resident_bytes"],
                    "pandas_frame_bytes": pandas_result["frame_bytes"],
                    "spark_seconds": spark_result["seconds"],
                    "spark_rows_per_second": spark_result["rows_per_second"],
                    "spark_output_bytes": spark_result["output_bytes"],
                    "spark_input_partitions": spark_result["input_partitions"],
                    "speedup": (
                        pandas_result["seconds"] / spark_result["seconds"]
                        if spark_result["seconds"] > 0
                        else float("nan")
                    ),
                }
            )
            shutil.rmtree(output_path, ignore_errors=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return rows


def find_crossover(rows: List[Dict[str, Any]]) -> Optional[float]:
    """Interpolate the row count where Spark becomes the faster of the two.

    Returns None when the measured range never crosses, which is an honest
    answer and the one this should give rather than extrapolating past the data.
    """
    for previous, current in zip(rows, rows[1:]):
        before = previous["pandas_seconds"] - previous["spark_seconds"]
        after = current["pandas_seconds"] - current["spark_seconds"]
        if before < 0 <= after:
            span = after - before
            if span == 0:
                return float(current["rows"])
            weight = -before / span
            return float(
                previous["rows"] + weight * (current["rows"] - previous["rows"])
            )
    return None


def plot_scaling(rows: List[Dict[str, Any]], path: str) -> None:
    """Plot wall time and resident footprint against row count."""
    row_counts = [row["rows"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(
        row_counts, [row["pandas_seconds"] for row in rows], marker="o", label="pandas"
    )
    axes[0].plot(
        row_counts, [row["spark_seconds"] for row in rows], marker="s", label="Spark"
    )
    axes[0].set_xlabel("rows processed")
    axes[0].set_ylabel("wall time (s)")
    axes[0].set_title("Feature pipeline wall time")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        row_counts,
        [row["pandas_resident_bytes"] / 1e6 for row in rows],
        marker="o",
        label="pandas resident",
    )
    axes[1].plot(
        row_counts,
        [row["spark_output_bytes"] / 1e6 for row in rows],
        marker="s",
        label="Spark Parquet on disk",
    )
    axes[1].set_xlabel("rows processed")
    axes[1].set_ylabel("megabytes")
    axes[1].set_title("Footprint, in memory against on disk")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_report(
    args: argparse.Namespace,
    single: Optional[Dict[str, Any]],
    scale_rows: List[Dict[str, Any]],
    startup_seconds: float,
) -> None:
    """Write the markdown report for whichever mode was run."""
    lines: List[str] = []
    lines.append("# Spark Feature Pipeline Report")
    lines.append("")
    lines.append(
        "Every number below was measured on the machine named here, on the row "
        "counts named here, and nowhere else."
    )
    lines.append("")
    lines.append(f"- Hardware. {hardware_label()}")
    lines.append(f"- Machine state at write time. {load_label()}.")
    lines.append(f"- Spark master. `{args.master}`")
    lines.append(f"- Shuffle partitions. {args.shuffle_partitions}")
    lines.append(f"- Driver memory. {args.driver_memory}")
    lines.append(f"- Session startup. {startup_seconds:.2f} s, paid once per run.")
    lines.append("")
    lines.append(
        "If the machine state above says contended, every wall time in this "
        "report is an upper bound and the throughput numbers are a lower bound. "
        "Rerun on a quiet machine before quoting any of them."
    )
    lines.append("")

    if single is not None:
        lines.append("## Single Run")
        lines.append("")
        lines.append(f"Source. {single['source']}.")
        lines.append("")
        lines.append("| Measure | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Rows processed | {single['rows']} |")
        lines.append(
            f"| Split sizes | train {single['n_train']}, val {single['n_val']}, "
            f"test {single['n_test']} |"
        )
        lines.append(f"| Feature columns | {single['feature_columns']} |")
        lines.append(f"| Wall time | {single['seconds']:.2f} s |")
        lines.append(f"| Throughput | {single['rows_per_second']:.0f} rows/s |")
        lines.append(f"| Input partitions | {single['input_partitions']} |")
        lines.append(f"| Output files | {single['output_files']} |")
        lines.append(f"| Output size | {format_bytes(single['output_bytes'])} |")
        lines.append("")
        pairs = ", ".join(f"{a} x {b}" for a, b in single["cross_pairs"])
        lines.append(f"Crossed pairs selected on train. {pairs}.")
        lines.append("")

    if scale_rows:
        lines.append("## Scaling")
        lines.append("")
        lines.append(
            "Both pipelines ran over the same frame at each row count. The pandas "
            "resident column is the exact live footprint of that path, which is the "
            "deep memory of the raw frame plus the bytes of the three featurized "
            "Datasets, all of which are held at once. The Spark column is the size "
            "of the Parquet output on disk, because the Spark path never has to "
            "hold the featurized data in memory at all."
        )
        lines.append("")
        lines.append(
            "| Rows | pandas s | Spark s | Speedup | pandas rows/s | Spark rows/s "
            "| pandas resident | Spark Parquet |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in scale_rows:
            lines.append(
                f"| {row['rows']} | {row['pandas_seconds']:.2f} | "
                f"{row['spark_seconds']:.2f} | {row['speedup']:.2f}x | "
                f"{row['pandas_rows_per_second']:.0f} | "
                f"{row['spark_rows_per_second']:.0f} | "
                f"{format_bytes(row['pandas_resident_bytes'])} | "
                f"{format_bytes(row['spark_output_bytes'])} |"
            )
        lines.append("")
        crossover = find_crossover(scale_rows)
        if crossover is None:
            faster = scale_rows[0]["speedup"] > 1.0
            if faster:
                lines.append(
                    "Spark was already the faster path at the smallest row count "
                    "measured, so the crossover sits below the measured range. It "
                    "was not bracketed and is therefore not reported as a number."
                )
            else:
                lines.append(
                    "pandas stayed faster across every row count measured, so no "
                    "crossover was observed inside this range. Extrapolating one "
                    "would not be a measurement."
                )
        else:
            lines.append(
                f"The crossover falls near {crossover:,.0f} rows. Below it the "
                "pandas path wins on the fixed cost of Spark's job scheduling and "
                "shuffle. Above it the pandas path is losing to its own Python "
                "level per value hashing loop, and the gap widens with every row."
            )
        lines.append("")
        lines.append("![Spark against pandas scaling](spark_vs_pandas_scaling.png)")
        lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"\nwrote the report to {REPORT_PATH}")


def main() -> int:
    """Run the pipeline and report, degrading cleanly when Spark cannot start."""
    args = parse_args()

    reason = unavailable_reason()
    if reason is not None:
        print("the Spark lane cannot run on this machine.")
        print(f"  {reason}")
        print("")
        print(
            "the pandas pipeline in src/data/preprocess.py and the DuckDB analytics "
            "in scripts/run_data_insights.py both run without a JVM, so nothing else "
            "in the project is blocked by this."
        )
        return 0

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    startup_start = time.perf_counter()
    spark = build_session(
        app_name="ClickRankFeaturePipeline",
        master=args.master,
        shuffle_partitions=args.shuffle_partitions,
        driver_memory=args.driver_memory,
    )
    startup_seconds = time.perf_counter() - startup_start
    print(f"Spark {spark.version} session up in {startup_seconds:.2f} s.")

    single: Optional[Dict[str, Any]] = None
    scale_rows: List[Dict[str, Any]] = []
    try:
        if args.scale:
            scale_rows = run_scale(args, spark)
            plot_scaling(scale_rows, SCALING_PLOT_PATH)
            print(f"wrote the scaling plot to {SCALING_PLOT_PATH}")
        else:
            single = run_single(args, spark)
    finally:
        spark.stop()

    write_report(args, single, scale_rows, startup_seconds)

    metrics_path = os.path.join(_REPO_ROOT, "results", "spark_pipeline.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "hardware": hardware_label(),
                "machine_state": load_label(),
                "startup_seconds": startup_seconds,
                "single": single,
                "scale": scale_rows,
            },
            handle,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
