#!/usr/bin/env python
"""Score a shard offline through the same bundle the online service serves.

This is the batch lane of the decisioning pipeline. The online service answers
one auction now. This answers every row in a shard as fast as the machine
allows, which is what feeds nightly candidate precomputation, backfills after a
model swap, and the retrospective analysis that asks what the live model would
have done on yesterday's traffic.

The thing worth arguing about is not the throughput, it is the sharing. This job
does not have its own featurizer, its own model loader, or its own calibration.
It loads the same serving bundle scripts/serve.py loads, selects a backend
through the same registry in the same preference order, and calls the same
ScoringEngine method the http handler calls. The only differences are the batch
size and where the rows come from.

That matters because the classic way an offline number and an online number
disagree is that they were produced by two pieces of code that were supposed to
be the same and drifted. A feature that was clipped in one and not the other, a
default fill that changed on one side, a categorical map rebuilt from a
different day of data. Every one of those is invisible in code review and
obvious only in a metric that moved for no reason. Sharing one artifact and one
code path removes the opportunity rather than adding a check for it, and
tests/test_serving.py still adds the check.

Run from the repository root.

    python scripts/run_batch_scoring.py
    python scripts/run_batch_scoring.py --input data/criteo.csv --limit 50000
    python scripts/run_batch_scoring.py --input shard.parquet --output scored.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

# Insert the repository root onto sys.path so that "import src" works when this
# script is run directly from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402

from src.inference.common import NOT_AVAILABLE, jsonable  # noqa: E402
from src.inference.hardware import collect_hardware_record  # noqa: E402
from src.serving.artifact import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BUNDLE_DIR,
    load_or_build_bundle,
)
from src.serving.batch import DEFAULT_CHUNK_ROWS, run_batch_job  # noqa: E402
from src.serving.runtime import ScoringEngine, select_backend  # noqa: E402

DEFAULT_OUTPUT_DIR: str = os.path.join("results", "serving")


def parse_args() -> argparse.Namespace:
    """Parse the command line flags for the batch scoring job."""
    parser = argparse.ArgumentParser(
        description="Batch score a shard through the ClickRank serving bundle."
    )
    parser.add_argument(
        "--input",
        default="",
        help=(
            "Parquet or CSV shard to score. Defaults to data/criteo.csv when it "
            "exists, and otherwise to a synthetic shard written next to the output."
        ),
    )
    parser.add_argument(
        "--output",
        default="",
        help="Where the scored rows are written. Defaults to a parquet file in the output directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory the scored file and the run report are written to.",
    )
    parser.add_argument(
        "--bundle-dir",
        default=DEFAULT_BUNDLE_DIR,
        help="Directory holding the serving bundle. Built when it is not there.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory holding the checkpoint and the exported graph.",
    )
    parser.add_argument(
        "--model", default="deepfm", help="Which trained ranker to score with."
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=DEFAULT_CHUNK_ROWS,
        help="Rows featurized and scored per chunk. This bounds memory, not speed.",
    )
    parser.add_argument(
        "--limit", type=int, default=100000, help="Cap on rows read from the shard."
    )
    parser.add_argument(
        "--id-column",
        default="",
        help="Column that identifies a row in the scored output. Inferred when omitted.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Pin one backend key instead of probing the preference order.",
    )
    parser.add_argument(
        "--allow-reduced-precision",
        action="store_true",
        help="Let the probe consider fp16 and int8 backends, which change the output.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100000,
        help="Rows used when a bundle or a synthetic shard has to be created.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for a synthetic shard."
    )
    return parser.parse_args()


def ensure_input(args: argparse.Namespace) -> str:
    """Resolve the shard to score, writing a synthetic one when there is none.

    The synthetic fallback uses the test split of the same generator and the
    same seed the bundle was fitted on, which is the split nothing in the build
    touched. Scoring the rows a model was fitted on would produce a rows per
    second figure that is perfectly valid and a mean probability that means
    nothing, and someone will read the second number.
    """
    if args.input:
        if not os.path.exists(args.input):
            raise SystemExit(f"no shard at {args.input}.")
        return args.input

    default_real = os.path.join("data", "criteo.csv")
    if os.path.exists(default_real):
        print(f"no --input given. Using the real Criteo file at {default_real}.")
        return default_real

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "batch_input.parquet")
    if os.path.exists(path):
        print(f"no --input given and no Criteo file present. Reusing {path}.")
        return path

    from src.data.loader import generate_synthetic
    from src.data.split import temporal_split

    print(
        "no --input given and no Criteo file present. Writing a synthetic shard "
        f"from the held out test split to {path}."
    )
    frame = generate_synthetic(int(args.sample_size), seed=int(args.seed))
    _, _, test_df = temporal_split(frame)
    test_df = test_df.reset_index(drop=True)
    test_df.insert(0, "ad_id", [f"row-{i}" for i in range(len(test_df))])
    test_df.to_parquet(path, index=False)
    return path


def main() -> None:
    """Load the bundle, select a backend, score the shard, and report."""
    args = parse_args()
    hardware = collect_hardware_record()
    host = hardware.get("host", {})
    hardware_label = (
        f"{host.get('cpu_model', NOT_AVAILABLE)}, "
        f"{host.get('logical_cores', NOT_AVAILABLE)} logical cores"
    )
    print(f"hardware. {hardware_label}.")

    bundle = load_or_build_bundle(
        bundle_dir=args.bundle_dir,
        verbose=True,
        artifact_dir=args.artifact_dir,
        model_name=args.model,
        sample_size=args.sample_size,
    )

    print("probing backends in preference order.")
    selection = select_backend(
        bundle,
        preferred=args.backend,
        allow_reduced_precision=args.allow_reduced_precision,
        max_batch=max(args.chunk_rows, 4096),
        verbose=True,
    )
    print(f"selected the {selection.label} backend.")
    engine = ScoringEngine(bundle, selection, max_candidates=max(args.chunk_rows, 4096))
    engine.warmup(verbose=True)

    input_path = ensure_input(args)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = args.output or os.path.join(args.output_dir, "batch_scored.parquet")

    print(f"scoring {input_path} in chunks of {args.chunk_rows} rows.")
    result = run_batch_job(
        engine,
        input_path=input_path,
        output_path=output_path,
        chunk_rows=args.chunk_rows,
        limit=args.limit,
        id_column=args.id_column or None,
        hardware_label=hardware_label,
    )

    record: Dict[str, Any] = {
        **result.as_dict(),
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend_label": selection.label,
        "lane": "batch",
        "shared_with_online_lane": {
            "bundle_dir": bundle.bundle_dir,
            "feature_pipeline_sha256": bundle.manifest.get("feature_pipeline", {}).get(
                "sha256", ""
            ),
            "checkpoint_sha256": bundle.manifest.get("model", {}).get(
                "checkpoint_sha256", ""
            ),
            "onnx_sha256": bundle.manifest.get("model", {}).get("onnx_sha256", ""),
            "note": (
                "these fingerprints are the same ones /health reports, which is "
                "what makes the online and batch probabilities identical rather "
                "than merely similar"
            ),
        },
        "hardware": hardware,
    }

    report_path = os.path.join(args.output_dir, "batch_scoring.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(jsonable(record), handle, indent=2)
        handle.write("\n")

    print("")
    print(f"scored {result.rows} rows in {result.wall_seconds:.2f} s.")
    print(f"  throughput      {result.rows_per_second:,.0f} rows per second")
    print(
        f"  feature time    {result.feature_seconds:.2f} s "
        f"({result.feature_seconds / result.wall_seconds * 100:.0f} percent of the wall clock)"
    )
    print(
        f"  model time      {result.model_seconds:.2f} s "
        f"({result.model_seconds / result.wall_seconds * 100:.0f} percent of the wall clock)"
    )
    print(
        f"  shard io time   {result.io_seconds:.2f} s "
        f"({result.io_seconds / result.wall_seconds * 100:.0f} percent of the wall clock), "
        "which is reading the shard, marshalling rows, and writing the output"
    )
    print(f"  backend         {selection.label}")
    print(f"  hardware        {hardware_label}")
    print(f"  mean p_click    {result.mean_probability:.4f}")
    print(f"wrote {result.output_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
