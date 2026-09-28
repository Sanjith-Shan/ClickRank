#!/usr/bin/env python
"""Main entry point for the ClickRank click through rate benchmark.

This script runs the full pipeline end to end. It loads or synthesizes data,
splits it temporally, builds the featurized datasets, trains every selected
model under the same training budget, scores them on the held out test split,
and writes the results to disk.

Outputs written under the output directory.
- metrics.json holds the per model metrics so the report can be regenerated.
- calibration.png overlays the reliability curves of every model.
- benchmark_report.md is the human readable comparison report.
- one state dict file per torch model named after the model.

The script runs without any download. When no real data file is present it falls
back to the synthetic generator which carries a real pairwise interaction signal.
That signal is what lets the factorization machine family and the deep cross
network beat the logistic regression baseline.

Run from the repository root.
    python scripts/run_benchmark.py --synthetic --sample-size 100000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Insert the repository root onto sys.path so that "import src" works when this
# script is run directly from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from src.data.loader import generate_synthetic, load_raw
from src.data.preprocess import build_datasets
from src.data.split import temporal_split
from src.evaluation.calibration import (
    calibration_curve_data,
    expected_calibration_error,
    plot_calibration,
)
from src.evaluation.metrics import compute_all
from src.evaluation.report import generate_report
from src.models.dcn import DCNModel
from src.models.deepfm import DeepFMModel
from src.models.dlrm import DLRMModel, torchrec_available
from src.models.dnn import DNNModel
from src.models.fm import FMModel
from src.models.logistic import LogisticModel
from src.schema import CAT_COLS
from src.train.config import MODEL_ORDER, SEED, get_config
from src.train.trainer import get_device, set_seed

# Maps a model name to its BaseModel class. The keys match MODEL_ORDER.
_MODEL_CLASSES = {
    "logistic": LogisticModel,
    "fm": FMModel,
    "deepfm": DeepFMModel,
    "dcn": DCNModel,
    "dnn": DNNModel,
    "dlrm": DLRMModel,
}

# Names that identify torch backed models. These have a trainable module whose
# state dict is saved and whose parameters are counted.
_TORCH_MODELS = {"fm", "deepfm", "dcn", "dnn", "dlrm"}


def parse_args() -> argparse.Namespace:
    """Parse the command line flags for the benchmark run."""
    parser = argparse.ArgumentParser(
        description="Run the ClickRank click through rate benchmark."
    )
    parser.add_argument(
        "--data-path",
        default="data/criteo.csv",
        help="Path to a Criteo style data file. Falls back to synthetic if missing.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100000,
        help="Number of rows to load or synthesize.",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma separated model names to run, or 'all' for every model.",
    )
    parser.add_argument(
        "--output",
        default="results/",
        help="Directory where metrics, plots, reports, and weights are written.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device. One of auto, cuda, mps, or cpu.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic data even when the data file exists.",
    )
    return parser.parse_args()


def select_models(models_arg: str) -> list:
    """Resolve the models flag into an ordered list of model names.

    The order always follows MODEL_ORDER so the report and the run are stable.
    An unknown name raises a clear error rather than failing later.
    """
    if models_arg.strip().lower() == "all":
        # DLRM needs TorchRec, which only installs on Linux, so "all" skips it elsewhere.
        return [m for m in MODEL_ORDER if m != "dlrm" or torchrec_available()]

    requested = [m.strip().lower() for m in models_arg.split(",") if m.strip()]
    unknown = [m for m in requested if m not in _MODEL_CLASSES]
    if unknown:
        raise ValueError(
            f"Unknown model names {unknown}. Known names are {list(_MODEL_CLASSES)}."
        )
    # Keep the canonical order regardless of how the user listed them.
    return [m for m in MODEL_ORDER if m in requested]


def load_data(args: argparse.Namespace):
    """Load real data when available, otherwise synthesize it.

    Synthetic data is used when the synthetic flag is set, when the data file is
    missing, or when the real file yields fewer rows than the requested sample
    size. Each fallback prints a short note so the run is transparent.
    """
    use_synthetic = args.synthetic
    if not use_synthetic and not os.path.exists(args.data_path):
        print(
            f"data file {args.data_path} not found. "
            "Falling back to synthetic data."
        )
        use_synthetic = True

    if not use_synthetic:
        try:
            df = load_raw(args.data_path, sample_size=args.sample_size)
        except Exception as exc:  # noqa: BLE001 broad on purpose for a graceful fallback
            print(
                f"failed to read {args.data_path} ({exc}). "
                "Falling back to synthetic data."
            )
            use_synthetic = True
        else:
            if len(df) < args.sample_size:
                print(
                    f"data file had {len(df)} rows, fewer than the requested "
                    f"{args.sample_size}. Falling back to synthetic data."
                )
                use_synthetic = True
            else:
                print(f"loaded {len(df)} rows from {args.data_path}.")
                return df

    df = generate_synthetic(args.sample_size, seed=SEED)
    print(f"generated {len(df)} synthetic rows with seed {SEED}.")
    return df


def make_groups(test_df, n_groups: int) -> np.ndarray:
    """Synthesize impression groups for the test split for group AUC.

    Real ad systems group impressions by a user id or a query id, and GAUC ranks
    ads within each such group. The benchmark has no user or query column, so we
    derive a stable surrogate group id by hashing one categorical column into
    roughly n_groups buckets. This keeps GAUC computable while standing in for
    the real grouping key.
    """
    n = len(test_df)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    n_groups = max(1, n_groups)
    col = CAT_COLS[0]
    values = test_df[col].astype(str).to_numpy()
    groups = np.array(
        [int.from_bytes(v.encode("utf-8"), "little", signed=False) % n_groups for v in values],
        dtype=np.int64,
    )
    return groups


def count_params(model, name: str) -> int:
    """Count the trainable parameters of a model.

    Torch models report the number of elements across all parameters. The
    logistic baseline reports the size of its coefficient vector plus its
    intercept, or 0 when it fell back to a constant predictor.
    """
    if name in _TORCH_MODELS and getattr(model, "module", None) is not None:
        return int(sum(p.numel() for p in model.module.parameters()))
    clf = getattr(model, "clf", None)
    if clf is not None and hasattr(clf, "coef_"):
        return int(clf.coef_.size + clf.intercept_.size)
    return 0


def logistic_feature_importance(model, n_numerical: int, n_top: int = 15) -> list:
    """Extract the top logistic regression coefficients by absolute magnitude.

    The logistic baseline stacks the numerical block before the per field
    frequency block, so the first n_numerical coefficients name numerical
    features and the rest name the categorical frequency fields. Returns a list
    of (feature_name, coefficient) tuples sorted by absolute magnitude.
    """
    clf = getattr(model, "clf", None)
    if clf is None or not hasattr(clf, "coef_"):
        return []

    coefs = np.asarray(clf.coef_).reshape(-1)
    names = []
    for i in range(len(coefs)):
        if i < n_numerical:
            names.append(f"num_{i}")
        else:
            names.append(f"freq_{CAT_COLS[i - n_numerical]}")

    order = np.argsort(np.abs(coefs))[::-1][:n_top]
    return [(names[i], float(coefs[i])) for i in order]


def print_summary_table(results: list) -> None:
    """Print a simple aligned summary table of the results to stdout."""
    headers = ["Model", "AUC", "LogLoss", "NE", "RelaImpr", "GAUC", "ECE", "Train s", "Params"]
    rows = []
    ordered = sorted(results, key=lambda r: r.get("auc", 0.0), reverse=True)
    for r in ordered:
        rows.append(
            [
                str(r.get("name", "")),
                f"{r.get('auc', float('nan')):.4f}",
                f"{r.get('logloss', float('nan')):.4f}",
                f"{r.get('ne', float('nan')):.4f}",
                f"{r.get('relaimpr', float('nan')):.4f}",
                "n/a" if r.get("gauc") is None else f"{r['gauc']:.4f}",
                f"{r.get('ece', float('nan')):.4f}",
                f"{r.get('train_time', float('nan')):.2f}",
                str(r.get("n_params", 0)),
            ]
        )

    widths = [
        max(len(headers[c]), *(len(row[c]) for row in rows)) if rows else len(headers[c])
        for c in range(len(headers))
    ]
    line = "  ".join(h.ljust(widths[c]) for c, h in enumerate(headers))
    print()
    print(line)
    print("  ".join("-" * widths[c] for c in range(len(headers))))
    for row in rows:
        print("  ".join(row[c].ljust(widths[c]) for c in range(len(headers))))
    print()


def main() -> None:
    """Run the benchmark end to end."""
    args = parse_args()
    set_seed(SEED)
    device = get_device(args.device)
    print(f"using device {device}.")

    os.makedirs(args.output, exist_ok=True)

    # Load data and split it temporally.
    df = load_data(args)
    train_df, val_df, test_df = temporal_split(df)
    print(
        f"split sizes train={len(train_df)} val={len(val_df)} test={len(test_df)}."
    )

    # Build the featurized datasets and the field metadata.
    train_ds, val_ds, test_ds, meta = build_datasets(train_df, val_df, test_df)
    y_test = np.asarray(test_ds.label, dtype=np.float64).reshape(-1)

    # Synthesize impression groups for group AUC. Roughly N over 20 groups.
    groups = make_groups(test_df, n_groups=max(1, len(test_df) // 20))

    selected = select_models(args.models)
    print(f"running models {selected}.")

    results = []
    curves = {}
    feature_importance = None
    fitted_models = {}

    for name in selected:
        print(f"\n=== {name} ===")
        config = get_config(name)
        model = _MODEL_CLASSES[name]()

        start = time.time()
        model.fit(train_ds, val_ds, meta, config)
        train_time = time.time() - start

        preds = np.asarray(model.predict_proba(test_ds), dtype=np.float64).reshape(-1)

        metrics = compute_all(y_test, preds, groups=groups)
        ece = expected_calibration_error(y_test, preds, n_bins=10)
        n_params = count_params(model, name)

        record = {
            "name": model.get_name(),
            "auc": float(metrics["auc"]),
            "logloss": float(metrics["logloss"]),
            "ne": float(metrics["ne"]),
            "relaimpr": float(metrics["relaimpr"]),
            "gauc": float(metrics["gauc"]) if "gauc" in metrics else None,
            "ece": float(ece),
            "train_time": float(train_time),
            "n_params": int(n_params),
        }
        results.append(record)
        fitted_models[name] = model

        # Calibration curve data for the overlay plot.
        mean_pred, mean_true, _counts = calibration_curve_data(y_test, preds, n_bins=10)
        curves[model.get_name()] = (mean_pred, mean_true)

        # Feature importance from the logistic baseline only.
        if name == "logistic":
            feature_importance = logistic_feature_importance(model, meta.n_numerical)

        print(
            f"{model.get_name()}: auc={record['auc']:.4f} "
            f"logloss={record['logloss']:.4f} ne={record['ne']:.4f} "
            f"ece={record['ece']:.4f} train_s={record['train_time']:.2f}"
        )

    # Write the calibration overlay plot.
    calibration_png = os.path.join(args.output, "calibration.png")
    plot_calibration(curves, calibration_png)
    print(f"\nwrote {calibration_png}.")

    # Dump the metrics json so the report can be regenerated standalone.
    metrics_json = os.path.join(args.output, "metrics.json")
    payload = {
        "results": results,
        "feature_importance": feature_importance,
        "calibration_png": calibration_png,
    }
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {metrics_json}.")

    # Generate the markdown report.
    report_path = generate_report(
        results,
        args.output,
        calibration_png=calibration_png,
        feature_importance=feature_importance,
    )
    print(f"wrote {report_path}.")

    # Save torch state dicts.
    for name in selected:
        if name not in _TORCH_MODELS:
            continue
        model = fitted_models[name]
        module = getattr(model, "module", None)
        if module is None:
            continue
        import torch

        out_path = os.path.join(args.output, f"{model.get_name()}.pt")
        torch.save(module.state_dict(), out_path)
        print(f"saved {out_path}.")

    # Print the final summary table.
    print_summary_table(results)
    print("benchmark complete.")


if __name__ == "__main__":
    main()
