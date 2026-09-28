#!/usr/bin/env python
"""Inference optimization benchmark for the ClickRank rankers.

A trained ranking model is only useful in production if it can score traffic
within a tight latency budget, so the model has to leave the training framework
and run on an inference optimized runtime. This script takes a trained
checkpoint, exports it once to ONNX, and then sweeps every serving backend this
project supports across a grid of model, runtime, precision, and batch size, on
the exact same held out test rows.

The runtimes are these.

- Raw PyTorch eager mode on the cpu. The reference. What the training code runs
  and what every accuracy delta in this report is measured against.
- Raw PyTorch eager mode on cuda, at fp32 and at fp16 through autocast.
- ONNX Runtime on the cpu execution provider. The open standard graph runtime.
- ONNX Runtime on the cuda execution provider.
- ONNX Runtime on the TensorRT execution provider, at fp16 and at int8.
- A natively built TensorRT engine, at fp32, fp16, and int8, driven through the
  TensorRT 10 tensor addressing api on an explicit cuda stream.
- OpenVINO on the cpu. Intel's inference toolkit, which reads the same ONNX
  graph and compiles it for the host cpu.

Two of the batch sizes in the default sweep matter more than the rest. Batch one
is the online serving case, where a single ad request has to be scored inside a
few milliseconds and there is no other work to hide latency behind. Batch four
thousand and ninety six is the offline scoring case, where the whole candidate
pool is scored in bulk and the only number anyone cares about is throughput. A
runtime can win one of those and lose the other, which is why the report shows
the whole curve rather than a single figure.

Two rules govern how the numbers are presented.

The first is that a cpu number and a gpu number are never blended. OpenVINO and
the ONNX Runtime cpu provider are the cpu lane. Everything with cuda or TensorRT
in its name is the gpu lane. They answer different questions and comparing them
directly would be meaningless, so every row carries the hardware it ran on and
the lanes are reported separately.

The second is that a measurement that did not happen is reported as not
available with the reason, never as a zero, an estimate, or a blank. On a
machine with no NVIDIA gpu every gpu row in this report reads not available and
the report says which piece was missing. That is the expected output on the
Apple Silicon laptop this project is developed on.

Timing on the gpu is synchronized. Every enqueue on a cuda stream returns
immediately, so a loop that does not synchronize measures how fast python fills
a queue rather than how fast the device empties it. Every gpu backend here
synchronizes before it returns a result.

Run from the repository root.
    python scripts/run_inference_benchmark.py
    python scripts/run_inference_benchmark.py --sample-size 200000 --batch-size 1024
    python scripts/run_inference_benchmark.py --models deepfm dcn --batch-sizes 1 256 4096
    python scripts/run_inference_benchmark.py --synthetic --sample-size 20000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Insert the repository root onto sys.path so that "import src" works when this
# script is run directly from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

# Select the Agg backend before importing pyplot so the charts render without a
# display, which keeps the script headless friendly on servers and in CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  import after backend selection
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

from src.data.loader import generate_synthetic, load_raw  # noqa: E402
from src.data.preprocess import build_datasets  # noqa: E402
from src.data.split import temporal_split  # noqa: E402
from src.inference.analysis import (  # noqa: E402
    achieved_bandwidth_gb_s,
    achieved_gflops,
    bound_verdict,
    cost_model_markdown,
    module_cost_model,
    speedup_prediction_check,
)
from src.inference.backends import (  # noqa: E402
    BackendContext,
    BackendResult,
    default_specs,
    probe_backends,
)

# Re-exported so the tests and any caller that imports this script by path keep
# finding the three single backend helpers the first version of it defined. The
# implementations moved into the registry, the names stayed here.
from src.inference.backends import (  # noqa: E402,F401  re-export on purpose
    make_onnx_runner,
    make_openvino_runner,
    make_torch_runner,
)
from src.inference.calibrator import calibration_cache_path  # noqa: E402
from src.inference.common import NOT_AVAILABLE, fmt, human_bytes, jsonable, sigmoid  # noqa: E402
from src.inference.export import (  # noqa: E402
    SUPPORTED_MODELS,
    checkpoint_path_for,
    display_name,
    export_onnx,
    load_or_train_module,
    make_batches,
    model_size_bytes,
    onnx_path_for,
)
from src.inference.hardware import (  # noqa: E402
    collect_hardware_record,
    cpu_lane_label,
    gpu_lane_label,
    gpu_unavailable_reason,
    hardware_markdown_lines,
    print_hardware_record,
)
from src.inference.power import PowerSampler, unavailable_power_record  # noqa: E402
from src.inference.trt_builder import ENGINE_DIR, engine_path_for  # noqa: E402
from src.inference.trt_runner import empty_profile  # noqa: E402
from src.train.config import SEED, get_config  # noqa: E402
from src.train.trainer import set_seed  # noqa: E402

# Kept so the older tests that import this script by path keep working. The
# implementation now lives in src.inference.common alongside every other backend.
_sigmoid = sigmoid


def parse_args() -> argparse.Namespace:
    """Parse the command line flags for the inference benchmark."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ranking model inference across PyTorch, ONNX Runtime, "
            "OpenVINO, and TensorRT, over a model by runtime by precision by "
            "batch size sweep."
        )
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
        help="Number of rows to load or synthesize before the temporal split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help=(
            "The headline batch size. It is always included in the sweep and it "
            "is the batch the summary table and the bar chart report."
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 32, 256, 1024, 4096],
        help=(
            "Batch sizes to sweep. Batch one is the online serving latency case "
            "and the largest batch is the offline scoring throughput case."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["deepfm"],
        choices=list(SUPPORTED_MODELS),
        help="Which models to benchmark. Both are DLRM shaped rankers.",
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=["fp32", "fp16", "int8"],
        choices=["fp32", "fp16", "int8"],
        help="Which precisions to include in the sweep. The cpu lane is always included.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Number of timed passes over the timing batches, for a stable p99.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup batches run before timing each backend.",
    )
    parser.add_argument(
        "--timing-batches",
        type=int,
        default=32,
        help=(
            "How many batches each timed pass covers. Accuracy always uses every "
            "test row, this flag only bounds the timing loop."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help=(
            "Path to the state dict of the first model listed. Defaults to "
            "results/<Model>.pt and is trained and saved there if missing."
        ),
    )
    parser.add_argument(
        "--output",
        default="results/",
        help="Directory where the report, charts, and ONNX graphs are written.",
    )
    parser.add_argument(
        "--engine-dir",
        default=ENGINE_DIR,
        help="Directory holding serialized TensorRT engines and calibration caches.",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=0,
        help=(
            "Largest batch the TensorRT engines were built for. Defaults to the "
            "largest batch in the sweep."
        ),
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Which gpu to use for every cuda and TensorRT backend.",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=20,
        help="Iterations of the TensorRT layer profiler used for the time breakdown.",
    )
    parser.add_argument(
        "--no-power",
        action="store_true",
        help="Skip the NVML power sampling even when a gpu is present.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic data even when the data file exists.",
    )
    return parser.parse_args()


def load_dataframe(args: argparse.Namespace):
    """Load real Criteo data when available, otherwise synthesize it.

    This mirrors the data loading in run_benchmark.py so the inference benchmark
    sees the same rows. Synthetic data is used when the synthetic flag is set,
    when the file is missing, or when the file yields fewer rows than requested.
    """
    use_synthetic = args.synthetic
    if not use_synthetic and not os.path.exists(args.data_path):
        print(f"data file {args.data_path} not found. Falling back to synthetic data.")
        use_synthetic = True

    if not use_synthetic:
        try:
            df = load_raw(args.data_path, sample_size=args.sample_size)
        except Exception as exc:  # noqa: BLE001 broad on purpose for a graceful fallback
            print(f"failed to read {args.data_path} ({exc}). Falling back to synthetic data.")
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


def build_module(meta, config: dict, checkpoint: str, train_ds, val_ds) -> torch.nn.Module:
    """Return an eval mode DeepFM module on the cpu with trained weights.

    Kept at this name and signature because the earlier version of this script
    exposed it. The implementation now lives in src.inference.export so the
    engine builder and the benchmark cannot drift apart.
    """
    return load_or_train_module("deepfm", meta, config, checkpoint, train_ds, val_ds)


def predict_all(
    run_batch: Callable[[np.ndarray, np.ndarray], np.ndarray],
    batches: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Score every test batch once and return the predictions in row order.

    This pass is untimed. Its only job is to produce the probabilities the
    accuracy metrics and the correctness gate are computed from, over every test
    row rather than over the bounded subset the timing loop uses.
    """
    preds: List[np.ndarray] = []
    for numerical, cat in batches:
        out = run_batch(numerical, cat)
        preds.append(np.asarray(out, dtype=np.float64).reshape(-1))
    if not preds:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(preds, axis=0)


def time_backend(
    run_batch: Callable[[np.ndarray, np.ndarray], np.ndarray],
    batches: Sequence[Tuple[np.ndarray, np.ndarray]],
    warmup: int,
    repeats: int,
    sync: Optional[Callable[[], None]] = None,
) -> np.ndarray:
    """Time a backend over a bounded set of batches and return the latencies.

    The backend is warmed up first so one time setup like lazy graph
    compilation, kernel autotuning, and allocator warmup does not pollute the
    timings. The device is then synchronized once before the clock starts so no
    warmup work is still in flight when the first batch is timed.

    Every gpu backend synchronizes inside its own run function, so the interval
    measured around each call is completed work rather than queue depth. The
    extra sync here is the belt to that pair of braces.

    Returns one latency in milliseconds per timed batch.
    """
    if not batches:
        return np.zeros((0,), dtype=np.float64)

    for i in range(max(0, warmup)):
        numerical, cat = batches[i % len(batches)]
        run_batch(numerical, cat)
    if sync is not None:
        sync()

    latencies: List[float] = []
    for _ in range(max(1, repeats)):
        for numerical, cat in batches:
            start = time.perf_counter()
            run_batch(numerical, cat)
            latencies.append((time.perf_counter() - start) * 1000.0)
    return np.asarray(latencies, dtype=np.float64)


def accuracy_metrics(y_true: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """Return AUC and LogLoss for one set of predictions, or nan when undefined."""
    result = {"auc": float("nan"), "logloss": float("nan")}
    if len(preds) != len(y_true) or len(y_true) == 0:
        return result
    if len(np.unique(y_true)) > 1:
        result["auc"] = float(roc_auc_score(y_true, preds))
    clipped = np.clip(preds, 1e-7, 1.0 - 1e-7)
    try:
        result["logloss"] = float(log_loss(y_true, clipped, labels=[0, 1]))
    except ValueError:
        result["logloss"] = float("nan")
    return result


def drift_metrics(preds: np.ndarray, reference: Optional[np.ndarray]) -> Dict[str, Any]:
    """Compare one backend's probabilities against the PyTorch fp32 reference.

    This is the correctness gate. A matching AUC is a weak claim because AUC
    only depends on the ordering, so two backends can agree on AUC to four
    decimals while one of them has shifted every probability by a percent, which
    would wreck the calibration that ad pricing depends on. The mean absolute
    difference of the probabilities is the number that actually says how far a
    backend drifted.
    """
    if reference is None or len(preds) != len(reference) or len(preds) == 0:
        return {
            "mean_abs_diff": None,
            "max_abs_diff": None,
            "note": "no reference predictions were available for this cell",
        }
    diff = np.abs(np.asarray(preds, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
    return {
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "note": "",
    }


def resolve_engine_paths(model_name: str, engine_dir: str, max_batch: int) -> Dict[str, str]:
    """Return the engine path per precision for one model, existing or not.

    The paths are returned whether or not the files exist, because the backend
    registry needs the path to name in its not available message when the engine
    is missing.
    """
    pretty = display_name(model_name)
    return {
        precision: engine_path_for(pretty, precision, max_batch, engine_dir)
        for precision in ("fp32", "fp16", "int8")
    }


def load_build_report(engine_dir: str) -> Dict[str, Any]:
    """Read the engine build report, or return an empty record when absent."""
    path = os.path.join(engine_dir, "build_report.json")
    if not os.path.exists(path):
        return {"engines": [], "reason": f"no build report at {path}"}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001 a corrupt report must not end the sweep
        return {"engines": [], "reason": f"the build report at {path} could not be read ({exc})"}


def build_seconds_for(
    build_report: Dict[str, Any], model_name: str, precision: str
) -> Optional[float]:
    """Look one engine build time up in the build report."""
    pretty = display_name(model_name)
    for record in build_report.get("engines", []):
        if record.get("model") == pretty and record.get("precision") == precision:
            return record.get("build_seconds")
    return None


def measure_cell(
    result: BackendResult,
    model_name: str,
    batch_size: int,
    test_ds,
    y_true: np.ndarray,
    reference: Optional[np.ndarray],
    args: argparse.Namespace,
    sizes: Dict[str, Any],
    build_report: Dict[str, Any],
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Run one model, runtime, precision, batch size cell and return its record.

    Two passes happen here. The accuracy pass scores every test row once,
    untimed, so AUC, LogLoss, and the drift against the reference are computed
    on the full held out split. The timing pass then runs a bounded set of
    batches many times so the latency percentiles are stable without making the
    sweep take all afternoon.

    Power is sampled around the timing pass only, so the joules reported belong
    to a region of known work rather than to the whole cell.
    """
    spec = result.spec
    batches = make_batches(test_ds, batch_size)
    # A final short batch would be timed as if it were a full one, which would
    # report a latency for a batch width the runtime never saw. Full width
    # batches are preferred, and when the test split cannot fill even one the
    # width that actually ran is recorded and the report says so.
    full_width = [b for b in batches if len(b[0]) == batch_size]
    timed = (full_width or batches)[: max(1, args.timing_batches)]
    effective_batch = int(len(timed[0][0])) if timed else 0

    preds = predict_all(result.runner, batches)
    metrics = accuracy_metrics(y_true, preds)
    drift = drift_metrics(preds, reference)

    if spec.is_gpu:
        try:
            torch.cuda.reset_peak_memory_stats(args.device_index)
        except Exception:  # noqa: BLE001 the counter is a convenience, not a gate
            pass

    sampler = None
    if spec.is_gpu and not args.no_power:
        sampler = PowerSampler(device_index=args.device_index).start()

    latencies = time_backend(
        result.runner, timed, args.warmup, args.repeats, result.sync
    )

    rows_timed = int(sum(len(b[0]) for b in timed)) * max(1, args.repeats)
    if sampler is not None:
        sampler.stop()
        power = sampler.summarize(n_inferences=rows_timed)
    elif spec.is_gpu:
        power = unavailable_power_record(
            "power sampling was disabled for this run with the no power flag"
        )
    else:
        power = unavailable_power_record(
            "power sampling goes through NVML, which reports gpu power only, so "
            "there is no reading for a cpu lane backend"
        )

    total_seconds = float(latencies.sum()) / 1000.0
    throughput = rows_timed / total_seconds if total_seconds > 0 else float("nan")

    peak_gpu_memory = power.get("peak_memory_used_bytes")
    if spec.is_gpu and peak_gpu_memory is None:
        try:
            peak_gpu_memory = int(torch.cuda.max_memory_allocated(args.device_index))
        except Exception:  # noqa: BLE001
            peak_gpu_memory = None

    record = {
        "model": display_name(model_name),
        "backend_key": spec.key,
        "label": spec.label,
        "short_label": spec.short_label,
        "runtime": spec.runtime,
        "precision": spec.precision,
        "device": spec.device,
        "lane": spec.lane,
        "batch_size": int(batch_size),
        "effective_batch_size": effective_batch,
        "n_test_rows": int(len(y_true)),
        "n_timed_batches": len(timed),
        "n_rows_timed": rows_timed,
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "mean_ms": float(latencies.mean()) if latencies.size else float("nan"),
        "p50_ms": float(np.percentile(latencies, 50)) if latencies.size else float("nan"),
        "p99_ms": float(np.percentile(latencies, 99)) if latencies.size else float("nan"),
        "min_ms": float(latencies.min()) if latencies.size else float("nan"),
        "throughput_samples_per_s": throughput,
        "auc": metrics["auc"],
        "logloss": metrics["logloss"],
        "mean_abs_diff_vs_reference": drift["mean_abs_diff"],
        "max_abs_diff_vs_reference": drift["max_abs_diff"],
        "model_size_bytes": sizes.get("checkpoint_bytes"),
        "onnx_size_bytes": sizes.get("onnx_bytes"),
        "engine_size_bytes": result.extra.get("engine_bytes"),
        "engine_build_seconds": build_seconds_for(build_report, model_name, spec.precision)
        if spec.runtime == "tensorrt"
        else None,
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "power": power,
        "providers": result.extra.get("providers"),
    }
    return record, preds


def collect_layer_profile(
    results: Dict[str, BackendResult],
    test_ds,
    batch_size: int,
    iterations: int,
) -> Dict[str, Any]:
    """Profile the native TensorRT engine layer by layer, when there is one.

    The profile is the evidence for the memory bound argument. It says how much
    of the engine's time goes into the embedding gather and how much into the
    perceptron, which is the only way to settle that question by measurement
    rather than by assertion.
    """
    for key in ("tensorrt-fp16", "tensorrt-fp32", "tensorrt-int8"):
        result = results.get(key)
        if result is None or not result.available:
            continue
        runner = result.extra.get("trt_runner")
        if runner is None:
            continue
        batches = make_batches(test_ds, batch_size)
        if not batches:
            continue
        numerical, cat = batches[0]
        try:
            profile = runner.profile_layers(numerical, cat, iterations=iterations)
        except Exception as exc:  # noqa: BLE001 a failed profile is a missing row, not a crash
            return empty_profile(f"the TensorRT layer profiler raised {exc}")
        profile["backend_key"] = key
        profile["reason"] = ""
        return profile
    return empty_profile(
        "no native TensorRT engine ran on this host, so there is no per layer "
        "time breakdown to report"
    )


def main() -> None:
    """Run the inference benchmark end to end."""
    args = parse_args()
    set_seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.engine_dir, exist_ok=True)

    batch_sizes = sorted({int(b) for b in args.batch_sizes} | {int(args.batch_size)})
    max_batch = int(args.max_batch) if args.max_batch else max(batch_sizes)

    hardware = collect_hardware_record(args.device_index)
    print_hardware_record(hardware)
    print()
    print(
        "backends are grouped into two lanes. The cpu lane is PyTorch eager, "
        "ONNX Runtime on the cpu provider, and OpenVINO. The gpu lane is "
        "everything with cuda or TensorRT in its name. The two lanes are not "
        "compared against each other."
    )
    print()

    # Data and features. Same temporal split and same feature pipeline as
    # run_benchmark.py, so the test rows match the main benchmark exactly.
    df = load_dataframe(args)
    train_df, val_df, test_df = temporal_split(df)
    train_ds, val_ds, test_ds, meta = build_datasets(train_df, val_df, test_df)
    y_test = np.asarray(test_ds.label, dtype=np.float64).reshape(-1)
    print(f"test split has {len(y_test)} rows.")

    build_report = load_build_report(args.engine_dir)

    rows: List[Dict[str, Any]] = []
    unavailable: List[Dict[str, Any]] = []
    profiles: Dict[str, Any] = {}
    cost_models: Dict[str, Any] = {}
    model_records: Dict[str, Any] = {}

    for index, model_name in enumerate(args.models):
        pretty = display_name(model_name)
        print(f"\n=== {pretty} ===")
        config = get_config(model_name)
        checkpoint = (
            args.checkpoint
            if (args.checkpoint and index == 0)
            else checkpoint_path_for(model_name, args.output)
        )
        module = load_or_train_module(
            model_name, meta, config, checkpoint, train_ds, val_ds
        )

        onnx_path = onnx_path_for(model_name, args.output)
        export_onnx(module, meta, args.batch_size, onnx_path)

        sizes = {
            "checkpoint_bytes": os.path.getsize(checkpoint)
            if os.path.exists(checkpoint)
            else model_size_bytes(module),
            "onnx_bytes": os.path.getsize(onnx_path) if os.path.exists(onnx_path) else None,
            "parameter_bytes": model_size_bytes(module),
        }
        model_records[pretty] = {
            "config": config,
            "sizes": sizes,
            "onnx_path": onnx_path,
            "checkpoint": checkpoint,
            "n_parameters": int(sum(p.numel() for p in module.parameters())),
        }

        ctx = BackendContext(
            model_name=pretty,
            module=module,
            onnx_path=onnx_path,
            n_numerical=meta.n_numerical,
            n_embed_fields=meta.n_embed_fields,
            max_batch=max_batch,
            engine_paths=resolve_engine_paths(model_name, args.engine_dir, max_batch),
            calibration_cache=calibration_cache_path(pretty, args.engine_dir),
            trt_engine_cache_dir=os.path.join(args.engine_dir, "ort_trt_cache"),
            device_index=args.device_index,
        )

        print("probing backends.")
        specs = default_specs(args.precisions)
        probed = probe_backends(ctx, specs)
        by_key = {r.spec.key: r for r in probed}

        for result in probed:
            if not result.available:
                unavailable.append(
                    {
                        "model": pretty,
                        "key": result.spec.key,
                        "label": result.spec.label,
                        "lane": result.spec.lane,
                        "precision": result.spec.precision,
                        "reason": result.note,
                    }
                )

        # The reference predictions are eager PyTorch fp32 on the cpu, recomputed
        # per batch size so a drift comparison is always against a reference that
        # scored the identical rows in the identical order.
        for batch_size in batch_sizes:
            reference: Optional[np.ndarray] = None
            reference_result = by_key.get("pytorch-cpu-fp32")
            if reference_result is not None and reference_result.available:
                reference = predict_all(
                    reference_result.runner, make_batches(test_ds, batch_size)
                )

            for result in probed:
                if not result.available:
                    continue
                record, _preds = measure_cell(
                    result,
                    model_name,
                    batch_size,
                    test_ds,
                    y_test,
                    reference,
                    args,
                    sizes,
                    build_report,
                )
                rows.append(record)
                print(
                    f"  {record['label']} at batch {batch_size}. "
                    f"mean {fmt(record['mean_ms'])} ms, "
                    f"p99 {fmt(record['p99_ms'])} ms, "
                    f"{fmt(record['throughput_samples_per_s'], ',.0f')} samples/s, "
                    f"auc {fmt(record['auc'], '.4f')}"
                )

        profiles[pretty] = collect_layer_profile(
            by_key, test_ds, min(args.batch_size, max_batch), args.profile_iters
        )
        cost_models[pretty] = {
            str(batch_size): module_cost_model(
                module,
                meta.n_embed_fields,
                config["embed_dim"],
                meta.n_numerical,
                batch_size,
            )
            for batch_size in batch_sizes
        }

    if not rows:
        print("no backend ran on this host, so there is nothing to report.")
        return

    print_summary_table(rows, args.batch_size)

    payload = {
        "hardware": hardware,
        "arguments": vars(args),
        "seed": SEED,
        "n_test_rows": int(len(y_test)),
        "batch_sizes": batch_sizes,
        "models": model_records,
        "measurements": rows,
        "unavailable_backends": unavailable,
        "tensorrt_layer_profiles": profiles,
        "cost_models": cost_models,
        "engine_build_report": build_report,
    }

    json_path = os.path.join(args.output, "inference_benchmark.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2)

    chart_paths = write_charts(rows, args)
    report_path = os.path.join(args.output, "inference_benchmark.md")
    write_report(
        payload, report_path, chart_paths, args, hardware, profiles, cost_models
    )

    print(f"\nwrote {json_path}.")
    for path in chart_paths.values():
        print(f"wrote {path}.")
    print(f"wrote {report_path}.")
    print("inference benchmark complete.")


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _sentence_case(text: str) -> str:
    """Uppercase only the first character of a sentence.

    The built in capitalize lowercases everything after the first character,
    which turns macOS into macos and NVIDIA into nvidia. Product names in a
    report have to survive being placed at the start of a sentence.
    """
    if not text:
        return NOT_AVAILABLE
    return text[0].upper() + text[1:]


def _spec_order() -> Dict[str, int]:
    """Return the canonical display order of every backend key."""
    return {spec.key: i for i, spec in enumerate(default_specs())}


def _select(rows: List[Dict[str, Any]], **filters) -> List[Dict[str, Any]]:
    """Return the rows whose fields all match the given filters."""
    out = []
    for row in rows:
        if all(row.get(k) == v for k, v in filters.items()):
            out.append(row)
    return out


def print_summary_table(rows: List[Dict[str, Any]], batch_size: int) -> None:
    """Print the headline table for the primary batch size to stdout."""
    selected = _select(rows, batch_size=int(batch_size))
    headers = [
        "Model",
        "Backend",
        "Lane",
        "Latency (ms/batch)",
        "p50 (ms)",
        "p99 (ms)",
        "Throughput (samples/s)",
        "AUC",
    ]
    table = []
    for row in selected:
        table.append(
            [
                row["model"],
                row["label"],
                row["lane"],
                fmt(row["mean_ms"]),
                fmt(row["p50_ms"]),
                fmt(row["p99_ms"]),
                fmt(row["throughput_samples_per_s"], ",.0f"),
                fmt(row["auc"], ".4f"),
            ]
        )
    if not table:
        return
    widths = [
        max(len(headers[c]), *(len(r[c]) for r in table)) for c in range(len(headers))
    ]
    print()
    print(f"summary at batch size {batch_size}")
    print("  ".join(h.ljust(widths[c]) for c, h in enumerate(headers)))
    print("  ".join("-" * widths[c] for c in range(len(headers))))
    for r in table:
        print("  ".join(r[c].ljust(widths[c]) for c in range(len(headers))))
    print()


def write_charts(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, str]:
    """Write every chart and return a mapping from chart key to path."""
    paths = {
        "bar": os.path.join(args.output, "inference_latency.png"),
        "latency": os.path.join(args.output, "inference_latency_vs_batch.png"),
        "throughput": os.path.join(args.output, "inference_throughput_vs_batch.png"),
        "accuracy": os.path.join(args.output, "inference_accuracy_vs_precision.png"),
    }
    plot_latency_bar(rows, args.batch_size, paths["bar"])
    plot_vs_batch(
        rows,
        "mean_ms",
        "Latency (ms per batch)",
        "Inference latency against batch size (log log, lower is better)",
        paths["latency"],
        log_y=True,
    )
    plot_vs_batch(
        rows,
        "throughput_samples_per_s",
        "Throughput (samples per second)",
        "Inference throughput against batch size (higher is better)",
        paths["throughput"],
        log_y=True,
    )
    plot_accuracy_vs_precision(rows, args.batch_size, paths["accuracy"])
    return paths


def plot_latency_bar(rows: List[Dict[str, Any]], batch_size: int, chart_path: str) -> None:
    """Save the bar chart of mean latency per backend at the headline batch size.

    This is the chart the README has always pointed at, so it keeps its name and
    its shape. Only backends that ran appear on it.
    """
    selected = _select(rows, batch_size=int(batch_size))
    if not selected:
        return
    labels = [f"{r['short_label']}\n({r['model']})" for r in selected]
    latencies = [r["mean_ms"] for r in selected]
    palette = ["#ee4c2c", "#5b8def", "#00b3a4", "#76b900", "#f2a900", "#8e5bd8"]
    colors = [palette[i % len(palette)] for i in range(len(selected))]

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(selected)), 4.8))
    bars = ax.bar(labels, latencies, color=colors, width=0.6)
    ax.set_ylabel("Latency (ms per batch)")
    ax.set_title(
        f"Inference latency by backend at batch {batch_size} (lower is better)"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, value in zip(bars, latencies):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)


def plot_vs_batch(
    rows: List[Dict[str, Any]],
    field: str,
    ylabel: str,
    title: str,
    chart_path: str,
    log_y: bool = True,
) -> None:
    """Save a line per backend of one metric against batch size on a log x axis."""
    if not rows:
        return
    series: Dict[str, List[Tuple[int, float]]] = {}
    for row in rows:
        value = row.get(field)
        if value is None or not np.isfinite(value):
            continue
        key = f"{row['model']} {row['short_label']}"
        series.setdefault(key, []).append((int(row["batch_size"]), float(value)))
    if not series:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for key, points in sorted(series.items()):
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", label=key)
    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Batch size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)


def plot_accuracy_vs_precision(
    rows: List[Dict[str, Any]], batch_size: int, chart_path: str
) -> None:
    """Save the AUC against precision chart at the headline batch size.

    On a host where only the fp32 cpu lane ran this chart holds a single point
    per runtime, which is the honest picture rather than an empty file.
    """
    selected = _select(rows, batch_size=int(batch_size))
    if not selected:
        return
    order = ["fp32", "fp16", "int8"]
    series: Dict[str, Dict[str, float]] = {}
    for row in selected:
        auc = row.get("auc")
        if auc is None or not np.isfinite(auc):
            continue
        key = f"{row['model']} {row['runtime']} ({row['device']})"
        series.setdefault(key, {})[row["precision"]] = float(auc)
    if not series:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for key, values in sorted(series.items()):
        xs = [order.index(p) for p in order if p in values]
        ys = [values[p] for p in order if p in values]
        ax.plot(xs, ys, marker="o", linestyle="-" if len(xs) > 1 else "None", label=key)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel("Engine precision")
    ax.set_ylabel("Test AUC")
    ax.set_title(f"Test AUC against precision at batch {batch_size}")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)


def _power_cell(power: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return the mean power, peak power, and inferences per joule cells."""
    if not power or not power.get("available"):
        return NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE
    return (
        fmt(power.get("mean_watts"), ".1f"),
        fmt(power.get("peak_watts"), ".1f"),
        fmt(power.get("inferences_per_joule"), ",.0f"),
    )


def write_report(
    payload: Dict[str, Any],
    report_path: str,
    chart_paths: Dict[str, str],
    args: argparse.Namespace,
    hardware: Dict[str, Any],
    profiles: Dict[str, Any],
    cost_models: Dict[str, Any],
) -> None:
    """Write the full markdown report.

    The prose follows the project style. No em dashes, no semicolons, and no mid
    sentence colons. Every table cell for something that did not run says not
    available, and every section that reports a number names the hardware the
    number came from.
    """
    rows: List[Dict[str, Any]] = payload["measurements"]
    unavailable: List[Dict[str, Any]] = payload["unavailable_backends"]
    batch_sizes: List[int] = payload["batch_sizes"]
    n_test = payload["n_test_rows"]
    cpu_label = cpu_lane_label(hardware)
    gpu_label = gpu_lane_label(hardware)
    gpu_reason = gpu_unavailable_reason(hardware)

    lines: List[str] = []
    lines.append("# Inference Optimization Benchmark")
    lines.append("")
    lines.append(
        "This report compares every serving backend this project supports, "
        "running the same trained weights on the same held out test rows. The "
        "question it answers is how much serving latency drops when a trained "
        "ranker is exported and run through an inference optimized runtime "
        "instead of raw PyTorch, and what that costs in accuracy when the "
        "runtime also drops the numeric precision."
    )
    lines.append("")
    lines.append(
        f"The sweep covers {', '.join(payload['models'])} across "
        f"{len(set(r['backend_key'] for r in rows))} backends that ran, at the "
        f"batch sizes {', '.join(str(b) for b in batch_sizes)}. It scores "
        f"{n_test} held out test rows per cell with seed {payload['seed']}, and "
        f"it times {args.timing_batches} batches over {args.repeats} passes "
        f"after {args.warmup} warmup batches."
    )
    lines.append("")
    lines.append(
        "Batch one and the largest batch are in the sweep for different "
        "reasons. Batch one is the online serving case, where a single ad "
        "request is scored on its own inside a few milliseconds and there is no "
        "other work to hide the latency behind. The largest batch is the offline "
        "scoring case, where a whole candidate pool is scored in bulk and the "
        "only number that matters is throughput. A runtime can win one of those "
        "and lose the other, so the report shows the full curve."
    )
    lines.append("")

    # Provenance.
    lines.append("## Hardware and software provenance")
    lines.append("")
    lines.append(
        "Every number below was produced on this machine with these library "
        "versions. An inference measurement without its hardware is not a "
        "result, so this table is embedded in the markdown report and in the "
        "json artifact next to every raw measurement."
    )
    lines.append("")
    lines.extend(hardware_markdown_lines(hardware))
    lines.append("")

    # Lanes.
    lines.append("## Two lanes, never blended")
    lines.append("")
    lines.append(
        f"The cpu lane ran on {cpu_label}. It holds PyTorch eager mode, ONNX "
        "Runtime on the cpu execution provider, and OpenVINO. OpenVINO is "
        "Intel's cpu inference toolkit and it belongs to this lane only. It is "
        "not a like for like comparison against any gpu backend and it is never "
        "reported as one, because a cpu runtime and a gpu runtime answer "
        "different deployment questions and putting their numbers in the same "
        "ranking would be meaningless."
    )
    lines.append("")
    if gpu_label == NOT_AVAILABLE:
        lines.append(
            f"The gpu lane did not run on this host. {_sentence_case(gpu_reason)}. "
            "Every gpu row in every table below therefore reads not available. "
            "Nothing was estimated, extrapolated, or filled in from another "
            "machine. To fill those rows in, run this script on a cuda host "
            "after building the engines with scripts/build_trt_engines.py."
        )
    else:
        lines.append(
            f"The gpu lane ran on {gpu_label}. It holds PyTorch eager on cuda, "
            "ONNX Runtime on the cuda and TensorRT execution providers, and the "
            "natively built TensorRT engines. Every timing in this lane is taken "
            "with the cuda stream synchronized, so it measures completed device "
            "work rather than how fast the host filled a queue."
        )
    lines.append("")

    # Headline table.
    lines.append(f"## Summary at batch {args.batch_size}")
    lines.append("")
    lines.append(
        "This is the headline table. Latency is the mean wall time per batch, "
        "p50 and p99 are the median and the tail, throughput is the steady state "
        "samples per second, and AUC and LogLoss are measured over every held "
        "out test row rather than over the timing subset."
    )
    lines.append("")
    lines.append(
        "| Model | Backend | Lane | Hardware | Latency (ms/batch) | p50 (ms) | "
        "p99 (ms) | Throughput (samples/s) | AUC | LogLoss |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in _select(rows, batch_size=int(args.batch_size)):
        hardware_cell = cpu_label if row["lane"] == "cpu" else gpu_label
        lines.append(
            f"| {row['model']} | {row['label']} | {row['lane']} | {hardware_cell} | "
            f"{fmt(row['mean_ms'])} | {fmt(row['p50_ms'])} | {fmt(row['p99_ms'])} | "
            f"{fmt(row['throughput_samples_per_s'], ',.0f')} | "
            f"{fmt(row['auc'], '.4f')} | {fmt(row['logloss'], '.4f')} |"
        )
    lines.append("")
    lines.append(f"![Inference latency by backend]({os.path.basename(chart_paths['bar'])})")
    lines.append("")

    # Unavailable backends.
    lines.append("## Backends that did not run")
    lines.append("")
    if not unavailable:
        lines.append("Every backend in the grid ran on this host.")
    else:
        lines.append(
            "Each of these was asked for and could not be built. The reason is "
            "the exact one the registry reported, not a summary of it."
        )
        lines.append("")
        lines.append("| Model | Backend | Lane | Reason |")
        lines.append("| --- | --- | --- | --- |")
        seen = set()
        for item in unavailable:
            key = (item["model"], item["key"])
            if key in seen:
                continue
            seen.add(key)
            reason = item["reason"].replace("|", " ")
            lines.append(
                f"| {item['model']} | {item['label']} | {item['lane']} | {reason} |"
            )
    lines.append("")

    # Full sweep.
    lines.append("## Full sweep")
    lines.append("")
    lines.append(
        "One row per model, backend, and batch size. Engine build time is a "
        "deploy time cost and it is reported in its own column so it is never "
        "confused with a serving number."
    )
    lines.append("")
    lines.append(
        "| Model | Backend | Precision | Lane | Batch | Rows timed per batch | "
        "Mean (ms) | p50 (ms) | p99 (ms) | Throughput (samples/s) | AUC | "
        "LogLoss | Build (s) | Peak GPU memory | Model on disk |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- | --- | --- |"
    )
    order = _spec_order()
    ragged = False
    for row in sorted(
        rows,
        key=lambda r: (r["model"], order.get(r["backend_key"], 99), r["batch_size"]),
    ):
        engine_size = row.get("engine_size_bytes") or row.get("model_size_bytes")
        effective = row.get("effective_batch_size", row["batch_size"])
        if effective != row["batch_size"]:
            ragged = True
        lines.append(
            f"| {row['model']} | {row['short_label']} | {row['precision']} | "
            f"{row['lane']} | {row['batch_size']} | {effective} | "
            f"{fmt(row['mean_ms'])} | "
            f"{fmt(row['p50_ms'])} | {fmt(row['p99_ms'])} | "
            f"{fmt(row['throughput_samples_per_s'], ',.0f')} | "
            f"{fmt(row['auc'], '.4f')} | {fmt(row['logloss'], '.4f')} | "
            f"{fmt(row['engine_build_seconds'], '.1f')} | "
            f"{human_bytes(row.get('peak_gpu_memory_bytes'))} | "
            f"{human_bytes(engine_size)} |"
        )
    lines.append("")
    if ragged:
        lines.append(
            "The rows timed per batch column is there because the test split is "
            "smaller than the largest batch in the sweep. Where the two columns "
            "differ, the latency was measured on the narrower batch that "
            "actually ran and not on the batch size that was asked for. Raise "
            "the sample size to fill those batches properly."
        )
        lines.append("")
    lines.append(
        f"![Latency against batch size]({os.path.basename(chart_paths['latency'])})"
    )
    lines.append("")
    lines.append(
        f"![Throughput against batch size]({os.path.basename(chart_paths['throughput'])})"
    )
    lines.append("")

    # Accuracy delta.
    lines.extend(_accuracy_delta_section(rows, args, chart_paths))

    # Correctness gate.
    lines.extend(_correctness_section(rows, args))

    # Power.
    lines.extend(_power_section(rows, args, hardware))

    # Analysis.
    lines.extend(
        _analysis_section(rows, args, hardware, profiles, cost_models, payload)
    )

    lines.append("")
    lines.append("## Reproducing this report")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/build_trt_engines.py")
    lines.append(
        "python scripts/run_inference_benchmark.py --models "
        + " ".join(args.models)
        + " --batch-sizes "
        + " ".join(str(b) for b in batch_sizes)
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "The engine builder is a separate step because a TensorRT plan is tied "
        "to the gpu architecture, the driver, and the TensorRT version that "
        "produced it, so it has to be built on the machine that will serve it. "
        "On a host with no gpu the builder prints what is missing and exits "
        "cleanly, and this benchmark then reports every gpu row as not "
        "available with the same reason."
    )
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _accuracy_delta_section(
    rows: List[Dict[str, Any]], args: argparse.Namespace, chart_paths: Dict[str, str]
) -> List[str]:
    """Build the accuracy delta against the fp32 reference section."""
    lines = ["## Accuracy against the fp32 reference", ""]
    lines.append(
        "Reduced precision is not free. An fp16 engine rounds every activation "
        "to half the mantissa and an int8 engine replaces the numbers entirely "
        "with a quantized approximation chosen by a calibrator. This table "
        "reports what that cost, measured against eager PyTorch fp32 on exactly "
        "the same test rows. The regression is reported whatever it is. Nothing "
        "here was tuned to make a lower precision engine look better."
    )
    lines.append("")

    selected = _select(rows, batch_size=int(args.batch_size))
    reference_by_model: Dict[str, Dict[str, Any]] = {}
    for row in selected:
        if row["backend_key"] == "pytorch-cpu-fp32":
            reference_by_model[row["model"]] = row

    lines.append(
        "| Model | Backend | Precision | AUC | AUC delta | LogLoss | LogLoss delta |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in selected:
        ref = reference_by_model.get(row["model"])
        if ref is None:
            auc_delta = NOT_AVAILABLE
            ll_delta = NOT_AVAILABLE
        else:
            auc_delta = fmt(
                (row["auc"] - ref["auc"])
                if np.isfinite(row["auc"]) and np.isfinite(ref["auc"])
                else float("nan"),
                "+.5f",
            )
            ll_delta = fmt(
                (row["logloss"] - ref["logloss"])
                if np.isfinite(row["logloss"]) and np.isfinite(ref["logloss"])
                else float("nan"),
                "+.5f",
            )
        lines.append(
            f"| {row['model']} | {row['short_label']} | {row['precision']} | "
            f"{fmt(row['auc'], '.5f')} | {auc_delta} | "
            f"{fmt(row['logloss'], '.5f')} | {ll_delta} |"
        )
    lines.append("")

    reduced = [r for r in selected if r["precision"] in ("fp16", "int8")]
    if not reduced:
        lines.append(
            "No reduced precision engine ran on this host, so there is no "
            "precision cost to report. Every row above is fp32, which means the "
            "deltas that are not exactly zero are floating point noise between "
            "runtimes rather than a quantization loss. The fp16 and int8 rows "
            "are not available for the reason given in the backends that did "
            "not run table."
        )
    else:
        worst = min(
            (r for r in reduced if np.isfinite(r["auc"])),
            key=lambda r: r["auc"],
            default=None,
        )
        if worst is not None:
            ref = reference_by_model.get(worst["model"])
            if ref is not None and np.isfinite(ref["auc"]):
                delta = worst["auc"] - ref["auc"]
                lines.append(
                    f"The largest accuracy cost on this run belongs to "
                    f"{worst['short_label']} at {worst['precision']}, which moved "
                    f"AUC by {delta:+.5f} against the fp32 reference."
                )
    lines.append("")
    lines.append(
        f"![Test AUC against precision]({os.path.basename(chart_paths['accuracy'])})"
    )
    lines.append("")
    return lines


def _correctness_section(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[str]:
    """Build the numeric drift correctness gate section."""
    lines = ["## Correctness gate", ""]
    lines.append(
        "A matching AUC is a weak claim. AUC depends only on the ordering of the "
        "predictions, so a backend could shift every probability by a full "
        "percent and still report the same AUC to four decimals, which would "
        "leave the model badly calibrated while the table looked clean. "
        "Calibration is what ad pricing multiplies a bid by, so the gate here is "
        "the mean absolute difference between each backend's probabilities and "
        "the eager PyTorch fp32 probabilities on identical rows. It says how far "
        "a backend drifted rather than asserting that it did not."
    )
    lines.append("")
    lines.append(
        "| Model | Backend | Precision | Batch | Mean abs difference | Max abs difference |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in _select(rows, batch_size=int(args.batch_size)):
        lines.append(
            f"| {row['model']} | {row['short_label']} | {row['precision']} | "
            f"{row['batch_size']} | "
            f"{fmt(row.get('mean_abs_diff_vs_reference'), '.3e')} | "
            f"{fmt(row.get('max_abs_diff_vs_reference'), '.3e')} |"
        )
    lines.append("")
    return lines


def _power_section(
    rows: List[Dict[str, Any]], args: argparse.Namespace, hardware: Dict[str, Any]
) -> List[str]:
    """Build the power and energy efficiency section."""
    lines = ["## Power and energy", ""]
    lines.append(
        "Latency says how fast. Power says what that speed costs, and across a "
        "serving fleet the second number is the one that sets the bill. Power is "
        "sampled through NVML on a background thread around the timing loop "
        "only, so the joules reported belong to a region of known work. Energy "
        "is the trapezoid integral of the sampled watt curve, and inferences per "
        "joule is the efficiency figure that compares two precisions without "
        "reference to wall clock."
    )
    lines.append("")
    lines.append(
        "| Model | Backend | Precision | Batch | Mean power (W) | Peak power (W) | "
        "Energy (J) | Inferences per joule |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in _select(rows, batch_size=int(args.batch_size)):
        power = row.get("power") or {}
        mean_w, peak_w, per_joule = _power_cell(power)
        lines.append(
            f"| {row['model']} | {row['short_label']} | {row['precision']} | "
            f"{row['batch_size']} | {mean_w} | {peak_w} | "
            f"{fmt(power.get('energy_joules'), '.2f')} | {per_joule} |"
        )
    lines.append("")
    if not hardware.get("gpu", {}).get("available"):
        lines.append(
            "Every cell in this table reads not available on this run. NVML "
            "reports gpu power and there is no gpu on this host, so no watt was "
            "ever measured. A cpu power figure is not substituted in, because a "
            "package power reading taken from a different sensor on a different "
            "device is not the same measurement and presenting it here would "
            "invite exactly the comparison this report refuses to make."
        )
        lines.append("")
    return lines


def _analysis_section(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    hardware: Dict[str, Any],
    profiles: Dict[str, Any],
    cost_models: Dict[str, Any],
    payload: Dict[str, Any],
) -> List[str]:
    """Build the memory bound against compute bound analysis section.

    The prediction this section tests is that a DLRM shaped ranker is mostly a
    large embedding table and a small perceptron, so it is bound by memory
    bandwidth rather than by arithmetic, fp16 gives far less than the usual
    factor of two, and int8 costs real accuracy for a speedup that was never
    available. The section computes the evidence and states whichever way it
    falls, including that the prediction was wrong.
    """
    lines = ["## Is this model memory bound or compute bound", ""]
    lines.append(
        "A DeepFM or a DCN is one very large embedding table followed by a very "
        "small multilayer perceptron. The multiply accumulate work in the "
        "perceptron is small, while the embedding lookup drags scattered rows "
        "out of a table tens of megabytes wide with almost no locality. If that "
        "shape dominates then the wall clock is set by memory bandwidth, a "
        "narrower multiply buys much less than the factor of two that half "
        "precision seems to promise, and int8 trades accuracy for a speedup that "
        "was never on the table. This section computes the evidence rather than "
        "asserting the conclusion, and it reports the answer the numbers give."
    )
    lines.append("")

    gpu = hardware.get("gpu", {})
    peak_bw = gpu.get("peak_memory_bandwidth_gb_s")

    for model, per_batch in cost_models.items():
        lines.append(f"### {model}")
        lines.append("")
        lines.append(
            "| Batch | FLOPs per row | Bytes per row | Gather share of bytes | "
            "Arithmetic intensity | Achieved GB/s | Achieved GFLOP/s | Verdict |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")

        verdicts: Dict[int, Dict[str, Any]] = {}
        for batch_key in sorted(per_batch, key=lambda k: int(k)):
            cost = per_batch[batch_key]
            batch = int(batch_key)
            fastest = _fastest_row(rows, model, batch)
            latency = fastest["mean_ms"] if fastest else None
            # The measured latency belongs to the batch width that actually ran,
            # which is narrower than the requested one when the test split
            # cannot fill it, so the byte and operation counts are scaled to the
            # width that was timed rather than the width that was asked for.
            timed_rows = (
                fastest.get("effective_batch_size", batch) if fastest else batch
            ) or batch
            achieved_bw = achieved_bandwidth_gb_s(
                cost["bytes_per_row"] * timed_rows, latency
            )
            achieved_fl = achieved_gflops(cost["flops_per_row"] * timed_rows, latency)
            verdict = bound_verdict(
                cost["arithmetic_intensity_flops_per_byte"],
                achieved_bw if (fastest and fastest["lane"] == "gpu") else None,
                peak_bw,
            )
            verdicts[batch] = verdict
            lines.append(
                f"| {batch} | {cost['flops_per_row']:,.0f} | "
                f"{cost['bytes_per_row']:,.0f} | "
                f"{fmt(cost['gather_share_of_bytes_pct'], '.1f')} percent | "
                f"{fmt(cost['arithmetic_intensity_flops_per_byte'], '.2f')} | "
                f"{fmt(achieved_bw, ',.1f')} | {fmt(achieved_fl, ',.1f')} | "
                f"{verdict['verdict']} |"
            )
        lines.append("")

        small = min(verdicts) if verdicts else None
        large = max(verdicts) if verdicts else None
        if small is not None and large is not None and small != large:
            lines.append(
                f"The arithmetic intensity moves with the batch size, which is "
                f"the whole story for this model. At batch {small} the "
                f"perceptron weights have to cross the memory bus for a single "
                f"row, so the intensity is "
                f"{fmt(per_batch[str(small)]['arithmetic_intensity_flops_per_byte'], '.2f')} "
                f"operations per byte and the workload is {verdicts[small]['verdict']}. "
                f"At batch {large} those same weights amortize over the whole "
                f"batch and the intensity rises to "
                f"{fmt(per_batch[str(large)]['arithmetic_intensity_flops_per_byte'], '.2f')} "
                f"operations per byte, which is {verdicts[large]['verdict']}. "
                "The online serving case and the offline scoring case therefore "
                "sit on opposite sides of the roofline, and an optimization that "
                "helps one can do nothing at all for the other."
            )
            lines.append("")

        headline = verdicts.get(int(args.batch_size)) or (
            verdicts[large] if large is not None else None
        )
        if headline is not None:
            lines.append(
                f"At the headline batch size of {args.batch_size} the reading is "
                f"{headline['verdict']}, because {headline['explanation']}."
            )
            lines.append("")
            lines.append("The cost model at that batch size in full.")
            lines.append("")
            key = str(args.batch_size) if str(args.batch_size) in per_batch else str(large)
            lines.extend(cost_model_markdown(per_batch[key], headline))
            lines.append("")

        # The fp16 speedup check.
        fp32_row = _fastest_row(rows, model, int(args.batch_size), precision="fp32", lane="gpu")
        fp16_row = _fastest_row(rows, model, int(args.batch_size), precision="fp16", lane="gpu")
        verdict_label = headline["verdict"] if headline else "inconclusive"
        check = speedup_prediction_check(
            fp32_row["mean_ms"] if fp32_row else None,
            fp16_row["mean_ms"] if fp16_row else None,
            verdict_label,
        )
        lines.append(f"On the fp16 prediction, {check['statement']}.")
        lines.append("")

        # The layer profile.
        profile = profiles.get(model) or {}
        lines.append("#### Where the time goes inside the engine")
        lines.append("")
        if profile.get("iterations"):
            lines.append(
                "The TensorRT layer profiler was attached for "
                f"{profile['iterations']} iterations at batch "
                f"{profile.get('batch_size', NOT_AVAILABLE)}. Attaching a "
                "profiler adds overhead and suppresses some fusion, so these "
                "milliseconds are not the latency the tables report. What the "
                "profile is for is the shape of the distribution, which is where "
                "the time goes rather than how much of it there is."
            )
            lines.append("")
            lines.append("| Bucket | Time per iteration (ms) | Share |")
            lines.append("| --- | --- | --- |")
            lines.append(
                f"| Embedding gather and data movement | {fmt(profile.get('gather_ms'), '.4f')} | "
                f"{fmt(profile.get('gather_share_pct'), '.1f')} percent |"
            )
            lines.append(
                f"| Matrix multiply and perceptron | {fmt(profile.get('matmul_ms'), '.4f')} | "
                f"{fmt(profile.get('matmul_share_pct'), '.1f')} percent |"
            )
            lines.append(
                f"| Everything else | {fmt(profile.get('other_ms'), '.4f')} | "
                f"{fmt(profile.get('other_share_pct'), '.1f')} percent |"
            )
            lines.append("")
            gather_share = profile.get("gather_share_pct")
            matmul_share = profile.get("matmul_share_pct")
            if gather_share is not None and matmul_share is not None:
                if gather_share > matmul_share:
                    lines.append(
                        f"The gather bucket owns {gather_share:.1f} percent of the "
                        "engine time against "
                        f"{matmul_share:.1f} percent for the perceptron, which is "
                        "direct measured support for the memory bound reading. "
                        "Time spent moving embedding rows does not shrink when "
                        "the multiply gets narrower."
                    )
                else:
                    lines.append(
                        f"The perceptron owns {matmul_share:.1f} percent of the "
                        "engine time against "
                        f"{gather_share:.1f} percent for the gather. That "
                        "contradicts the memory bound prediction for this model "
                        "at this batch size, and the measurement is what stands. "
                        "The hash bucket space this project uses keeps the "
                        "embedding tables small enough that the perceptron is the "
                        "larger cost, which is a real difference from a "
                        "production scale DLRM with tables in the tens of "
                        "gigabytes."
                    )
                lines.append("")
        else:
            lines.append(
                "No per layer profile was taken on this run. "
                f"{_sentence_case(profile.get('reason') or NOT_AVAILABLE)}. The "
                "arithmetic intensity above is computed from the module "
                "definition and does not need a gpu, so it stands on its own, "
                "but the measured split between gather time and perceptron time "
                "needs a TensorRT engine and is not available here."
            )
            lines.append("")

    if not gpu.get("available"):
        lines.append(
            "One caveat applies to this whole section on this run. The achieved "
            "bandwidth and the achieved arithmetic throughput columns are filled "
            "from the fastest backend that ran, which on this host is a cpu "
            "backend, so they are not compared against a gpu peak and the "
            "bandwidth utilization is not available. The arithmetic intensity "
            "and the ridge point comparison are properties of the model and are "
            "correct regardless of which device ran."
        )
        lines.append("")
    return lines


def _fastest_row(
    rows: List[Dict[str, Any]],
    model: str,
    batch_size: int,
    precision: Optional[str] = None,
    lane: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the lowest mean latency row matching the filters, or None."""
    candidates = [
        r
        for r in rows
        if r["model"] == model
        and r["batch_size"] == batch_size
        and (precision is None or r["precision"] == precision)
        and (lane is None or r["lane"] == lane)
        and r.get("mean_ms") is not None
        and np.isfinite(r["mean_ms"])
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r["mean_ms"])


if __name__ == "__main__":
    main()
