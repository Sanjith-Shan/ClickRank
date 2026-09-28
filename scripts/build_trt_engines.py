#!/usr/bin/env python
"""Build the TensorRT engines the inference benchmark serves from.

TensorRT is an ahead of time compiler, so building an engine is a deploy step
and not a serving step. It reads an ONNX graph, picks the fastest kernel for
every layer on the specific gpu it is running on, fuses what it can, and writes
a serialized plan. That plan is tied to the gpu architecture, the driver, and
the TensorRT version that produced it, which is exactly why this is a separate
reproducible script with its own report rather than something hidden inside the
benchmark.

The script does four things in order. It trains or loads a checkpoint for every
requested model. It exports each one to ONNX with a dynamic batch axis. It
builds one engine per requested precision, running INT8 calibration on rows
drawn from the validation split and writing the resulting scales to a cache. It
then writes results/trt/build_report.json and a short markdown table with the
build time and the disk size of every engine.

Build time is reported here and nowhere else. It belongs next to the engine it
produced, not folded into a latency number, because a build happens once at
deploy time and a batch is scored millions of times after that.

On a machine with no NVIDIA gpu this script prints what is missing, names the
container that has it, and exits zero. It does not crash and it does not
pretend. The development machine for this project is an Apple Silicon laptop,
so that path is the one that runs most often.

Run from the repository root.
    python scripts/build_trt_engines.py
    python scripts/build_trt_engines.py --models deepfm dcn --precisions fp32 fp16 int8
    python scripts/build_trt_engines.py --max-batch 8192 --calib-batches 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

# Insert the repository root onto sys.path so that "import src" works when this
# script is run directly from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data.loader import generate_synthetic, load_raw  # noqa: E402
from src.data.preprocess import build_datasets  # noqa: E402
from src.data.split import temporal_split  # noqa: E402
from src.inference.quantize import build_calibration_feeds, quantize_onnx_int8
from src.inference.calibrator import (  # noqa: E402
    calibration_arrays,
    calibration_cache_path,
    describe_calibration,
    make_entropy_calibrator,
)
from src.inference.common import jsonable  # noqa: E402
from src.inference.export import (  # noqa: E402
    SUPPORTED_MODELS,
    checkpoint_path_for,
    dataset_arrays,
    display_name,
    export_onnx,
    load_or_train_module,
    onnx_path_for,
)
from src.inference.hardware import collect_hardware_record, print_hardware_record  # noqa: E402
from src.inference.trt_builder import (  # noqa: E402
    ENGINE_DIR,
    PRECISIONS,
    EngineRecord,
    build_engine,
    convert_onnx_to_fp16,
    build_report_markdown,
    tensorrt_available,
)
from src.train.config import SEED, get_config  # noqa: E402
from src.train.trainer import set_seed  # noqa: E402


def _legacy_calibrator_api() -> bool:
    """Return True when this TensorRT exposes the old int8 calibrator classes.

    TensorRT 11 removed implicit quantization, so on that version there is no
    calibrator to attach and int8 has to come from an explicitly quantized
    graph instead. Returning False here selects that path.
    """
    try:
        import tensorrt as trt
    except Exception:  # noqa: BLE001 no TensorRT at all
        return False
    return hasattr(trt, "IInt8EntropyCalibrator2")


def parse_args() -> argparse.Namespace:
    """Parse the command line flags for the engine builder."""
    parser = argparse.ArgumentParser(
        description="Build TensorRT engines for the ClickRank rankers."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["deepfm", "dcn"],
        choices=list(SUPPORTED_MODELS),
        help="Which models to export and build engines for.",
    )
    parser.add_argument(
        "--int8-mode",
        default="qdq",
        choices=["qdq", "calibrator"],
        help=(
            "How int8 is expressed. qdq rewrites the graph with explicit "
            "quantize and dequantize pairs before the builder runs, which works "
            "on every TensorRT version including 11 where the calibrator api "
            "was removed. calibrator uses the older implicit path and only "
            "works on TensorRT 10 and below."
        ),
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=["fp32", "fp16", "int8"],
        choices=list(PRECISIONS),
        help="Which engine precisions to build.",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=4096,
        help="Largest batch the optimization profile covers.",
    )
    parser.add_argument(
        "--min-batch",
        type=int,
        default=1,
        help="Smallest batch the optimization profile covers.",
    )
    parser.add_argument(
        "--opt-batch",
        type=int,
        default=256,
        help="Batch size TensorRT tunes kernel selection for.",
    )
    parser.add_argument(
        "--calib-batches",
        type=int,
        default=32,
        help="Number of validation batches fed to the INT8 calibrator.",
    )
    parser.add_argument(
        "--calib-batch-size",
        type=int,
        default=256,
        help="Batch size the INT8 calibration runs at.",
    )
    parser.add_argument(
        "--workspace-mb",
        type=int,
        default=4096,
        help="Upper bound on the scratch memory TensorRT may use while building.",
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
        "--synthetic",
        action="store_true",
        help="Force synthetic data even when the data file exists.",
    )
    parser.add_argument(
        "--output",
        default="results/",
        help="Directory where checkpoints and ONNX graphs are written.",
    )
    parser.add_argument(
        "--engine-dir",
        default=ENGINE_DIR,
        help="Directory where serialized engines and calibration caches are written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild engines even when one already exists at the target path.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export the ONNX graphs and stop, without attempting any engine build.",
    )
    return parser.parse_args()


def load_dataframe(args: argparse.Namespace):
    """Load real Criteo data when available, otherwise synthesize it.

    This mirrors the data loading in run_benchmark.py so the calibration rows
    come from the same distribution the model was trained and evaluated on.
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


def write_unavailable_report(
    args: argparse.Namespace, hardware: Dict[str, Any], reason: str
) -> str:
    """Write a build report for a host that cannot build engines, and say why.

    An empty report is still a report. It records the machine, it records that
    no engine was produced, and it records the reason, so the artifact directory
    never contains a stale engine next to a claim that one was just built.
    """
    os.makedirs(args.engine_dir, exist_ok=True)
    path = os.path.join(args.engine_dir, "build_report.json")
    payload = {
        "hardware": hardware,
        "tensorrt_available": False,
        "reason": reason,
        "requested_models": list(args.models),
        "requested_precisions": list(args.precisions),
        "engines": [],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2)

    md_path = os.path.join(args.engine_dir, "build_report.md")
    lines = [
        "# TensorRT Engine Build Report",
        "",
        "No engine was built on this run.",
        "",
        reason,
        "",
        "## Host",
        "",
    ]
    from src.inference.hardware import hardware_markdown_lines

    lines.extend(hardware_markdown_lines(hardware))
    lines.append("")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def main() -> int:
    """Build every requested engine and write the build report."""
    args = parse_args()
    set_seed(SEED)

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.engine_dir, exist_ok=True)

    hardware = collect_hardware_record()
    print_hardware_record(hardware)
    print()

    available, reason = tensorrt_available()
    if not available and not args.export_only:
        print("no TensorRT engine can be built on this host.")
        print()
        print(reason)
        print()
        print(
            "Nothing was built and nothing was faked. Rerun this command inside "
            "a cuda container and the engines will appear under "
            f"{args.engine_dir}. Pass --export-only to write just the ONNX "
            "graphs here, which is the portable artifact the builder consumes."
        )
        path = write_unavailable_report(args, hardware, reason)
        print(f"wrote {path} recording that no engine was built and why.")
        return 0

    # Data and features. The calibration rows come from the validation split so
    # the test rows the benchmark scores stay unseen by the quantizer.
    df = load_dataframe(args)
    train_df, val_df, test_df = temporal_split(df)
    train_ds, val_ds, _test_ds, meta = build_datasets(train_df, val_df, test_df)
    print(
        f"train {len(train_ds)} rows, validation {len(val_ds)} rows, "
        f"{meta.n_numerical} numerical features, {meta.n_embed_fields} embedded fields."
    )
    val_numerical, val_cat = dataset_arrays(val_ds)

    records: List[EngineRecord] = []
    exports: Dict[str, str] = {}

    for model_name in args.models:
        pretty = display_name(model_name)
        print(f"\n=== {pretty} ===")
        config = get_config(model_name)
        checkpoint = checkpoint_path_for(model_name, args.output)
        module = load_or_train_module(
            model_name, meta, config, checkpoint, train_ds, val_ds
        )
        onnx_path = onnx_path_for(model_name, args.output)
        export_onnx(module, meta, args.opt_batch, onnx_path)
        exports[model_name] = onnx_path

        if args.export_only:
            continue

        cache_path = calibration_cache_path(pretty, args.engine_dir)

        # TensorRT 11 removed BuilderFlag.FP16 and builds strongly typed
        # networks, so on that version an fp16 engine has to be built from an
        # fp16 graph rather than from a flag. The conversion keeps the input and
        # output tensors in fp32 so that every backend is still fed the same
        # arrays and compared on the same outputs, which means the only thing
        # that changes between the fp32 row and the fp16 row is the precision
        # the arithmetic ran at. On older TensorRT the flag still works and this
        # graph is simply never used.
        # TensorRT 11 also removed the int8 calibrator, so int8 moves to
        # explicit quantization. The graph gets QuantizeLinear and
        # DequantizeLinear pairs inserted by a calibration pass that runs here,
        # before the builder, using validation rows rather than test rows.
        int8_onnx_path = onnx_path
        quant_record: Dict[str, Any] = {}
        if "int8" in args.precisions and args.int8_mode == "qdq":
            feeds = build_calibration_feeds(
                val_numerical,
                val_cat,
                batch_size=args.calib_batch_size,
                max_batches=args.calib_batches,
            )
            quant_record = quantize_onnx_int8(
                onnx_path, onnx_path.replace(".onnx", "_int8.onnx"), feeds
            )
            print(f"  int8 graph. {quant_record.get('message')}")
            if quant_record.get("ok"):
                int8_onnx_path = quant_record["int8_onnx_path"]

        fp16_onnx_path = onnx_path
        if "fp16" in args.precisions:
            candidate = onnx_path.replace(".onnx", "_fp16.onnx")
            ok, message = convert_onnx_to_fp16(onnx_path, candidate)
            print(f"  fp16 graph. {message}")
            if ok:
                fp16_onnx_path = candidate

        for precision in args.precisions:
            print(f"\n-- {pretty} {precision} --")
            calibrator_factory = None
            calibration_record: Dict[str, Any] = {}
            if precision == "fp16":
                graph_for_precision = fp16_onnx_path
            elif precision == "int8":
                graph_for_precision = int8_onnx_path
            else:
                graph_for_precision = onnx_path

            if precision == "int8" and args.int8_mode == "qdq":
                calibration_record = dict(quant_record)
                if not quant_record.get("ok"):
                    print(
                        "  no int8 graph was produced, so no int8 engine was "
                        "built. " + str(quant_record.get("message"))
                    )
                    records.append(
                        EngineRecord(
                            model=pretty,
                            precision=precision,
                            engine_path="",
                            ok=False,
                            message=str(quant_record.get("message")),
                            calibration=calibration_record,
                        )
                    )
                    continue
            elif precision == "int8":
                calib_numerical, calib_cat, n_calib_batches = calibration_arrays(
                    val_numerical,
                    val_cat,
                    batch_size=args.calib_batch_size,
                    max_batches=args.calib_batches,
                    seed=SEED,
                )
                calibration_record = describe_calibration(
                    cache_path, n_calib_batches, args.calib_batch_size
                )
                if n_calib_batches == 0:
                    print(
                        "  the validation split is too small to fill even one "
                        "calibration batch, so the int8 engine was not built."
                    )
                    records.append(
                        EngineRecord(
                            model=pretty,
                            precision=precision,
                            engine_path="",
                            ok=False,
                            message=(
                                "the validation split held fewer rows than one "
                                "calibration batch, so no int8 engine was built"
                            ),
                            calibration=calibration_record,
                        )
                    )
                    continue

                print(
                    f"  calibrating on {n_calib_batches} validation batches of "
                    f"{args.calib_batch_size} rows, drawn with seed {SEED}."
                )

                def calibrator_factory(  # noqa: F811  bound per precision on purpose
                    input_dtypes, _n=calib_numerical, _c=calib_cat
                ):
                    """Build the entropy calibrator once the graph dtypes are known."""
                    return make_entropy_calibrator(
                        feeds={"numerical": _n, "cat": _c},
                        batch_size=args.calib_batch_size,
                        cache_path=cache_path,
                        input_dtypes={
                            name: dtype_name for name, dtype_name in input_dtypes.items()
                        },
                    )

            record = build_engine(
                onnx_path=graph_for_precision,
                model_name=pretty,
                precision=precision,
                min_batch=args.min_batch,
                opt_batch=args.opt_batch,
                max_batch=args.max_batch,
                workspace_mb=args.workspace_mb,
                calibrator_factory=calibrator_factory,
                calibration_batch_size=args.calib_batch_size,
                calibration_record=calibration_record,
                output_dir=args.engine_dir,
                overwrite=args.overwrite,
            )
            if precision == "int8":
                record.calibration = describe_calibration(
                    cache_path, calibration_record.get("batches", 0), args.calib_batch_size
                )
            records.append(record)
            if not record.ok:
                print(f"  this engine was not built. {record.message}")

    if args.export_only:
        print("\nexport only was requested, so no engine build was attempted.")
        print(f"onnx graphs written for {', '.join(display_name(m) for m in exports)}.")
        return 0

    write_build_report(args, hardware, records, exports)
    return 0


def write_build_report(
    args: argparse.Namespace,
    hardware: Dict[str, Any],
    records: List[EngineRecord],
    exports: Dict[str, str],
) -> None:
    """Write the json and markdown build reports."""
    from src.inference.hardware import hardware_markdown_lines

    json_path = os.path.join(args.engine_dir, "build_report.json")
    payload = {
        "hardware": hardware,
        "tensorrt_available": True,
        "reason": "",
        "requested_models": list(args.models),
        "requested_precisions": list(args.precisions),
        "profile": {
            "min_batch": args.min_batch,
            "opt_batch": args.opt_batch,
            "max_batch": args.max_batch,
        },
        "onnx_graphs": exports,
        "engines": [r.as_dict() for r in records],
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2)

    lines = [
        "# TensorRT Engine Build Report",
        "",
        (
            "Every engine below was compiled on the gpu named in the host table, "
            "for the batch range in the optimization profile, from the ONNX "
            "graph exported by this same script. A serialized plan is tied to "
            "the gpu architecture, the driver, and the TensorRT version that "
            "produced it, so these files are not portable and have to be rebuilt "
            "on the machine that will serve them."
        ),
        "",
        (
            f"The optimization profile covers batches from {args.min_batch} to "
            f"{args.max_batch} and is tuned for {args.opt_batch}. The optimum is "
            "not the maximum on purpose, because tuning only for the largest "
            "batch would leave the online serving case at batch one running on a "
            "kernel that was chosen for a workload it never sees."
        ),
        "",
        "## Engines",
        "",
    ]
    lines.extend(build_report_markdown(records))
    lines.append("")
    lines.append(
        "Build time is a deploy time cost and it is reported here rather than "
        "anywhere in the latency tables. An engine is built once and then scores "
        "batches for as long as the model is in production, so folding the build "
        "into a serving number would misrepresent both."
    )

    failed = [r for r in records if not r.ok]
    if failed:
        lines.append("")
        lines.append("## Engines that were not built")
        lines.append("")
        for record in failed:
            lines.append(f"- {record.model} {record.precision}. {record.message}")

    warned = [r for r in records if r.warnings]
    if warned:
        lines.append("")
        lines.append("## Builder warnings")
        lines.append("")
        for record in warned:
            for warning in record.warnings:
                lines.append(f"- {record.model} {record.precision}. {warning}")

    int8_records = [r for r in records if r.precision == "int8" and r.calibration]
    if int8_records:
        lines.append("")
        lines.append("## INT8 calibration")
        lines.append("")
        lines.append(
            "Calibration rows are drawn from the validation split and never from "
            "the test split, so the INT8 accuracy the benchmark reports is "
            "measured on rows the quantizer has not seen. The scales are written "
            "to a cache file next to the engine, and a later build reads that "
            "cache instead of calibrating again, which is what makes an INT8 "
            "engine reproducible."
        )
        lines.append("")
        lines.append("| Model | Cache | Batches | Batch size | Cache size |")
        lines.append("| --- | --- | --- | --- | --- |")
        for record in int8_records:
            calib = record.calibration
            size = calib.get("cache_bytes")
            lines.append(
                f"| {record.model} | {os.path.basename(calib.get('cache_path', ''))} | "
                f"{calib.get('batches')} | {calib.get('batch_size')} | "
                f"{size if size is not None else 'not available'} |"
            )

    lines.append("")
    lines.append("## Host")
    lines.append("")
    lines.extend(hardware_markdown_lines(hardware))
    lines.append("")

    md_path = os.path.join(args.engine_dir, "build_report.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print()
    print("\n".join(build_report_markdown(records)))
    print()
    print(f"wrote {json_path}.")
    print(f"wrote {md_path}.")
    print("engine build complete.")


if __name__ == "__main__":
    sys.exit(main())
