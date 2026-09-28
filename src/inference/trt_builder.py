"""Build serialized TensorRT engines from an exported ONNX graph.

TensorRT is an ahead of time compiler. It reads a network, picks the fastest
kernel for every layer on the specific gpu it is running on, fuses what it can,
and writes a serialized plan. That plan is not portable across gpu
architectures or across TensorRT versions, which is why the build is a separate
reproducible step with its own report rather than something hidden inside the
benchmark.

Three things in this module are worth reading closely.

The optimization profile is what makes one engine serve many batch sizes. The
exported graph has a dynamic batch axis, and TensorRT needs a minimum, an
optimum, and a maximum for that axis before it can choose kernels. The optimum
is the batch size the engine is tuned for, so the profile here defaults to a
mid range optimum rather than the maximum, because tuning only for the largest
batch would leave the online serving case at batch one on a badly chosen kernel.

INT8 needs its own calibration profile. With dynamic shapes TensorRT refuses to
calibrate until it is told which shape to calibrate at, so this module sets a
separate profile whose minimum, optimum, and maximum are all the calibration
batch size.

Every builder message is captured rather than printed and forgotten. TensorRT
warns about things that change the meaning of a number, for example that the
platform has no fast INT8 path and the request was silently ignored, and those
warnings belong in the build report next to the engine they describe.

Nothing here imports tensorrt at module scope, so this file is importable on a
machine with no gpu.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.inference.common import NOT_AVAILABLE

# Precisions this builder understands, in the order a report should show them.
PRECISIONS: Tuple[str, ...] = ("fp32", "fp16", "int8")

# Where serialized engines and calibration caches are written.
ENGINE_DIR = os.path.join("results", "trt")

# The message printed on a machine that cannot build an engine. It names the
# container to use rather than leaving the reader to guess.
DOCKER_HINT = (
    "TensorRT engine builds need a NVIDIA gpu, the cuda runtime, and the "
    "tensorrt python package. None of those exist on Apple Silicon or on any "
    "cpu only host. Run this on a cuda machine through the gpu image this "
    "repository ships, with docker build -f docker/Dockerfile.tensorrt -t "
    "clickrank-trt . and then docker run --gpus all clickrank-trt. That "
    "image is built on nvcr.io/nvidia/tensorrt:25.01-py3, which carries "
    "TensorRT 10 and the cuda runtime already installed."
)


@dataclass
class EngineRecord:
    """Everything a report needs to know about one engine build.

    Build time is kept separate from every inference measurement on purpose. A
    TensorRT build can take minutes and it happens once at deploy time, so
    folding it into a latency number would be dishonest in one direction and
    ignoring it entirely would be dishonest in the other.
    """

    model: str
    precision: str
    engine_path: str
    ok: bool
    message: str = ""
    build_seconds: Optional[float] = None
    size_bytes: Optional[int] = None
    min_batch: Optional[int] = None
    opt_batch: Optional[int] = None
    max_batch: Optional[int] = None
    onnx_path: str = ""
    tensorrt_version: str = NOT_AVAILABLE
    platform_has_fast_fp16: Optional[bool] = None
    platform_has_fast_int8: Optional[bool] = None
    calibration: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    input_dtypes: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain dict for the json build report."""
        return {
            "model": self.model,
            "precision": self.precision,
            "engine_path": self.engine_path,
            "ok": self.ok,
            "message": self.message,
            "build_seconds": self.build_seconds,
            "size_bytes": self.size_bytes,
            "min_batch": self.min_batch,
            "opt_batch": self.opt_batch,
            "max_batch": self.max_batch,
            "onnx_path": self.onnx_path,
            "tensorrt_version": self.tensorrt_version,
            "platform_has_fast_fp16": self.platform_has_fast_fp16,
            "platform_has_fast_int8": self.platform_has_fast_int8,
            "calibration": self.calibration,
            "warnings": self.warnings,
            "input_dtypes": self.input_dtypes,
        }


def tensorrt_available() -> Tuple[bool, str]:
    """Return whether TensorRT can build here, and the reason when it cannot."""
    try:
        import tensorrt  # noqa: F401
    except ImportError as exc:
        return False, (
            f"the tensorrt python package is not installed ({exc}). {DOCKER_HINT}"
        )
    except Exception as exc:  # noqa: BLE001 a driver mismatch raises here too
        return False, f"tensorrt failed to import ({exc}). {DOCKER_HINT}"

    try:
        import torch

        if not torch.cuda.is_available():
            return False, (
                "tensorrt imported but torch reports no cuda device, so there is "
                f"no gpu to build an engine for. {DOCKER_HINT}"
            )
    except Exception as exc:  # noqa: BLE001
        return False, f"the cuda probe failed ({exc}). {DOCKER_HINT}"
    return True, ""


def engine_path_for(
    model_name: str,
    precision: str,
    max_batch: int,
    output_dir: str = ENGINE_DIR,
) -> str:
    """Return the canonical engine path for one model, precision, and max batch.

    The name carries the maximum batch because an engine is only valid inside
    the batch range its optimization profile was built for, so two engines that
    differ only in that range must not share a file name.
    """
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(
        output_dir, f"{model_name.lower()}_{precision}_bs{int(max_batch)}.engine"
    )


def _make_logger(trt, collected: List[str]):
    """Return a TensorRT logger that records warnings and errors into a list.

    Subclassing trt.ILogger is the supported way to intercept builder messages.
    If the subclass cannot be created on this TensorRT version the function
    falls back to the stock logger and the report simply carries no warnings,
    which is stated rather than silently pretended away.
    """
    try:

        class _RecordingLogger(trt.ILogger):
            """Capture builder messages so they land in the build report."""

            def __init__(self) -> None:
                trt.ILogger.__init__(self)

            def log(self, severity, msg) -> None:  # noqa: A003 name fixed by the api
                text = f"{severity.name.lower()} {msg}"
                if severity in (
                    trt.ILogger.Severity.ERROR,
                    trt.ILogger.Severity.WARNING,
                    trt.ILogger.Severity.INTERNAL_ERROR,
                ):
                    collected.append(text)
                    print(f"  tensorrt {text}")

        return _RecordingLogger()
    except Exception:  # noqa: BLE001 fall back rather than fail the build
        collected.append(
            "builder messages could not be captured on this TensorRT version, "
            "so the warning list below is empty rather than complete"
        )
        return trt.Logger(trt.Logger.WARNING)


def _numpy_dtype_name(trt, dtype) -> str:
    """Map a TensorRT dtype to the numpy dtype name the host side must supply."""
    mapping = {
        "FLOAT": "float32",
        "HALF": "float16",
        "INT8": "int8",
        "INT32": "int32",
        "INT64": "int64",
        "BOOL": "bool",
        "BF16": "float32",
        "UINT8": "uint8",
        "FP8": "float32",
    }
    name = getattr(dtype, "name", str(dtype))
    return mapping.get(name, "float32")


def _graph_is_fp16(onnx_path: str) -> bool:
    """Return True when the onnx graph carries fp16 initializers.

    TensorRT 11 builds strongly typed networks, so the precision of an engine is
    decided by the graph rather than by a builder flag. This is the check that
    tells an fp16 request apart from an fp32 graph that was mislabeled, and it
    looks at the initializers rather than the inputs because the fp16 conversion
    used here deliberately keeps the input and output tensors in fp32 so that
    callers do not have to change how they feed the model.
    """
    try:
        import onnx
    except Exception:  # noqa: BLE001 the check is advisory
        return False
    try:
        model = onnx.load(onnx_path, load_external_data=False)
    except Exception:  # noqa: BLE001
        return False
    fp16 = int(getattr(onnx.TensorProto, "FLOAT16", 10))
    for init in model.graph.initializer:
        if int(init.data_type) == fp16:
            return True
    return False


def _graph_has_qdq(onnx_path: str) -> bool:
    """Return True when the graph carries QuantizeLinear nodes.

    On TensorRT 11 this is what makes an int8 build possible at all. The builder
    no longer accepts a calibrator, so the quantization has to be described by
    the graph itself through explicit quantize and dequantize pairs.
    """
    try:
        import onnx
    except Exception:  # noqa: BLE001
        return False
    try:
        model = onnx.load(onnx_path, load_external_data=False)
    except Exception:  # noqa: BLE001
        return False
    return any(node.op_type == "QuantizeLinear" for node in model.graph.node)


def convert_onnx_to_fp16(onnx_path: str, out_path: str) -> Tuple[bool, str]:
    """Write an fp16 copy of an onnx graph, keeping the io tensors in fp32.

    Returns a tuple of (ok, message). Keeping the inputs and outputs in fp32
    matters because it means the benchmark feeds every backend the same arrays
    and compares the same outputs, so the only thing that changed between the
    fp32 row and the fp16 row is the precision the arithmetic ran at.

    This exists because TensorRT 11 removed BuilderFlag.FP16. On that version an
    fp16 engine has to be built from an fp16 graph, so the conversion moves out
    of the builder and into a preparation step.
    """
    try:
        import onnx
        from onnxconverter_common import float16 as occ_float16
    except Exception as exc:  # noqa: BLE001
        return False, (
            "the fp16 graph conversion needs the onnx and onnxconverter-common "
            f"packages and one of them is missing ({exc}). Install "
            "onnxconverter-common and rerun."
        )
    try:
        model = onnx.load(onnx_path)
        converted = occ_float16.convert_float_to_float16(
            model, keep_io_types=True, disable_shape_infer=True
        )
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        onnx.save(converted, out_path)
    except Exception as exc:  # noqa: BLE001 surface the real reason
        return False, f"the fp16 graph conversion failed ({exc})."
    return True, f"wrote an fp16 onnx graph to {out_path}."


def build_engine(
    onnx_path: str,
    model_name: str,
    precision: str,
    min_batch: int = 1,
    opt_batch: int = 256,
    max_batch: int = 4096,
    workspace_mb: int = 4096,
    calibrator_factory=None,
    calibration_batch_size: int = 256,
    calibration_record: Optional[Dict[str, Any]] = None,
    output_dir: str = ENGINE_DIR,
    overwrite: bool = False,
    verbose: bool = True,
) -> EngineRecord:
    """Build one serialized TensorRT engine and return a record describing it.

    Parameters
    ----------
    onnx_path : str
        The exported ONNX graph. Must have a dynamic batch axis.
    model_name : str
        Used in the engine file name and in the report.
    precision : str
        One of fp32, fp16, or int8.
    min_batch, opt_batch, max_batch : int
        The optimization profile for the dynamic batch axis. The optimum is the
        batch size TensorRT tunes kernel selection for.
    workspace_mb : int
        Upper bound on the scratch memory TensorRT may use while picking kernels.
        A larger workspace lets it consider more tactics.
    calibrator_factory : callable, optional
        Called with the mapping from input name to numpy dtype name and returns
        a calibrator. Only used for int8. It is a factory rather than an
        instance because the dtypes are only known after the graph is parsed.
    calibration_batch_size : int
        The batch size the calibration profile is built at.
    overwrite : bool
        Rebuild even when an engine already exists at the target path.

    Returns
    -------
    EngineRecord
        On failure the record has ok set to false and message set to the reason.
        This function does not raise for an unavailable or unsupported build.
    """
    precision = precision.lower()
    if precision not in PRECISIONS:
        return EngineRecord(
            model=model_name,
            precision=precision,
            engine_path="",
            ok=False,
            message=f"unknown precision {precision}, expected one of {list(PRECISIONS)}",
        )

    ok, reason = tensorrt_available()
    target = engine_path_for(model_name, precision, max_batch, output_dir)
    if not ok:
        return EngineRecord(
            model=model_name,
            precision=precision,
            engine_path=target,
            ok=False,
            message=reason,
            min_batch=min_batch,
            opt_batch=opt_batch,
            max_batch=max_batch,
            onnx_path=onnx_path,
        )

    if not os.path.exists(onnx_path):
        return EngineRecord(
            model=model_name,
            precision=precision,
            engine_path=target,
            ok=False,
            message=f"the onnx graph {onnx_path} does not exist",
            onnx_path=onnx_path,
        )

    import tensorrt as trt

    warnings_collected: List[str] = []
    logger = _make_logger(trt, warnings_collected)

    if os.path.exists(target) and not overwrite:
        if verbose:
            print(f"  engine {target} already exists, reusing it.")
        return EngineRecord(
            model=model_name,
            precision=precision,
            engine_path=target,
            ok=True,
            message="reused an engine that already existed on disk",
            build_seconds=0.0,
            size_bytes=os.path.getsize(target),
            min_batch=min_batch,
            opt_batch=opt_batch,
            max_batch=max_batch,
            onnx_path=onnx_path,
            tensorrt_version=str(trt.__version__),
            calibration=dict(calibration_record or {}),
        )

    builder = trt.Builder(logger)

    # TensorRT 10 dropped implicit batch entirely and the explicit batch flag is
    # a no op there, but it is still required on 8.x. Setting it when the enum
    # exists keeps one code path correct on both.
    # TensorRT 11 removed builder flag precision entirely. There is no
    # BuilderFlag.FP16 and no BuilderFlag.INT8 any more, and a network is
    # instead STRONGLY_TYPED, meaning the precision of every tensor is read off
    # the graph rather than requested at build time. The practical consequence
    # is that an fp16 engine comes from an fp16 onnx graph and an int8 engine
    # comes from a graph carrying explicit quantize and dequantize nodes.
    legacy_precision_flags = hasattr(trt.BuilderFlag, "FP16")

    flags = 0
    explicit = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    if explicit is not None:
        flags = 1 << int(explicit)
    if not legacy_precision_flags:
        strongly = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
        if strongly is not None:
            flags |= 1 << int(strongly)
    network = builder.create_network(flags)

    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as handle:
        parsed = parser.parse(handle.read())
    if not parsed:
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        return EngineRecord(
            model=model_name,
            precision=precision,
            engine_path=target,
            ok=False,
            message="the onnx parser rejected the graph. " + " ".join(errors),
            onnx_path=onnx_path,
            tensorrt_version=str(trt.__version__),
            warnings=warnings_collected,
        )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_mb) * 1024 * 1024
    )

    # One optimization profile covering the whole batch range the benchmark
    # sweeps. Every input in this graph is batched on axis zero.
    profile = builder.create_optimization_profile()
    input_dtypes: Dict[str, str] = {}
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        shape = list(tensor.shape)
        trailing = tuple(int(d) for d in shape[1:])
        profile.set_shape(
            tensor.name,
            (int(min_batch),) + trailing,
            (int(opt_batch),) + trailing,
            (int(max_batch),) + trailing,
        )
        input_dtypes[tensor.name] = _numpy_dtype_name(trt, tensor.dtype)
    config.add_optimization_profile(profile)

    # These two probes were removed in TensorRT 11 along with the flags they
    # described, so they are read through getattr and reported as unknown rather
    # than assumed.
    fast_fp16 = getattr(builder, "platform_has_fast_fp16", None)
    fast_int8 = getattr(builder, "platform_has_fast_int8", None)
    fast_fp16 = None if fast_fp16 is None else bool(fast_fp16)
    fast_int8 = None if fast_int8 is None else bool(fast_int8)

    if legacy_precision_flags:
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
            if fast_fp16 is False:
                warnings_collected.append(
                    "this gpu reports no fast fp16 path, so the fp16 engine may "
                    "fall back to fp32 kernels and show no speedup"
                )
        elif precision == "int8" and _graph_has_qdq(onnx_path):
            # Explicit quantization on a TensorRT that still has the old flags.
            # The graph already carries the scales, so the INT8 builder flag is
            # deliberately not set. Setting it here tells the builder to expect
            # a calibrator, and a build with the flag on and no calibrator fails
            # with "calibration failure occurred with no scaling factors
            # detected" even though the graph was fully quantized. Explicit and
            # implicit quantization are alternatives rather than layers.
            if fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            warnings_collected.append(
                "built from an explicitly quantized qdq graph, so the int8 "
                "builder flag and the calibrator were both left off on purpose"
            )
        elif precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            # Leaving fp16 on as well lets TensorRT pick a fp16 kernel for any
            # layer it decides not to quantize, which is the standard mixed int8
            # recipe.
            if fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            if fast_int8 is False:
                warnings_collected.append(
                    "this gpu reports no fast int8 path, so the int8 engine may "
                    "fall back to higher precision kernels and show no speedup"
                )
    else:
        # The strongly typed path. Precision is a property of the graph that was
        # handed in, so there is nothing to set here and the only honest thing to
        # do is refuse a precision this graph cannot express.
        if precision == "fp16":
            if not _graph_is_fp16(onnx_path):
                return EngineRecord(
                    model=model_name,
                    precision=precision,
                    engine_path=target,
                    ok=False,
                    onnx_path=onnx_path,
                    tensorrt_version=getattr(trt, "__version__", NOT_AVAILABLE),
                    message=(
                    "this TensorRT build is version "
                    f"{getattr(trt, '__version__', 'unknown')}, which removed "
                    "BuilderFlag.FP16 and builds strongly typed networks "
                    "instead. An fp16 engine therefore has to come from an fp16 "
                    "onnx graph, and the graph at "
                    f"{onnx_path} is fp32. Convert it first with "
                    "scripts/build_trt_engines.py, which writes an fp16 graph "
                    "next to the fp32 one."
                    ),
                )
            warnings_collected.append(
                "built as a strongly typed network from an fp16 onnx graph, "
                "because this TensorRT version has no fp16 builder flag"
            )
        elif precision == "int8":
            if _graph_has_qdq(onnx_path):
                # Explicit quantization. The graph already says where int8
                # begins and ends and with what scale, so the builder needs no
                # flag and no calibrator. This is the supported int8 path on
                # TensorRT 11.
                warnings_collected.append(
                    "built as a strongly typed network from an explicitly "
                    "quantized qdq graph, because this TensorRT version removed "
                    "the implicit quantization api"
                )
            else:
                return EngineRecord(
                    model=model_name,
                    precision=precision,
                    engine_path=target,
                    ok=False,
                    onnx_path=onnx_path,
                    tensorrt_version=getattr(trt, "__version__", NOT_AVAILABLE),
                    message=(
                "this TensorRT build is version "
                f"{getattr(trt, '__version__', 'unknown')}, which removed the "
                "implicit quantization api. There is no BuilderFlag.INT8, no "
                "IInt8EntropyCalibrator2, and no set_calibration_profile, so a "
                "calibrator cannot be attached at all. An int8 engine on this "
                "version requires explicit quantization, meaning quantize and "
                "dequantize nodes baked into the onnx graph by a quantization "
                "toolkit before the builder ever sees it. That is a different "
                    "workflow rather than a flag. The graph handed in carries "
                    "no QuantizeLinear nodes, so there is nothing to build and "
                    "no int8 number is reported."
                    ),
                )
    # The calibrator block below is the implicit quantization path only. A graph
    # that already carries qdq nodes has its scales baked in, so there is
    # nothing to calibrate and attaching a calibrator here would put the builder
    # into the implicit mode that explicit quantization replaces.
    if precision == "int8" and not _graph_has_qdq(onnx_path):
        if calibrator_factory is None:
            return EngineRecord(
                model=model_name,
                precision=precision,
                engine_path=target,
                ok=False,
                message=(
                    "an int8 build needs a calibrator and none was supplied, so "
                    "the build was not attempted rather than producing an engine "
                    "with arbitrary scales"
                ),
                onnx_path=onnx_path,
                tensorrt_version=str(trt.__version__),
                platform_has_fast_fp16=fast_fp16,
                platform_has_fast_int8=fast_int8,
                warnings=warnings_collected,
            )
        calibrator = calibrator_factory(input_dtypes)
        if calibrator is None:
            return EngineRecord(
                model=model_name,
                precision=precision,
                engine_path=target,
                ok=False,
                message=(
                    "the calibrator could not be built, most likely because the "
                    "calibration pool was empty, so no int8 engine was produced"
                ),
                onnx_path=onnx_path,
                tensorrt_version=str(trt.__version__),
                platform_has_fast_fp16=fast_fp16,
                platform_has_fast_int8=fast_int8,
                warnings=warnings_collected,
            )
        config.int8_calibrator = calibrator

        # With dynamic shapes TensorRT will not calibrate until it is told which
        # shape to calibrate at, so the calibration profile pins every dimension
        # to the calibration batch size.
        calib_profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            trailing = tuple(int(d) for d in list(tensor.shape)[1:])
            fixed = (int(calibration_batch_size),) + trailing
            calib_profile.set_shape(tensor.name, fixed, fixed, fixed)
        if not config.set_calibration_profile(calib_profile):
            warnings_collected.append(
                "TensorRT refused the calibration profile, so calibration may "
                "run at a shape this build did not intend"
            )

    if verbose:
        print(
            f"  building the {precision} engine for {model_name} with a batch "
            f"profile of min {min_batch}, opt {opt_batch}, max {max_batch}."
        )

    start = time.perf_counter()
    try:
        plan = builder.build_serialized_network(network, config)
    except Exception as exc:  # noqa: BLE001 report the failure rather than crash the sweep
        return EngineRecord(
            model=model_name,
            precision=precision,
            engine_path=target,
            ok=False,
            message=f"the TensorRT build raised {exc}",
            build_seconds=time.perf_counter() - start,
            onnx_path=onnx_path,
            tensorrt_version=str(trt.__version__),
            platform_has_fast_fp16=fast_fp16,
            platform_has_fast_int8=fast_int8,
            warnings=warnings_collected,
            input_dtypes=input_dtypes,
        )
    build_seconds = time.perf_counter() - start

    if plan is None:
        return EngineRecord(
            model=model_name,
            precision=precision,
            engine_path=target,
            ok=False,
            message=(
                "the TensorRT builder returned no plan. The captured builder "
                "messages above name the reason."
            ),
            build_seconds=build_seconds,
            onnx_path=onnx_path,
            tensorrt_version=str(trt.__version__),
            platform_has_fast_fp16=fast_fp16,
            platform_has_fast_int8=fast_int8,
            warnings=warnings_collected,
            input_dtypes=input_dtypes,
        )

    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    with open(target, "wb") as handle:
        handle.write(bytearray(plan))
    size_bytes = os.path.getsize(target)

    if verbose:
        print(
            f"  wrote {target} in {build_seconds:.1f} s "
            f"({size_bytes / 1e6:.1f} MB on disk)."
        )

    return EngineRecord(
        model=model_name,
        precision=precision,
        engine_path=target,
        ok=True,
        message="",
        build_seconds=build_seconds,
        size_bytes=size_bytes,
        min_batch=min_batch,
        opt_batch=opt_batch,
        max_batch=max_batch,
        onnx_path=onnx_path,
        tensorrt_version=str(trt.__version__),
        platform_has_fast_fp16=fast_fp16,
        platform_has_fast_int8=fast_int8,
        calibration=dict(calibration_record or {}),
        warnings=warnings_collected,
        input_dtypes=input_dtypes,
    )


def build_report_markdown(records: List[EngineRecord]) -> List[str]:
    """Render the engine build records as markdown lines.

    Build time and engine size sit in their own table because they are deploy
    time costs, not serving costs. A reader who wants latency should never find
    a build number mixed into it.
    """
    lines = [
        "| Model | Precision | Engine | Build time (s) | Size on disk | Status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        name = os.path.basename(record.engine_path) if record.engine_path else NOT_AVAILABLE
        if record.ok:
            build = f"{record.build_seconds:.1f}" if record.build_seconds is not None else NOT_AVAILABLE
            size = f"{record.size_bytes / 1e6:.1f} MB" if record.size_bytes else NOT_AVAILABLE
            status = "built"
        else:
            build = NOT_AVAILABLE
            size = NOT_AVAILABLE
            status = NOT_AVAILABLE
        lines.append(
            f"| {record.model} | {record.precision} | {name} | {build} | {size} | {status} |"
        )
    return lines
