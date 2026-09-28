"""Inference serving stack for the ClickRank rankers.

A trained ranker only earns its keep if it can score live traffic inside a
latency budget, and the way it gets there is by leaving the training framework
for an inference optimized runtime. This package holds everything that stands
between a checkpoint and a served prediction.

- hardware. Provenance. The cpu, the platform, the library versions, and the
  gpu when there is one. Every report embeds this record so no number in this
  project is ever printed without the machine it came from.
- backends. The registry that constructs a runner for each runtime, precision,
  and device combination and says plainly when one is not available.
- trt_builder. Builds serialized TensorRT engines from an exported ONNX graph
  at fp32, fp16, and int8, with an optimization profile for the dynamic batch.
- trt_runner. Loads a serialized engine, drives it through the TensorRT 10
  tensor addressing API on an explicit cuda stream, and profiles per layer time.
- calibrator. The INT8 entropy calibrator, fed from the validation split so the
  test accuracy stays honest, with a committed calibration cache.
- power. NVML sampling of gpu watts and utilization, reporting energy in joules
  and inferences per joule.
- analysis. The roofline cost model that decides from measurement whether this
  model is memory bandwidth bound or compute bound.

Every module in this package imports cleanly on a machine with no cuda and no
TensorRT. Anything that needs a gpu is imported lazily inside a function, and
anything that cannot run reports not available with a reason rather than
raising. That is what lets the whole benchmark run end to end on an Apple
Silicon laptop and produce a complete report with the gpu rows honestly empty.
"""

from __future__ import annotations

from src.inference.common import NOT_AVAILABLE, human_bytes, sigmoid

__all__ = [
    "NOT_AVAILABLE",
    "human_bytes",
    "sigmoid",
]
