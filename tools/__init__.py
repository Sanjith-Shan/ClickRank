"""Benchmark automation and profiling helpers for ClickRank.

This package holds the pieces of the automation layer that are worth importing
rather than shelling out to. Nothing in `src/` depends on it and nothing here is
needed to run a benchmark, train a model, or serve one. It exists so that the
shell drivers in `scripts/` have somewhere to put logic that is easier to write
and test in Python than in shell.

- `tools.sweep_config` reads a sweep config from `benchmarks/` and turns it into
  the command line the benchmark script actually accepts.
- `tools.nvtx` is an optional NVTX range helper for making an Nsight Systems
  timeline readable. It is inert when NVTX is not present.
- `tools.summarize_profile` turns the CSV that `nsys stats` emits into the same
  markdown table style the rest of the project reports in.
"""

from __future__ import annotations

__all__ = ["sweep_config", "nvtx", "summarize_profile"]
