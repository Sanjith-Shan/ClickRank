#!/usr/bin/env python
"""Performance regression gate for the ClickRank inference benchmark.

A benchmark that nobody watches rots. The numbers in `results/` were true on the
day they were produced and nothing checks that they are still true after the
next dependency bump, the next export change, or the next refactor of the
feature pipeline. This script turns the benchmark from a report into a gate. It
reads a fresh `results/inference_benchmark.json`, reads a committed baseline
from `results/baselines/`, compares them cell by cell, and exits non zero when
something got worse.

A cell is one model at one backend at one batch size, which is the granularity
the benchmark already measures at. Comparing headline numbers only would hide
the interesting failures, because a runtime that got slower at batch size 1 and
faster at batch size 4096 has a flat headline and a broken online serving path.

Three kinds of finding come out of a comparison.

The first is latency. Latency is compared on the median rather than the mean,
because the median is the robust statistic and one stalled batch on a laptop
moves the mean and leaves the median alone. The tail is compared separately on
p99, because the tail is what misses a serving deadline, and because a change
that shows up only in the tail is a different fact about the system than a
change that moves the whole distribution.

The second is accuracy. AUC and logloss are compared against the baseline with
a tight absolute tolerance and no noise allowance at all. The benchmark scores
the same held out rows with the same seed on every run, so the scores are
deterministic given the same weights and the same graph. A moved AUC is not
measurement jitter, it is a changed model, and a faster model that ranks worse
is not an improvement.

The third is structure. A cell that was in the baseline and is not in the new
run is treated as a failure. This is the finding that no latency comparison can
produce, and it is the one most likely to happen quietly. When a runtime loses
its accelerated execution provider it does not crash, it falls back, and the row
either disappears from the table or reappears with the same name and a slower
kernel underneath. Cells that appeared since the baseline are reported too, as
information rather than as a failure.

The exit code says which kind fired, so a CI job can treat them differently.

    0   no regression, possibly with improvements and warnings
    1   usage or input error
    2   latency regression
    3   accuracy regression
    4   a cell in the baseline did not run in the new results
    5   the baseline and the results come from different hardware

Accuracy outranks a missing cell and a missing cell outranks latency, so the
single exit code always names the most serious finding.

Usage.

    python scripts/check_regression.py
    python scripts/check_regression.py --results results/gpu_full/inference_benchmark.json \
        --baseline results/baselines/gpu_full.json
    python scripts/check_regression.py --update-baseline

The statistics, their limits, and the reason a baseline is scoped to one machine
are written up in `docs/BENCHMARK_AUTOMATION.md`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_LATENCY_REGRESSION = 2
EXIT_ACCURACY_REGRESSION = 3
EXIT_MISSING_CELL = 4
EXIT_HARDWARE_MISMATCH = 5

# Ordered most serious first. The report reports every finding it has, and this
# is only the tie break for the one number the process is allowed to exit with.
_EXIT_PRECEDENCE = (
    EXIT_HARDWARE_MISMATCH,
    EXIT_ACCURACY_REGRESSION,
    EXIT_MISSING_CELL,
    EXIT_LATENCY_REGRESSION,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_RESULTS = "results/inference_benchmark.json"
DEFAULT_BASELINE = "results/baselines/inference_benchmark.json"

# A latency move has to clear three independent floors before it is called a
# regression. Any one of them on its own produces false alarms. See the doc.
DEFAULT_LATENCY_REL_TOL = 0.10  # fraction of the baseline median
DEFAULT_LATENCY_ABS_TOL_MS = 0.10  # milliseconds
DEFAULT_TAIL_REL_TOL = 0.25  # p99 is noisier, so its relative floor is looser
DEFAULT_NOISE_SIGMAS = 2.0  # multiples of the standard error of the difference

# Accuracy gets no noise allowance. These are the tolerances for a genuinely
# unchanged model, sized just above float32 accumulation order effects.
DEFAULT_AUC_ABS_TOL = 0.001
DEFAULT_LOGLOSS_ABS_TOL = 0.001

BASELINE_VERSION = 1

# Expected value of the sample range divided by the population standard
# deviation, for a normal sample of n observations. This is the d2 constant from
# statistical process control and it is what turns an observed spread back into
# a standard deviation estimate when the raw samples are not available. The
# benchmark records min, p50, p99, and mean per cell rather than the individual
# latencies, so a range based estimator is the only one that can be built from
# the artifact as it stands.
_D2 = {
    2: 1.128,
    3: 1.693,
    4: 2.059,
    5: 2.326,
    6: 2.534,
    7: 2.704,
    8: 2.847,
    9: 2.970,
    10: 3.078,
}

# The standard error of a sample median is about 1.253 times the standard error
# of the mean for a normal population. Latency is compared on the median, so
# this factor belongs in the noise floor.
_MEDIAN_SE_FACTOR = 1.2533


def _d2(n: int) -> float:
    """Return the range to sigma conversion constant for a sample of size n.

    Above ten samples the constant is clamped rather than extrapolated. Clamping
    low means the estimated sigma comes out high, which widens the noise floor
    and makes the gate quieter rather than louder, and a gate that errs toward
    silence is the safe direction for an estimator this crude.
    """
    if n < 2:
        return _D2[2]
    return _D2.get(n, _D2[10])


def _finite(value: Any) -> Optional[float]:
    """Coerce a json value to a float, returning None for null, nan, and inf."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One comparable measurement, which is one model on one backend at one batch size.

    `backend_key` is the benchmark's own identifier and it already encodes the
    runtime, the device, and the precision as `runtime-device-precision`, for
    example `onnxruntime-cpu-fp32`. The device is part of the identity on
    purpose. The same runtime on a cpu and on a cuda device are different
    deployments answering different questions, and folding them into one cell
    would compare hardware while pretending to compare runtimes.
    """

    key: str
    model: str
    backend_key: str
    label: str
    runtime: str
    precision: str
    device: str
    lane: str
    batch_size: int
    n_samples: int
    mean_ms: Optional[float]
    p50_ms: Optional[float]
    p99_ms: Optional[float]
    min_ms: Optional[float]
    throughput: Optional[float]
    auc: Optional[float]
    logloss: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "model": self.model,
            "backend_key": self.backend_key,
            "label": self.label,
            "runtime": self.runtime,
            "precision": self.precision,
            "device": self.device,
            "lane": self.lane,
            "batch_size": self.batch_size,
            "n_samples": self.n_samples,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p99_ms": self.p99_ms,
            "min_ms": self.min_ms,
            "throughput_samples_per_s": self.throughput,
            "auc": self.auc,
            "logloss": self.logloss,
        }

    @property
    def sigma_ms(self) -> Optional[float]:
        """Estimate the per batch latency standard deviation from the recorded spread.

        The estimate is the observed spread divided by the d2 constant for the
        sample size. p99 stands in for the maximum, which understates the true
        range and therefore understates sigma once the sample count is large
        enough for p99 to sit well below the maximum. At the sample counts this
        benchmark actually runs, which are single digits, p99 and the maximum
        are the same observation.
        """
        if self.p99_ms is None or self.min_ms is None:
            return None
        spread = max(self.p99_ms - self.min_ms, 0.0)
        return spread / _d2(self.n_samples)


def cell_key(model: str, backend_key: str, batch_size: Any) -> str:
    """Build the stable identifier a cell is matched on across runs.

    The separator is a slash rather than a pipe so the key can sit inside a
    markdown table cell without splitting the row.
    """
    return f"{model}/{backend_key}/bs{batch_size}"


def measurement_to_cell(record: Dict[str, Any]) -> Cell:
    """Convert one entry of the benchmark `measurements` list into a Cell."""
    model = str(record.get("model", "unknown"))
    backend_key = str(record.get("backend_key", "unknown"))
    batch_size = record.get("batch_size")
    # A raw measurement records the repeat count and the number of timed
    # batches. A baseline stores the product it already worked out, so read that
    # back when it is there and recompute it when it is not.
    if record.get("n_samples") is not None:
        n_samples = int(record["n_samples"])
    else:
        repeats = int(record.get("repeats") or 1)
        timed = int(record.get("n_timed_batches") or 1)
        n_samples = repeats * timed
    return Cell(
        key=cell_key(model, backend_key, batch_size),
        model=model,
        backend_key=backend_key,
        label=str(record.get("label", backend_key)),
        runtime=str(record.get("runtime", "unknown")),
        precision=str(record.get("precision", "unknown")),
        device=str(record.get("device", "unknown")),
        lane=str(record.get("lane", "unknown")),
        batch_size=int(batch_size) if batch_size is not None else 0,
        n_samples=max(n_samples, 1),
        mean_ms=_finite(record.get("mean_ms")),
        p50_ms=_finite(record.get("p50_ms")),
        p99_ms=_finite(record.get("p99_ms")),
        min_ms=_finite(record.get("min_ms")),
        throughput=_finite(record.get("throughput_samples_per_s")),
        auc=_finite(record.get("auc")),
        logloss=_finite(record.get("logloss")),
    )


def extract_cells(payload: Dict[str, Any]) -> Dict[str, Cell]:
    """Pull every comparable cell out of a benchmark payload or a baseline file.

    Both shapes are accepted. A raw benchmark artifact carries a `measurements`
    list. A baseline file carries a `cells` map that was built from one. Reading
    both here means the comparison code never has to care which side it was
    handed.
    """
    cells: Dict[str, Cell] = {}
    if isinstance(payload.get("cells"), dict):
        for key, record in payload["cells"].items():
            cells[key] = measurement_to_cell(record)
        return cells
    for record in payload.get("measurements") or []:
        cell = measurement_to_cell(record)
        cells[cell.key] = cell
    return cells


def unavailable_reasons(payload: Dict[str, Any]) -> Dict[str, str]:
    """Map a backend key to the reason the benchmark gave for not running it.

    This is what turns a disappeared cell from a mystery into a diagnosis. The
    benchmark already records why a backend was skipped, so a missing cell can
    usually be reported with the runtime's own words rather than a guess.
    """
    out: Dict[str, str] = {}
    for record in payload.get("unavailable_backends") or []:
        key = record.get("key")
        if key:
            out[str(key)] = str(record.get("reason", "no reason recorded"))
    return out


# ---------------------------------------------------------------------------
# Hardware scoping
# ---------------------------------------------------------------------------

# The fields that define whether two runs are comparable at all. A latency
# number belongs to the machine that produced it, so a baseline is an artifact
# of one machine and comparing across machines measures the machines.
_FINGERPRINT_FIELDS = (
    "cpu_model",
    "system",
    "machine",
    "logical_cores",
    "gpu_name",
    "gpu_driver_version",
)

# Version drift that changes the numbers without changing the machine. This is a
# warning rather than a refusal, because a dependency bump moving the latency is
# exactly the event the gate exists to catch, and refusing to compare would hide
# it.
_LIBRARY_FIELDS = ("python", "numpy", "torch", "onnxruntime", "openvino", "tensorrt")


def hardware_fingerprint(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a hardware record down to the fields that decide comparability."""
    hardware = payload.get("hardware") or {}
    host = hardware.get("host") or {}
    gpu = hardware.get("gpu") or {}
    return {
        "cpu_model": host.get("cpu_model", "unknown"),
        "system": host.get("system", "unknown"),
        "machine": host.get("machine", "unknown"),
        "logical_cores": host.get("logical_cores"),
        "gpu_name": gpu.get("name", "not available"),
        "gpu_driver_version": gpu.get("driver_version", "not available"),
    }


def library_versions(payload: Dict[str, Any]) -> Dict[str, str]:
    """Reduce a hardware record down to the library versions worth reporting."""
    hardware = payload.get("hardware") or {}
    libs = hardware.get("libraries") or {}
    python = (hardware.get("python") or {}).get("version", "unknown")
    out: Dict[str, str] = {"python": str(python)}
    for name in ("numpy", "torch", "onnxruntime", "openvino", "tensorrt"):
        value = libs.get(name)
        if isinstance(value, dict):
            value = value.get("version", "not available")
        out[name] = str(value if value is not None else "not available")
    return out


def fingerprint_differences(
    baseline: Dict[str, Any], current: Dict[str, Any]
) -> List[Tuple[str, Any, Any]]:
    """List the fingerprint fields that disagree between two runs."""
    diffs: List[Tuple[str, Any, Any]] = []
    for name in _FINGERPRINT_FIELDS:
        old = baseline.get(name)
        new = current.get(name)
        if old != new:
            diffs.append((name, old, new))
    return diffs


def version_differences(
    baseline: Dict[str, str], current: Dict[str, str]
) -> List[Tuple[str, Any, Any]]:
    """List the library versions that changed between two runs."""
    diffs: List[Tuple[str, Any, Any]] = []
    for name in _LIBRARY_FIELDS:
        old = baseline.get(name)
        new = current.get(name)
        if old != new:
            diffs.append((name, old, new))
    return diffs


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tolerances:
    """Every knob the gate has, in one place, so a run can record what it used.

    `latency_rel` and `latency_abs_ms` are the relative and absolute floors. Both
    have to be cleared, which is the rule that stops a 0.01 ms move on a 0.05 ms
    measurement from being announced as a twenty percent regression and stops a
    0.5 ms move on a 400 ms measurement from being announced at all.

    `noise_sigmas` is the third floor and it is the one that uses the recorded
    distribution rather than a fixed number.
    """

    latency_rel: float = DEFAULT_LATENCY_REL_TOL
    latency_abs_ms: float = DEFAULT_LATENCY_ABS_TOL_MS
    tail_rel: float = DEFAULT_TAIL_REL_TOL
    noise_sigmas: float = DEFAULT_NOISE_SIGMAS
    auc_abs: float = DEFAULT_AUC_ABS_TOL
    logloss_abs: float = DEFAULT_LOGLOSS_ABS_TOL
    fail_on_tail: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "latency_rel": self.latency_rel,
            "latency_abs_ms": self.latency_abs_ms,
            "tail_rel": self.tail_rel,
            "noise_sigmas": self.noise_sigmas,
            "auc_abs": self.auc_abs,
            "logloss_abs": self.logloss_abs,
            "fail_on_tail": self.fail_on_tail,
        }


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

# Verdicts a single metric on a single cell can land on.
VERDICT_OK = "ok"
VERDICT_NOISE = "within noise"
VERDICT_REGRESSION = "regression"
VERDICT_IMPROVEMENT = "improvement"
VERDICT_UNKNOWN = "not comparable"


@dataclass
class MetricFinding:
    """The outcome of comparing one metric on one cell."""

    cell_key: str
    label: str
    metric: str
    baseline: Optional[float]
    current: Optional[float]
    delta: Optional[float]
    rel_delta: Optional[float]
    noise_floor: Optional[float]
    verdict: str
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cell": self.cell_key,
            "label": self.label,
            "metric": self.metric,
            "baseline": self.baseline,
            "current": self.current,
            "delta": self.delta,
            "rel_delta": self.rel_delta,
            "noise_floor": self.noise_floor,
            "verdict": self.verdict,
            "note": self.note,
        }


@dataclass
class Report:
    """Everything one comparison found, plus the exit code that summarizes it."""

    baseline_path: str
    results_path: str
    tolerances: Tolerances
    hardware_baseline: Dict[str, Any] = field(default_factory=dict)
    hardware_current: Dict[str, Any] = field(default_factory=dict)
    hardware_diffs: List[Tuple[str, Any, Any]] = field(default_factory=list)
    hardware_refused: bool = False
    version_diffs: List[Tuple[str, Any, Any]] = field(default_factory=list)
    latency: List[MetricFinding] = field(default_factory=list)
    tail: List[MetricFinding] = field(default_factory=list)
    accuracy: List[MetricFinding] = field(default_factory=list)
    missing_cells: List[Tuple[str, str, str]] = field(default_factory=list)
    new_cells: List[Tuple[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # -- convenience views ---------------------------------------------------

    @property
    def latency_regressions(self) -> List[MetricFinding]:
        return [f for f in self.latency if f.verdict == VERDICT_REGRESSION]

    @property
    def latency_improvements(self) -> List[MetricFinding]:
        return [f for f in self.latency if f.verdict == VERDICT_IMPROVEMENT]

    @property
    def tail_regressions(self) -> List[MetricFinding]:
        return [f for f in self.tail if f.verdict == VERDICT_REGRESSION]

    @property
    def accuracy_regressions(self) -> List[MetricFinding]:
        return [f for f in self.accuracy if f.verdict == VERDICT_REGRESSION]

    @property
    def accuracy_improvements(self) -> List[MetricFinding]:
        return [f for f in self.accuracy if f.verdict == VERDICT_IMPROVEMENT]

    @property
    def exit_code(self) -> int:
        """Return the most serious finding as a single process exit code."""
        codes = set()
        if self.hardware_refused:
            codes.add(EXIT_HARDWARE_MISMATCH)
        if self.accuracy_regressions:
            codes.add(EXIT_ACCURACY_REGRESSION)
        if self.missing_cells:
            codes.add(EXIT_MISSING_CELL)
        if self.latency_regressions:
            codes.add(EXIT_LATENCY_REGRESSION)
        if self.tolerances.fail_on_tail and self.tail_regressions:
            codes.add(EXIT_LATENCY_REGRESSION)
        for code in _EXIT_PRECEDENCE:
            if code in codes:
                return code
        return EXIT_OK

    def as_dict(self) -> Dict[str, Any]:
        return {
            "baseline_path": self.baseline_path,
            "results_path": self.results_path,
            "tolerances": self.tolerances.as_dict(),
            "hardware_baseline": self.hardware_baseline,
            "hardware_current": self.hardware_current,
            "hardware_differences": [
                {"field": name, "baseline": old, "current": new}
                for name, old, new in self.hardware_diffs
            ],
            "hardware_refused": self.hardware_refused,
            "library_differences": [
                {"library": name, "baseline": old, "current": new}
                for name, old, new in self.version_diffs
            ],
            "latency": [f.as_dict() for f in self.latency],
            "tail": [f.as_dict() for f in self.tail],
            "accuracy": [f.as_dict() for f in self.accuracy],
            "missing_cells": [
                {"cell": key, "label": label, "reason": reason}
                for key, label, reason in self.missing_cells
            ],
            "new_cells": [{"cell": key, "label": label} for key, label in self.new_cells],
            "notes": list(self.notes),
            "exit_code": self.exit_code,
        }


# ---------------------------------------------------------------------------
# The comparison itself
# ---------------------------------------------------------------------------


def _noise_floor_ms(base: Cell, current: Cell, tol: Tolerances) -> Optional[float]:
    """Return the smallest latency move that is not plausibly measurement noise.

    The two runs each give a sigma estimate from their own recorded spread. The
    quantity being compared is a difference of two medians, so the two standard
    errors combine in quadrature and the median correction factor applies to
    both. The floor is that combined standard error multiplied by the configured
    number of sigmas.

    Returning None means the spread was not recorded on one of the two sides, in
    which case the caller falls back to the fixed floors alone and says so.
    """
    sigma_base = base.sigma_ms
    sigma_cur = current.sigma_ms
    if sigma_base is None or sigma_cur is None:
        return None
    variance = (sigma_base**2) / max(base.n_samples, 1) + (sigma_cur**2) / max(
        current.n_samples, 1
    )
    standard_error = _MEDIAN_SE_FACTOR * math.sqrt(variance)
    return tol.noise_sigmas * standard_error


def compare_latency(
    base: Cell, current: Cell, tol: Tolerances, metric: str = "p50_ms"
) -> MetricFinding:
    """Compare one latency metric on one cell against three independent floors.

    A move is called a regression only when it clears the relative floor, the
    absolute floor, and the noise floor. Any one floor on its own misbehaves. A
    relative floor alone screams about sub microsecond moves on fast cells. An
    absolute floor alone goes silent on slow cells where a large relative
    regression is worth catching. A noise floor alone lets a small but real and
    perfectly repeatable regression through on a very quiet backend.

    An improvement is the same test with the sign flipped. Improvements are
    reported and never fail the gate, because the point of reporting them is to
    know when the baseline has become stale enough to be worth refreshing.
    """
    base_value = getattr(base, metric)
    current_value = getattr(current, metric)
    rel_tol = tol.tail_rel if metric == "p99_ms" else tol.latency_rel

    if base_value is None or current_value is None or base_value <= 0.0:
        return MetricFinding(
            cell_key=current.key,
            label=current.label,
            metric=metric,
            baseline=base_value,
            current=current_value,
            delta=None,
            rel_delta=None,
            noise_floor=None,
            verdict=VERDICT_UNKNOWN,
            note="one side has no usable measurement for this metric",
        )

    delta = current_value - base_value
    rel_delta = delta / base_value
    noise_floor = _noise_floor_ms(base, current, tol)

    magnitude = abs(delta)
    clears_rel = abs(rel_delta) >= rel_tol
    clears_abs = magnitude >= tol.latency_abs_ms
    clears_noise = noise_floor is None or magnitude >= noise_floor

    if clears_rel and clears_abs and clears_noise:
        verdict = VERDICT_REGRESSION if delta > 0 else VERDICT_IMPROVEMENT
        note = ""
    elif magnitude == 0.0:
        verdict = VERDICT_OK
        note = "identical"
    else:
        verdict = VERDICT_NOISE
        held_by = []
        if not clears_rel:
            held_by.append(f"relative floor {rel_tol:.0%}")
        if not clears_abs:
            held_by.append(f"absolute floor {tol.latency_abs_ms:.3f} ms")
        if not clears_noise and noise_floor is not None:
            held_by.append(f"noise floor {noise_floor:.3f} ms")
        note = "under the " + ", the ".join(held_by)

    if noise_floor is None and verdict in (VERDICT_REGRESSION, VERDICT_IMPROVEMENT):
        note = "no spread recorded, so only the fixed floors were applied"

    return MetricFinding(
        cell_key=current.key,
        label=current.label,
        metric=metric,
        baseline=base_value,
        current=current_value,
        delta=delta,
        rel_delta=rel_delta,
        noise_floor=noise_floor,
        verdict=verdict,
        note=note,
    )


def compare_accuracy(base: Cell, current: Cell, tol: Tolerances) -> List[MetricFinding]:
    """Compare AUC and logloss on one cell with an absolute tolerance and no noise term.

    There is no noise floor here on purpose. The benchmark scores the full held
    out split with a fixed seed, so two runs of an unchanged model over unchanged
    data produce the same score to well inside these tolerances. The only jitter
    available is float32 accumulation order, which is orders of magnitude below
    a tolerance of a thousandth of an AUC point. Anything larger is a real change
    in what the model predicts, and the gate should say so.

    Higher AUC is better and lower logloss is better, so the sign convention
    differs between the two and is handled explicitly rather than by a shared
    helper that would be easy to get backwards.
    """
    findings: List[MetricFinding] = []

    for metric, tolerance, higher_is_better in (
        ("auc", tol.auc_abs, True),
        ("logloss", tol.logloss_abs, False),
    ):
        base_value = getattr(base, metric)
        current_value = getattr(current, metric)
        if base_value is None or current_value is None:
            findings.append(
                MetricFinding(
                    cell_key=current.key,
                    label=current.label,
                    metric=metric,
                    baseline=base_value,
                    current=current_value,
                    delta=None,
                    rel_delta=None,
                    noise_floor=None,
                    verdict=VERDICT_UNKNOWN,
                    note="one side has no usable measurement for this metric",
                )
            )
            continue

        delta = current_value - base_value
        worse = -delta if higher_is_better else delta
        if abs(delta) < tolerance:
            verdict = VERDICT_OK
            note = f"inside the absolute tolerance of {tolerance}"
        elif worse > 0:
            verdict = VERDICT_REGRESSION
            note = "a changed score means a changed model, not measurement jitter"
        else:
            verdict = VERDICT_IMPROVEMENT
            note = "better than the baseline, which is still a changed model"

        findings.append(
            MetricFinding(
                cell_key=current.key,
                label=current.label,
                metric=metric,
                baseline=base_value,
                current=current_value,
                delta=delta,
                rel_delta=delta / base_value if base_value else None,
                noise_floor=None,
                verdict=verdict,
                note=note,
            )
        )

    return findings


def compare(
    baseline_payload: Dict[str, Any],
    current_payload: Dict[str, Any],
    tolerances: Optional[Tolerances] = None,
    baseline_path: str = DEFAULT_BASELINE,
    results_path: str = DEFAULT_RESULTS,
    allow_hardware_mismatch: bool = False,
) -> Report:
    """Compare a fresh benchmark payload against a baseline and return the findings.

    The hardware check runs first and short circuits everything else unless the
    caller has explicitly allowed a mismatch, because a latency table compared
    across two machines is a comparison of the machines and reporting it as a
    regression list would be worse than reporting nothing.
    """
    tol = tolerances or Tolerances()
    report = Report(
        baseline_path=baseline_path,
        results_path=results_path,
        tolerances=tol,
        hardware_baseline=hardware_fingerprint(baseline_payload),
        hardware_current=hardware_fingerprint(current_payload),
    )
    report.hardware_diffs = fingerprint_differences(
        report.hardware_baseline, report.hardware_current
    )
    report.version_diffs = version_differences(
        library_versions(baseline_payload), library_versions(current_payload)
    )

    if report.hardware_diffs and not allow_hardware_mismatch:
        report.hardware_refused = True
        report.notes.append(
            "the comparison was refused because the baseline was recorded on "
            "different hardware, and no cell was compared"
        )
        return report

    if report.hardware_diffs and allow_hardware_mismatch:
        report.notes.append(
            "hardware differs from the baseline and the comparison was forced, "
            "so every latency finding below is a comparison of two machines and "
            "not of two builds"
        )

    base_cells = extract_cells(baseline_payload)
    current_cells = extract_cells(current_payload)
    reasons = unavailable_reasons(current_payload)

    for key in sorted(base_cells):
        base = base_cells[key]
        current = current_cells.get(key)
        if current is None:
            reason = reasons.get(
                base.backend_key,
                "the new run recorded no measurement and no reason for this cell",
            )
            report.missing_cells.append((key, base.label, reason))
            continue
        report.latency.append(compare_latency(base, current, tol, "p50_ms"))
        report.tail.append(compare_latency(base, current, tol, "p99_ms"))
        report.accuracy.extend(compare_accuracy(base, current, tol))

    for key in sorted(current_cells):
        if key not in base_cells:
            report.new_cells.append((key, current_cells[key].label))

    return report


# ---------------------------------------------------------------------------
# Baseline files
# ---------------------------------------------------------------------------


def make_baseline(payload: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Build a committable baseline document from a benchmark payload.

    The hardware record travels with the baseline rather than being left in a
    README, because the hardware is what makes the numbers mean anything and a
    baseline separated from its machine is a set of numbers with no units.
    """
    cells = extract_cells(payload)
    return {
        "baseline_version": BASELINE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "seed": payload.get("seed"),
        "n_test_rows": payload.get("n_test_rows"),
        "arguments": payload.get("arguments"),
        "hardware_fingerprint": hardware_fingerprint(payload),
        "library_versions": library_versions(payload),
        "hardware": payload.get("hardware"),
        "cells": {key: cell.as_dict() for key, cell in sorted(cells.items())},
    }


def load_json(path: str) -> Dict[str, Any]:
    """Read a json document and fail with a readable message rather than a traceback."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_baseline(baseline: Dict[str, Any], path: str) -> str:
    """Write a baseline document, creating the directory if it is not there."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(baseline, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return path


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _score(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.5f}"


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.1f}%"


def _rule(char: str = "-", width: int = 78) -> str:
    return char * width


def render_report(report: Report) -> str:
    """Render the findings as text a person can read in a CI log.

    The tables use the same pipe style as the rest of the project's markdown, so
    a block of this output can be pasted into a pull request without reformatting.
    """
    lines: List[str] = []
    lines.append(_rule("="))
    lines.append("ClickRank inference regression gate")
    lines.append(_rule("="))
    lines.append(f"results  {report.results_path}")
    lines.append(f"baseline {report.baseline_path}")
    lines.append("")

    lines.append("Hardware scope")
    lines.append("")
    lines.append("| Field | Baseline | This run |")
    lines.append("| --- | --- | --- |")
    for name in _FINGERPRINT_FIELDS:
        old = report.hardware_baseline.get(name)
        new = report.hardware_current.get(name)
        mark = "" if old == new else "  <- differs"
        lines.append(f"| {name} | {old} | {new}{mark} |")
    lines.append("")

    if report.hardware_refused:
        lines.append(_rule("!"))
        lines.append("REFUSED. The baseline and this run come from different machines.")
        lines.append(_rule("!"))
        lines.append("")
        for name, old, new in report.hardware_diffs:
            lines.append(f"  {name} was {old} and is now {new}")
        lines.append("")
        lines.append(
            "A latency baseline is a property of one machine. Comparing across "
            "machines measures the machines rather than the change, so nothing "
            "was compared. Record a baseline on this machine with "
            "--update-baseline, or force the comparison with "
            "--allow-hardware-mismatch and treat every number below as a "
            "cross machine comparison."
        )
        lines.append("")
        lines.append(f"exit code {report.exit_code}")
        return "\n".join(lines)

    if report.version_diffs:
        lines.append("Library versions moved since the baseline was recorded.")
        lines.append("")
        for name, old, new in report.version_diffs:
            lines.append(f"  {name} was {old} and is now {new}")
        lines.append("")
        lines.append(
            "This is a warning and not a failure. A dependency bump changing the "
            "latency is exactly the event this gate exists to catch, so the "
            "comparison goes ahead and the versions are printed next to it."
        )
        lines.append("")

    lines.append("Latency, compared on the median")
    lines.append("")
    lines.append("| Cell | Baseline p50 | Now | Delta | Noise floor | Verdict |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for finding in report.latency:
        lines.append(
            f"| {finding.cell_key} | {_ms(finding.baseline)} | {_ms(finding.current)} "
            f"| {_pct(finding.rel_delta)} | {_ms(finding.noise_floor)} "
            f"| {finding.verdict} |"
        )
    if not report.latency:
        lines.append("| no cell was comparable | | | | | |")
    lines.append("")

    tail_moves = [f for f in report.tail if f.verdict in (VERDICT_REGRESSION, VERDICT_IMPROVEMENT)]
    if tail_moves:
        lines.append("Tail, compared on p99")
        lines.append("")
        lines.append("| Cell | Baseline p99 | Now | Delta | Verdict |")
        lines.append("| --- | --- | --- | --- | --- |")
        for finding in tail_moves:
            lines.append(
                f"| {finding.cell_key} | {_ms(finding.baseline)} | {_ms(finding.current)} "
                f"| {_pct(finding.rel_delta)} | {finding.verdict} |"
            )
        lines.append("")
        if report.tail_regressions and not report.tolerances.fail_on_tail:
            lines.append(
                "Tail moves are reported and do not fail the gate by default, "
                "because p99 over a handful of timed batches is close to a "
                "single maximum and moves for reasons that have nothing to do "
                "with the change under test. Pass --fail-on-tail to promote "
                "them once the repeat count is high enough for p99 to be stable."
            )
            lines.append("")

    accuracy_moves = [
        f for f in report.accuracy if f.verdict in (VERDICT_REGRESSION, VERDICT_IMPROVEMENT)
    ]
    lines.append("Accuracy")
    lines.append("")
    if accuracy_moves:
        lines.append("| Cell | Metric | Baseline | Now | Delta | Verdict |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for finding in accuracy_moves:
            delta = "n/a" if finding.delta is None else f"{finding.delta:+.5f}"
            lines.append(
                f"| {finding.cell_key} | {finding.metric} | {_score(finding.baseline)} "
                f"| {_score(finding.current)} | {delta} | {finding.verdict} |"
            )
    else:
        lines.append(
            "every comparable cell reproduced its baseline AUC and logloss "
            "inside the absolute tolerance"
        )
    lines.append("")

    if report.missing_cells:
        lines.append("Cells that were in the baseline and did not run")
        lines.append("")
        for key, label, reason in report.missing_cells:
            lines.append(f"  {key}")
            lines.append(f"    {label}")
            lines.append(f"    {reason}")
        lines.append("")
        lines.append(
            "A backend that stops running is a regression that no latency "
            "comparison can see, because there is no row left to compare. This "
            "is the failure mode of a silent fallback from an accelerated "
            "execution provider to a slower one."
        )
        lines.append("")

    if report.new_cells:
        lines.append("Cells that appeared since the baseline")
        lines.append("")
        for key, label in report.new_cells:
            lines.append(f"  {key}  {label}")
        lines.append("")
        lines.append(
            "New cells are information rather than a failure. They have no "
            "baseline to be compared against until the baseline is updated."
        )
        lines.append("")

    lines.append(_rule())
    lines.append("Summary")
    lines.append(_rule())
    lines.append(f"  cells compared            {len(report.latency)}")
    lines.append(f"  latency regressions       {len(report.latency_regressions)}")
    lines.append(f"  latency improvements      {len(report.latency_improvements)}")
    lines.append(f"  tail regressions          {len(report.tail_regressions)}")
    lines.append(f"  accuracy regressions      {len(report.accuracy_regressions)}")
    lines.append(f"  accuracy improvements     {len(report.accuracy_improvements)}")
    lines.append(f"  cells that disappeared    {len(report.missing_cells)}")
    lines.append(f"  cells that appeared       {len(report.new_cells)}")
    lines.append("")

    for note in report.notes:
        lines.append(f"note. {note}")
    if report.notes:
        lines.append("")

    code = report.exit_code
    verdicts = {
        EXIT_OK: "PASS. Nothing got worse.",
        EXIT_LATENCY_REGRESSION: "FAIL. A latency regression cleared every floor.",
        EXIT_ACCURACY_REGRESSION: (
            "FAIL. An accuracy regression fired. A faster model that ranks "
            "worse is not an improvement, so this outranks any latency finding."
        ),
        EXIT_MISSING_CELL: "FAIL. A cell in the baseline did not run.",
        EXIT_HARDWARE_MISMATCH: "FAIL. Hardware mismatch.",
    }
    lines.append(verdicts.get(code, "FAIL."))
    lines.append(f"exit code {code}")
    return "\n".join(lines)


def render_update_banner(path: str, cell_count: int, fingerprint: Dict[str, Any]) -> str:
    """Render the loud banner that a baseline rewrite prints.

    Loud on purpose. A baseline that is updated by accident silently blesses
    whatever regression was in the tree at the time and the gate goes quiet
    forever after, which is worse than having no gate, because it looks like it
    is still working.
    """
    lines = []
    lines.append("")
    lines.append(_rule("*"))
    lines.append("*** BASELINE REWRITTEN ***")
    lines.append(_rule("*"))
    lines.append("")
    lines.append(f"  file          {path}")
    lines.append(f"  cells         {cell_count}")
    lines.append(f"  cpu           {fingerprint.get('cpu_model')}")
    lines.append(f"  gpu           {fingerprint.get('gpu_name')}")
    lines.append(f"  system        {fingerprint.get('system')} {fingerprint.get('machine')}")
    lines.append("")
    lines.append(
        "  Every future run is now measured against these numbers. If a "
        "regression was present in this run it has just been accepted as "
        "normal. Read the diff before committing this file."
    )
    lines.append("")
    lines.append(_rule("*"))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="check_regression.py",
        description=(
            "Compare a fresh inference benchmark against a committed baseline, "
            "cell by cell, and exit non zero on a regression."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exit codes\n"
            "  0  no regression\n"
            "  1  usage or input error\n"
            "  2  latency regression\n"
            "  3  accuracy regression, which outranks a latency regression\n"
            "  4  a cell in the baseline did not run\n"
            "  5  the baseline came from different hardware\n"
        ),
    )
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS,
        help="Benchmark json to check. Default %(default)s",
    )
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE,
        help="Committed baseline to compare against. Default %(default)s",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help=(
            "Rewrite the baseline from --results instead of gating on it. "
            "Prints a loud banner, because a baseline updated by accident "
            "silently blesses whatever regression was in the tree."
        ),
    )
    parser.add_argument(
        "--allow-hardware-mismatch",
        action="store_true",
        help=(
            "Compare anyway when the baseline came from another machine. Every "
            "latency finding then compares two machines rather than two builds."
        ),
    )
    parser.add_argument(
        "--latency-rel-tol",
        type=float,
        default=DEFAULT_LATENCY_REL_TOL,
        help="Relative latency floor as a fraction of the baseline. Default %(default)s",
    )
    parser.add_argument(
        "--latency-abs-tol-ms",
        type=float,
        default=DEFAULT_LATENCY_ABS_TOL_MS,
        help="Absolute latency floor in milliseconds. Default %(default)s",
    )
    parser.add_argument(
        "--tail-rel-tol",
        type=float,
        default=DEFAULT_TAIL_REL_TOL,
        help="Relative floor for the p99 comparison. Default %(default)s",
    )
    parser.add_argument(
        "--noise-sigmas",
        type=float,
        default=DEFAULT_NOISE_SIGMAS,
        help=(
            "Multiples of the estimated standard error of the difference of "
            "medians that a latency move must clear. Default %(default)s"
        ),
    )
    parser.add_argument(
        "--auc-abs-tol",
        type=float,
        default=DEFAULT_AUC_ABS_TOL,
        help="Absolute AUC tolerance. Default %(default)s",
    )
    parser.add_argument(
        "--logloss-abs-tol",
        type=float,
        default=DEFAULT_LOGLOSS_ABS_TOL,
        help="Absolute logloss tolerance. Default %(default)s",
    )
    parser.add_argument(
        "--fail-on-tail",
        action="store_true",
        help=(
            "Promote a p99 regression to a failure. Off by default because p99 "
            "over a handful of timed batches is close to a single maximum."
        ),
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Also write the full findings as json to this path.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not os.path.exists(args.results):
        print(f"no benchmark results at {args.results}", file=sys.stderr)
        print(
            "run the benchmark first with python scripts/run_inference_benchmark.py",
            file=sys.stderr,
        )
        return EXIT_USAGE

    try:
        current_payload = load_json(args.results)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"could not read {args.results}. {exc}", file=sys.stderr)
        return EXIT_USAGE

    if not (current_payload.get("measurements") or current_payload.get("cells")):
        print(
            f"{args.results} has no measurements in it, so there is nothing to check",
            file=sys.stderr,
        )
        return EXIT_USAGE

    if args.update_baseline:
        baseline = make_baseline(current_payload, source=args.results)
        path = write_baseline(baseline, args.baseline)
        print(
            render_update_banner(
                path, len(baseline["cells"]), baseline["hardware_fingerprint"]
            )
        )
        return EXIT_OK

    if not os.path.exists(args.baseline):
        print(f"no baseline at {args.baseline}", file=sys.stderr)
        print(
            "record one from the current results with "
            f"python scripts/check_regression.py --results {args.results} "
            f"--baseline {args.baseline} --update-baseline",
            file=sys.stderr,
        )
        return EXIT_USAGE

    try:
        baseline_payload = load_json(args.baseline)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"could not read {args.baseline}. {exc}", file=sys.stderr)
        return EXIT_USAGE

    tolerances = Tolerances(
        latency_rel=args.latency_rel_tol,
        latency_abs_ms=args.latency_abs_tol_ms,
        tail_rel=args.tail_rel_tol,
        noise_sigmas=args.noise_sigmas,
        auc_abs=args.auc_abs_tol,
        logloss_abs=args.logloss_abs_tol,
        fail_on_tail=args.fail_on_tail,
    )

    report = compare(
        baseline_payload,
        current_payload,
        tolerances=tolerances,
        baseline_path=args.baseline,
        results_path=args.results,
        allow_hardware_mismatch=args.allow_hardware_mismatch,
    )

    print(render_report(report))

    if args.json_out:
        directory = os.path.dirname(os.path.abspath(args.json_out))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report.as_dict(), handle, indent=2)
            handle.write("\n")
        print(f"wrote the full findings to {args.json_out}")

    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
