"""Tests for the performance regression gate in `scripts/check_regression.py`.

The gate is the piece of the automation layer with real logic in it, so it is
the piece worth testing. Everything else in that layer either shells out to the
benchmark or formats a table.

Every test here builds its own benchmark payload in memory rather than reading
`results/`, so the suite does not depend on a benchmark having been run, does
not depend on which backends happen to be installed, and cannot be broken by
somebody rerunning the benchmark on a slightly different machine. The payloads
are the same shape the real artifact has, which is a hardware record plus a list
of measurements plus a list of backends that did not run.

What is asserted is the behaviour the gate exists for. A clear regression fires.
Noise does not. An improvement is reported and does not fail. A cell that
disappeared fires, because a silent fallback to a slower backend is a regression
that no latency comparison can see. An accuracy regression exits with its own
code and outranks a latency regression. A baseline from another machine is
refused rather than compared.

Everything here runs on a cpu only laptop in well under a second.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_gate():
    """Import scripts/check_regression.py by path, since scripts is not a package."""
    path = os.path.join(_REPO_ROOT, "scripts", "check_regression.py")
    spec = importlib.util.spec_from_file_location("check_regression", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_regression"] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

MAC = {
    "host": {
        "cpu_model": "Apple M3 Pro",
        "platform": "macOS-26.5.1-arm64-arm-64bit",
        "system": "Darwin",
        "machine": "arm64",
        "processor": "arm",
        "logical_cores": 12,
    },
    "python": {"version": "3.12.7", "implementation": "CPython"},
    "libraries": {
        "numpy": "2.5.2",
        "torch": {"version": "2.8.0", "cuda_available": False},
        "onnxruntime": {"version": "1.22.0"},
        "openvino": "2026.2.1",
        "tensorrt": {"version": "not available", "available": False},
    },
    "gpu": {"available": False, "name": "not available", "driver_version": "not available"},
}

A100 = {
    "host": {
        "cpu_model": "AMD EPYC 7763",
        "platform": "Linux-6.1-x86_64",
        "system": "Linux",
        "machine": "x86_64",
        "processor": "x86_64",
        "logical_cores": 64,
    },
    "python": {"version": "3.11.9", "implementation": "CPython"},
    "libraries": {
        "numpy": "2.1.0",
        "torch": {"version": "2.7.1", "cuda_available": True},
        "onnxruntime": {"version": "1.22.0"},
        "openvino": "not available",
        "tensorrt": {"version": "10.8.0.43", "available": True},
    },
    "gpu": {"available": True, "name": "NVIDIA A100 80GB PCIe", "driver_version": "570.86"},
}


def measurement(
    backend_key: str = "pytorch-cpu-fp32",
    batch_size: int = 1024,
    p50: float = 5.0,
    spread: float = 0.2,
    auc: float = 0.7827,
    logloss: float = 0.4512,
    repeats: int = 20,
    model: str = "DeepFM",
) -> Dict[str, Any]:
    """Build one measurement record.

    `spread` is the full min to p99 range. It is what the gate turns back into a
    standard deviation, so it is the knob that makes a cell quiet or noisy and
    therefore the knob that decides how large a move has to be before the gate
    is willing to call it real.
    """
    runtime, device, precision = backend_key.split("-")
    return {
        "model": model,
        "backend_key": backend_key,
        "label": f"{runtime} ({device}, {precision})",
        "short_label": runtime,
        "runtime": runtime,
        "precision": precision,
        "device": device,
        "lane": "gpu" if device == "cuda" else "cpu",
        "batch_size": batch_size,
        "effective_batch_size": batch_size,
        "n_timed_batches": 1,
        "repeats": repeats,
        "warmup": 10,
        "mean_ms": p50,
        "p50_ms": p50,
        "p99_ms": p50 + spread / 2.0,
        "min_ms": p50 - spread / 2.0,
        "throughput_samples_per_s": batch_size / (p50 / 1000.0),
        "auc": auc,
        "logloss": logloss,
    }


def payload(
    measurements: Optional[List[Dict[str, Any]]] = None,
    hardware: Optional[Dict[str, Any]] = None,
    unavailable: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Wrap measurements into the shape run_inference_benchmark.py writes."""
    if measurements is None:
        measurements = [
            measurement("pytorch-cpu-fp32"),
            measurement("openvino-cpu-fp32", p50=3.4, auc=0.7827),
        ]
    return {
        "hardware": copy.deepcopy(hardware if hardware is not None else MAC),
        "seed": 42,
        "n_test_rows": 10000,
        "arguments": {"sample_size": 100000, "repeats": 20},
        "batch_sizes": sorted({m["batch_size"] for m in measurements}),
        "measurements": measurements,
        "unavailable_backends": unavailable or [],
    }


def check(baseline: Dict[str, Any], current: Dict[str, Any], **kwargs: Any):
    """Run a comparison with the shipped default tolerances."""
    return gate.compare(baseline, current, **kwargs)


# ---------------------------------------------------------------------------
# Cell extraction
# ---------------------------------------------------------------------------


def test_cells_are_keyed_by_model_backend_and_batch_size():
    doc = payload(
        [
            measurement("pytorch-cpu-fp32", batch_size=1),
            measurement("pytorch-cpu-fp32", batch_size=4096),
            measurement("openvino-cpu-fp32", batch_size=1),
        ]
    )
    cells = gate.extract_cells(doc)
    assert set(cells) == {
        "DeepFM/pytorch-cpu-fp32/bs1",
        "DeepFM/pytorch-cpu-fp32/bs4096",
        "DeepFM/openvino-cpu-fp32/bs1",
    }
    # The key has to survive a markdown table, so it must not contain a pipe.
    assert all("|" not in key for key in cells)


def test_sample_count_is_repeats_times_timed_batches():
    doc = payload([measurement("pytorch-cpu-fp32", repeats=5)])
    record = doc["measurements"][0]
    record["n_timed_batches"] = 4
    cell = gate.extract_cells(doc)["DeepFM/pytorch-cpu-fp32/bs1024"]
    assert cell.n_samples == 20


# ---------------------------------------------------------------------------
# The core question. Does a real regression fire and does noise stay quiet
# ---------------------------------------------------------------------------


def test_identical_runs_pass_with_no_findings():
    doc = payload()
    report = check(doc, copy.deepcopy(doc))
    assert report.exit_code == gate.EXIT_OK
    assert report.latency_regressions == []
    assert report.latency_improvements == []
    assert report.accuracy_regressions == []
    assert report.missing_cells == []


def test_a_clear_regression_fires():
    """A fifty percent slowdown on a quiet cell is not something to argue about."""
    base = payload([measurement("openvino-cpu-fp32", p50=4.0, spread=0.2)])
    now = payload([measurement("openvino-cpu-fp32", p50=6.0, spread=0.2)])
    report = check(base, now)
    assert report.exit_code == gate.EXIT_LATENCY_REGRESSION
    assert len(report.latency_regressions) == 1
    finding = report.latency_regressions[0]
    assert finding.cell_key == "DeepFM/openvino-cpu-fp32/bs1024"
    assert finding.delta == pytest.approx(2.0)
    assert finding.rel_delta == pytest.approx(0.5)


def test_noise_on_a_jittery_cell_does_not_fire():
    """A move well inside the run to run spread is not evidence of anything.

    The baseline cell here is deliberately noisy, three timed samples spanning
    two milliseconds around a five millisecond median, which is what a laptop
    actually produces. A move of half a millisecond clears both fixed floors and
    is still far smaller than the uncertainty in the median itself, so the gate
    has to stay quiet. This is the case that makes a naive threshold useless.
    """
    base = payload([measurement("pytorch-cpu-fp32", p50=5.0, spread=2.0, repeats=3)])
    now = payload([measurement("pytorch-cpu-fp32", p50=5.6, spread=2.0, repeats=3)])
    report = check(base, now)
    assert report.exit_code == gate.EXIT_OK
    assert report.latency_regressions == []
    finding = report.latency[0]
    assert finding.verdict == gate.VERDICT_NOISE
    assert "noise floor" in finding.note


def test_the_absolute_floor_stops_a_large_relative_move_on_a_tiny_measurement():
    """Twenty percent of a twentieth of a millisecond is not a regression."""
    base = payload([measurement("openvino-cpu-fp32", p50=0.050, spread=0.002)])
    now = payload([measurement("openvino-cpu-fp32", p50=0.060, spread=0.002)])
    report = check(base, now)
    assert report.exit_code == gate.EXIT_OK
    finding = report.latency[0]
    assert finding.verdict == gate.VERDICT_NOISE
    assert finding.rel_delta == pytest.approx(0.2)
    assert "absolute floor" in finding.note


def test_the_relative_floor_stops_a_small_move_on_a_slow_cell():
    """Two milliseconds on a four hundred millisecond cell is half a percent."""
    base = payload([measurement("onnxruntime-cpu-fp32", p50=400.0, spread=1.0)])
    now = payload([measurement("onnxruntime-cpu-fp32", p50=402.0, spread=1.0)])
    report = check(base, now)
    assert report.exit_code == gate.EXIT_OK
    finding = report.latency[0]
    assert finding.verdict == gate.VERDICT_NOISE
    assert "relative floor" in finding.note


def test_a_quiet_cell_can_fire_on_a_move_that_a_noisy_cell_could_not():
    """The noise floor is per cell, which is the whole point of computing it.

    The same absolute and relative move is a regression on a backend with a tight
    distribution and is inside the noise on a backend with a wide one. A single
    global threshold cannot express that and would have to be set wrong for one
    of the two.
    """
    quiet_base = payload([measurement("openvino-cpu-fp32", p50=5.0, spread=0.05)])
    quiet_now = payload([measurement("openvino-cpu-fp32", p50=6.0, spread=0.05)])
    noisy_base = payload([measurement("openvino-cpu-fp32", p50=5.0, spread=6.0, repeats=3)])
    noisy_now = payload([measurement("openvino-cpu-fp32", p50=6.0, spread=6.0, repeats=3)])

    assert check(quiet_base, quiet_now).exit_code == gate.EXIT_LATENCY_REGRESSION
    assert check(noisy_base, noisy_now).exit_code == gate.EXIT_OK


# ---------------------------------------------------------------------------
# Improvements
# ---------------------------------------------------------------------------


def test_an_improvement_is_reported_and_does_not_fail():
    base = payload([measurement("openvino-cpu-fp32", p50=6.0, spread=0.2)])
    now = payload([measurement("openvino-cpu-fp32", p50=3.0, spread=0.2)])
    report = check(base, now)
    assert report.exit_code == gate.EXIT_OK
    assert len(report.latency_improvements) == 1
    finding = report.latency_improvements[0]
    assert finding.verdict == gate.VERDICT_IMPROVEMENT
    assert finding.delta == pytest.approx(-3.0)
    assert "improvement" in gate.render_report(report)


# ---------------------------------------------------------------------------
# Structure. Cells that appear and disappear
# ---------------------------------------------------------------------------


def test_a_disappeared_backend_fires_with_the_reason_the_benchmark_gave():
    """The failure a latency comparison structurally cannot catch.

    There is no slower row to compare against when a backend stops running. The
    row is simply gone, and a gate that only diffs numbers reports a clean pass
    on a build that lost its accelerated execution provider.
    """
    base = payload(
        [
            measurement("pytorch-cpu-fp32"),
            measurement("openvino-cpu-fp32", p50=3.4),
        ]
    )
    now = payload(
        [measurement("pytorch-cpu-fp32")],
        unavailable=[
            {
                "model": "DeepFM",
                "key": "openvino-cpu-fp32",
                "label": "OpenVINO (CPU, fp32)",
                "lane": "cpu",
                "precision": "fp32",
                "reason": "the openvino package is not installed on this host",
            }
        ],
    )
    report = check(base, now)
    assert report.exit_code == gate.EXIT_MISSING_CELL
    assert len(report.missing_cells) == 1
    key, _label, reason = report.missing_cells[0]
    assert key == "DeepFM/openvino-cpu-fp32/bs1024"
    assert "openvino package is not installed" in reason


def test_a_disappeared_cell_still_fires_when_the_run_gave_no_reason():
    base = payload([measurement("pytorch-cpu-fp32"), measurement("openvino-cpu-fp32")])
    now = payload([measurement("pytorch-cpu-fp32")])
    report = check(base, now)
    assert report.exit_code == gate.EXIT_MISSING_CELL
    assert "no measurement and no reason" in report.missing_cells[0][2]


def test_a_new_cell_is_reported_and_does_not_fail():
    base = payload([measurement("pytorch-cpu-fp32")])
    now = payload([measurement("pytorch-cpu-fp32"), measurement("openvino-cpu-fp32", p50=3.4)])
    report = check(base, now)
    assert report.exit_code == gate.EXIT_OK
    assert report.new_cells == [("DeepFM/openvino-cpu-fp32/bs1024", "openvino (cpu, fp32)")]


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def test_an_accuracy_regression_exits_with_the_accuracy_code():
    base = payload([measurement("openvino-cpu-fp32", auc=0.7827, logloss=0.4512)])
    now = payload([measurement("openvino-cpu-fp32", auc=0.7700, logloss=0.4512)])
    report = check(base, now)
    assert report.exit_code == gate.EXIT_ACCURACY_REGRESSION
    assert len(report.accuracy_regressions) == 1
    assert report.accuracy_regressions[0].metric == "auc"


def test_a_worse_logloss_is_an_accuracy_regression_even_when_auc_holds():
    base = payload([measurement("openvino-cpu-fp32", auc=0.7827, logloss=0.4512)])
    now = payload([measurement("openvino-cpu-fp32", auc=0.7827, logloss=0.4650)])
    report = check(base, now)
    assert report.exit_code == gate.EXIT_ACCURACY_REGRESSION
    assert [f.metric for f in report.accuracy_regressions] == ["logloss"]


def test_accuracy_has_no_noise_allowance_but_does_have_a_float_tolerance():
    """A float32 accumulation order difference must not fire the gate."""
    base = payload([measurement("openvino-cpu-fp32", auc=0.78270000)])
    now = payload([measurement("openvino-cpu-fp32", auc=0.78269998)])
    report = check(base, now)
    assert report.exit_code == gate.EXIT_OK
    assert report.accuracy_regressions == []


def test_an_accuracy_regression_outranks_a_latency_regression():
    """A faster model that ranks worse is not an improvement.

    Both findings are reported. The single exit code names the accuracy one,
    because that is the one that changes what gets shipped.
    """
    base = payload([measurement("openvino-cpu-fp32", p50=6.0, spread=0.2, auc=0.7827)])
    now = payload([measurement("openvino-cpu-fp32", p50=9.0, spread=0.2, auc=0.7600)])
    report = check(base, now)
    assert report.latency_regressions
    assert report.accuracy_regressions
    assert report.exit_code == gate.EXIT_ACCURACY_REGRESSION


def test_an_accuracy_improvement_is_reported_and_does_not_fail():
    base = payload([measurement("openvino-cpu-fp32", auc=0.7600)])
    now = payload([measurement("openvino-cpu-fp32", auc=0.7827)])
    report = check(base, now)
    assert report.exit_code == gate.EXIT_OK
    assert len(report.accuracy_improvements) == 1


# ---------------------------------------------------------------------------
# The tail
# ---------------------------------------------------------------------------


def test_a_tail_regression_is_reported_but_does_not_fail_by_default():
    base = payload([measurement("onnxruntime-cpu-fp32", p50=5.0, spread=1.0)])
    now = payload([measurement("onnxruntime-cpu-fp32", p50=5.0, spread=40.0)])
    report = check(base, now)
    assert report.tail_regressions
    assert report.exit_code == gate.EXIT_OK


def test_fail_on_tail_promotes_a_tail_regression_to_a_failure():
    base = payload([measurement("onnxruntime-cpu-fp32", p50=5.0, spread=1.0)])
    now = payload([measurement("onnxruntime-cpu-fp32", p50=5.0, spread=40.0)])
    report = check(base, now, tolerances=gate.Tolerances(fail_on_tail=True))
    assert report.exit_code == gate.EXIT_LATENCY_REGRESSION


# ---------------------------------------------------------------------------
# Hardware scoping
# ---------------------------------------------------------------------------


def test_a_baseline_from_another_machine_is_refused():
    """A latency baseline is a property of one machine and nothing else."""
    base = payload([measurement("pytorch-cpu-fp32", p50=5.0)], hardware=MAC)
    now = payload([measurement("pytorch-cpu-fp32", p50=1.0)], hardware=A100)
    report = check(base, now)
    assert report.exit_code == gate.EXIT_HARDWARE_MISMATCH
    assert report.hardware_refused
    # Nothing was compared, so no cell verdict was produced at all.
    assert report.latency == []
    assert report.accuracy == []
    assert report.missing_cells == []
    text = gate.render_report(report)
    assert "REFUSED" in text


def test_a_hardware_mismatch_can_be_forced_and_says_so_loudly():
    base = payload([measurement("pytorch-cpu-fp32", p50=5.0)], hardware=MAC)
    now = payload([measurement("pytorch-cpu-fp32", p50=1.0)], hardware=A100)
    report = check(base, now, allow_hardware_mismatch=True)
    assert not report.hardware_refused
    assert report.latency
    assert any("comparison of two machines" in note for note in report.notes)


def test_a_changed_gpu_driver_is_a_hardware_mismatch():
    """The driver is part of the machine for the purpose of a latency number."""
    other = copy.deepcopy(A100)
    other["gpu"]["driver_version"] = "580.12"
    base = payload([measurement("pytorch-cuda-fp32")], hardware=A100)
    now = payload([measurement("pytorch-cuda-fp32")], hardware=other)
    report = check(base, now)
    assert report.exit_code == gate.EXIT_HARDWARE_MISMATCH


def test_a_library_bump_warns_and_still_compares():
    """A dependency bump moving the latency is the event the gate exists for.

    Refusing to compare on a version change would hide exactly the regression
    worth catching, so a moved library version is printed as a warning next to a
    comparison that went ahead.
    """
    newer = copy.deepcopy(MAC)
    newer["libraries"]["torch"] = {"version": "2.9.0", "cuda_available": False}
    base = payload([measurement("pytorch-cpu-fp32", p50=4.0, spread=0.2)], hardware=MAC)
    now = payload([measurement("pytorch-cpu-fp32", p50=6.0, spread=0.2)], hardware=newer)
    report = check(base, now)
    assert not report.hardware_refused
    assert ("torch", "2.8.0", "2.9.0") in report.version_diffs
    assert report.exit_code == gate.EXIT_LATENCY_REGRESSION


# ---------------------------------------------------------------------------
# Baseline files and the command line
# ---------------------------------------------------------------------------


def test_a_baseline_round_trips_without_changing_a_verdict():
    """Writing a baseline and reading it back has to preserve the noise estimate.

    The sample count and the spread are what the noise floor is built from, so a
    baseline that dropped either of them would silently change how the gate
    behaves on the very next run.
    """
    doc = payload([measurement("pytorch-cpu-fp32", p50=5.0, spread=0.3, repeats=7)])
    baseline = gate.make_baseline(doc, source="results/inference_benchmark.json")
    round_tripped = json.loads(json.dumps(baseline))

    original = gate.extract_cells(doc)["DeepFM/pytorch-cpu-fp32/bs1024"]
    restored = gate.extract_cells(round_tripped)["DeepFM/pytorch-cpu-fp32/bs1024"]
    assert restored.n_samples == original.n_samples == 7
    assert restored.sigma_ms == pytest.approx(original.sigma_ms)
    assert restored.auc == original.auc

    report = check(round_tripped, doc)
    assert report.exit_code == gate.EXIT_OK


def test_a_baseline_carries_its_hardware_record():
    doc = payload()
    baseline = gate.make_baseline(doc, source="results/inference_benchmark.json")
    assert baseline["hardware_fingerprint"]["cpu_model"] == "Apple M3 Pro"
    assert baseline["hardware"]["host"]["logical_cores"] == 12
    assert baseline["library_versions"]["torch"] == "2.8.0"
    assert baseline["baseline_version"] == gate.BASELINE_VERSION


def test_update_baseline_writes_the_file_and_shouts_about_it(tmp_path, capsys):
    results = tmp_path / "inference_benchmark.json"
    results.write_text(json.dumps(payload()), encoding="utf-8")
    baseline = tmp_path / "baselines" / "cpu_only.json"

    code = gate.main(
        ["--results", str(results), "--baseline", str(baseline), "--update-baseline"]
    )
    assert code == gate.EXIT_OK
    assert baseline.exists()

    printed = capsys.readouterr().out
    assert "BASELINE REWRITTEN" in printed
    assert "Read the diff before committing this file." in printed


def test_main_gates_a_regression_end_to_end(tmp_path, capsys):
    baseline_doc = payload([measurement("openvino-cpu-fp32", p50=3.0, spread=0.1)])
    slower_doc = payload([measurement("openvino-cpu-fp32", p50=9.0, spread=0.1)])

    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(gate.make_baseline(baseline_doc, source="synthetic")), encoding="utf-8"
    )
    results = tmp_path / "results.json"
    results.write_text(json.dumps(slower_doc), encoding="utf-8")
    findings = tmp_path / "findings.json"

    code = gate.main(
        [
            "--results",
            str(results),
            "--baseline",
            str(baseline),
            "--json-out",
            str(findings),
        ]
    )
    assert code == gate.EXIT_LATENCY_REGRESSION
    assert "FAIL" in capsys.readouterr().out

    written = json.loads(findings.read_text(encoding="utf-8"))
    assert written["exit_code"] == gate.EXIT_LATENCY_REGRESSION
    assert written["latency"][0]["verdict"] == gate.VERDICT_REGRESSION


def test_main_reports_a_missing_baseline_as_a_usage_error(tmp_path, capsys):
    results = tmp_path / "results.json"
    results.write_text(json.dumps(payload()), encoding="utf-8")
    code = gate.main(
        ["--results", str(results), "--baseline", str(tmp_path / "nope.json")]
    )
    assert code == gate.EXIT_USAGE
    assert "--update-baseline" in capsys.readouterr().err


def test_main_reports_missing_results_as_a_usage_error(tmp_path, capsys):
    code = gate.main(["--results", str(tmp_path / "nothing.json")])
    assert code == gate.EXIT_USAGE
    assert "run_inference_benchmark.py" in capsys.readouterr().err


def test_tolerances_are_configurable_from_the_command_line(tmp_path):
    """A tighter relative floor turns a move the defaults ignore into a failure."""
    base_doc = payload([measurement("openvino-cpu-fp32", p50=5.00, spread=0.02)])
    now_doc = payload([measurement("openvino-cpu-fp32", p50=5.20, spread=0.02)])

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(gate.make_baseline(base_doc, "synthetic")), encoding="utf-8")
    results = tmp_path / "results.json"
    results.write_text(json.dumps(now_doc), encoding="utf-8")

    argv = ["--results", str(results), "--baseline", str(baseline)]
    assert gate.main(argv) == gate.EXIT_OK
    assert gate.main(argv + ["--latency-rel-tol", "0.02"]) == gate.EXIT_LATENCY_REGRESSION


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_a_cell_with_no_usable_measurement_is_not_comparable_rather_than_a_failure():
    base = payload([measurement("pytorch-cpu-fp32")])
    now = payload([measurement("pytorch-cpu-fp32")])
    now["measurements"][0]["p50_ms"] = None
    now["measurements"][0]["auc"] = float("nan")
    report = check(base, now)
    assert report.exit_code == gate.EXIT_OK
    assert report.latency[0].verdict == gate.VERDICT_UNKNOWN
    assert any(f.verdict == gate.VERDICT_UNKNOWN for f in report.accuracy)


def test_the_report_renders_for_every_outcome():
    """The renderer runs on real report objects, so a formatting slip fails here."""
    base = payload([measurement("pytorch-cpu-fp32", p50=4.0, spread=0.2)])
    for now in (
        payload([measurement("pytorch-cpu-fp32", p50=4.0, spread=0.2)]),
        payload([measurement("pytorch-cpu-fp32", p50=9.0, spread=0.2)]),
        payload([measurement("pytorch-cpu-fp32", p50=1.0, spread=0.2)]),
        payload([measurement("pytorch-cpu-fp32", p50=4.0, spread=0.2, auc=0.5)]),
        payload([measurement("openvino-cpu-fp32", p50=4.0, spread=0.2)]),
    ):
        text = gate.render_report(check(base, now))
        assert "ClickRank inference regression gate" in text
        assert "exit code" in text
