#!/usr/bin/env python
"""Drive the ranking service at controlled concurrency and report what breaks first.

This is the point of the serving layer. A single process batch timing loop, the
kind docs/INFERENCE.md reports, answers how fast the model runs when nothing
else is happening. A serving system is never in that state. Requests arrive
while other requests are in flight, they queue for a worker, and the number that
decides whether an auction is won is the tail of that whole path rather than the
mean of the model call inside it. Those are different quantities and the second
one is the honest one.

What this script measures.

**Closed loop, not open loop.** A fixed number of virtual clients each send one
request, wait for the response, and immediately send the next. The offered load
is therefore a consequence of the service's own speed rather than an independent
variable. That is the right model for a caller with a bounded connection pool,
which is what an upstream ad server is, and it is the wrong model for traffic
that arrives on its own schedule regardless of how the service is doing. An open
loop generator fires at a fixed rate and queues locally when the service falls
behind, which measures a different and usually worse tail. Saying which one was
run is not a formality, because the two produce different numbers from the same
service and reporting one as the other is a common and invisible error.

**Coordinated omission is bounded rather than eliminated.** A closed loop
generator cannot issue the next request until the previous one returns, so a
slow response suppresses the requests that would have been sent during it, and
the latencies that never got sampled were exactly the ones that would have been
worst. That is coordinated omission, and it means a closed loop tail is a
floor on the real tail rather than an estimate of it. It is not corrected here
because correcting it properly needs an open loop generator with an intended
send schedule, which is a different tool. What is done instead is to state the
bias, to hold the client count fixed and known so the bias is at least
consistent across cells, and to report the achieved concurrency next to the
requested one so a cell where the clients were the bottleneck is visible.

**The knee, not the maximum.** Peak throughput is a stress test answer. The
number a capacity plan is built on is the concurrency past which the tail
degrades faster than the throughput improves, because past that point more load
buys queueing rather than work. That crossover is reported per candidate set
size, and when the sweep never reaches it the report says so rather than
inventing one.

**Against a stated budget.** Throughput with no latency bound is not a serving
result. The report states a service level objective up front and answers the
only question that matters, which is the highest sustainable request rate that
holds the tail under it.

Run from the repository root, with or without a service already running.

    python scripts/run_load_test.py
    python scripts/run_load_test.py --url http://127.0.0.1:8000 --no-spawn
    python scripts/run_load_test.py --concurrency 1,4,16 --candidates 32 --duration 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Insert the repository root onto sys.path so that "import src" works when this
# script is run directly from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

# Select a non interactive backend before importing pyplot so the charts render
# on a headless machine with no display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  import after backend selection
import numpy as np  # noqa: E402

from src.inference.common import NOT_AVAILABLE, jsonable  # noqa: E402
from src.inference.hardware import collect_hardware_record  # noqa: E402
from src.schema import CAT_COLS, NUM_COLS  # noqa: E402

# The serving budget this report is graded against, in milliseconds of end to
# end p99 for one auction.
#
# The derivation, which is stated so it can be argued with rather than accepted.
# A page render budget of a few hundred milliseconds leaves the ad request
# roughly one hundred milliseconds end to end. Out of that the exchange, the
# network hops, the retrieval stage that narrows millions of candidates to a
# shortlist, the auction and pricing logic, and the creative selection all take
# their share, which leaves the ranking model a slice measured in tens of
# milliseconds rather than in hundreds. docs/INFERENCE.md makes the same
# argument from the other direction. Twenty five milliseconds at p99 is a
# defensible slice of that, tight enough that missing it is a real engineering
# problem and loose enough that meeting it is not trivially easy.
DEFAULT_SLO_MS: float = 25.0

# The concurrency ladder. It doubles by fours so five points cover more than two
# orders of magnitude, which is enough to see a knee without a sweep that takes
# an afternoon.
DEFAULT_CONCURRENCY: Tuple[int, ...] = (1, 4, 16, 64, 256)

# Candidate set sizes per request. One is the degenerate single ad case, and the
# larger sizes are what a real auction shortlist looks like after retrieval has
# narrowed the pool.
DEFAULT_CANDIDATES: Tuple[int, ...] = (1, 16, 64)


def parse_args() -> argparse.Namespace:
    """Parse the command line flags for the load test."""
    parser = argparse.ArgumentParser(
        description="Load test the ClickRank ranking service."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Base url of the service under test.",
    )
    parser.add_argument(
        "--spawn",
        dest="spawn",
        action="store_true",
        default=None,
        help="Start a service subprocess for this run. The default is to spawn "
        "one only when nothing is already answering at --url.",
    )
    parser.add_argument(
        "--no-spawn",
        dest="spawn",
        action="store_false",
        help="Never start a service. Fail when nothing answers at --url.",
    )
    parser.add_argument(
        "--concurrency",
        default=",".join(str(c) for c in DEFAULT_CONCURRENCY),
        help="Comma separated closed loop client counts to sweep.",
    )
    parser.add_argument(
        "--candidates",
        default=",".join(str(c) for c in DEFAULT_CANDIDATES),
        help="Comma separated candidate set sizes per request.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="Seconds of measured load per cell.",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=2.0,
        help="Seconds of load per cell that are run and then discarded.",
    )
    parser.add_argument(
        "--slo-ms",
        type=float,
        default=DEFAULT_SLO_MS,
        help="End to end p99 budget in milliseconds that the report grades against.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=20000,
        help="Rows of realistic request payload to draw candidate sets from.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the payload sampling."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Thread pool size passed to a spawned service. Its queueing parameter.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("results", "serving"),
        help="Directory the json, the markdown, and the charts are written to.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per request client timeout in seconds.",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Free text label stamped onto every number in the report.",
    )
    return parser.parse_args()


def _int_list(text: str) -> List[int]:
    """Parse a comma separated list of positive integers."""
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values or any(v < 1 for v in values):
        raise ValueError(f"expected a comma separated list of positive integers, got {text}")
    return values


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


def build_payload_rows(n_rows: int, seed: int) -> List[Dict[str, Any]]:
    """Draw realistic request rows from the project's own synthetic generator.

    Using the generator rather than random strings matters more than it looks
    like it does. The feature pipeline hashes every categorical value and looks
    every one of them up in a train frequency table, so the cost of featurizing
    a row depends on the shape of its values. Rows made of arbitrary noise would
    all miss the table and take one branch, which would measure a case that
    never occurs.
    """
    from src.data.loader import generate_synthetic

    frame = generate_synthetic(int(n_rows), seed=int(seed))
    columns = list(NUM_COLS) + list(CAT_COLS)
    records = frame[columns].to_dict(orient="records")
    rows: List[Dict[str, Any]] = []
    for record in records:
        row: Dict[str, Any] = {}
        for key, value in record.items():
            if key in CAT_COLS:
                row[key] = str(value)
            else:
                number = float(value)
                # json has no NaN, so a missing dense value goes over the wire as
                # null, which is exactly how a caller says a field is absent.
                row[key] = None if number != number else number
        rows.append(row)
    return rows


def build_requests(
    rows: Sequence[Dict[str, Any]], n_candidates: int, n_requests: int, seed: int
) -> List[Dict[str, Any]]:
    """Precompute a pool of request bodies so serialization is not in the timer.

    Building the json payload is client side work. Leaving it inside the request
    loop would fold the generator's own cost into the measured latency, which is
    the client side version of the mistake docs/INFERENCE.md warns about for
    kernel launches.
    """
    rng = np.random.default_rng(seed)
    pool: List[Dict[str, Any]] = []
    for i in range(n_requests):
        picks = rng.integers(0, len(rows), size=n_candidates)
        pool.append(
            {
                "request_id": f"load-{n_candidates}-{i}",
                "candidates": [
                    {"ad_id": f"ad-{j}", "features": rows[int(p)]}
                    for j, p in enumerate(picks)
                ],
            }
        )
    return pool


# ---------------------------------------------------------------------------
# The closed loop generator
# ---------------------------------------------------------------------------


class CellResult:
    """Everything one concurrency by candidate size cell measured."""

    def __init__(self, concurrency: int, n_candidates: int) -> None:
        self.concurrency = int(concurrency)
        self.n_candidates = int(n_candidates)
        self.latencies_ms: List[float] = []
        self.server_total_ms: List[float] = []
        self.server_feature_ms: List[float] = []
        self.server_model_ms: List[float] = []
        self.errors: int = 0
        self.error_examples: List[str] = []
        self.wall_seconds: float = 0.0
        self.load_average: Optional[List[float]] = None

    @property
    def completed(self) -> int:
        return len(self.latencies_ms)

    @property
    def total_attempts(self) -> int:
        return self.completed + self.errors

    def percentile(self, q: float) -> float:
        if not self.latencies_ms:
            return float("nan")
        return float(np.percentile(np.asarray(self.latencies_ms, dtype=np.float64), q))

    def as_dict(self) -> Dict[str, Any]:
        """Return the json friendly record for this cell."""
        rps = self.completed / self.wall_seconds if self.wall_seconds > 0 else float("nan")
        latencies = np.asarray(self.latencies_ms, dtype=np.float64)
        mean_latency = float(np.mean(latencies)) if latencies.size else float("nan")
        # Little's law read backwards. With a closed loop generator the achieved
        # concurrency should sit at the requested client count, and a gap means
        # the clients themselves were the bottleneck rather than the service.
        achieved = (rps * mean_latency / 1000.0) if latencies.size else float("nan")
        return {
            "concurrency": self.concurrency,
            "n_candidates": self.n_candidates,
            "completed": self.completed,
            "errors": self.errors,
            "error_rate": (
                self.errors / self.total_attempts if self.total_attempts else 0.0
            ),
            "error_examples": self.error_examples[:3],
            "wall_seconds": round(self.wall_seconds, 4),
            "throughput_rps": rps,
            "throughput_ads_per_second": rps * self.n_candidates,
            "latency_ms": {
                "mean": mean_latency,
                "p50": self.percentile(50),
                "p95": self.percentile(95),
                "p99": self.percentile(99),
                "p999": self.percentile(99.9),
                "max": float(np.max(latencies)) if latencies.size else float("nan"),
            },
            "server_side_ms": {
                "total_mean": _safe_mean(self.server_total_ms),
                "feature_mean": _safe_mean(self.server_feature_ms),
                "model_mean": _safe_mean(self.server_model_ms),
            },
            "achieved_concurrency": achieved,
            "host_load_average": self.load_average,
        }


def _safe_mean(values: Sequence[float]) -> float:
    """Mean of a list, or nan when it is empty."""
    return float(statistics.fmean(values)) if values else float("nan")


async def _client_loop(
    client,
    url: str,
    pool: Sequence[Dict[str, Any]],
    deadline: float,
    offset: int,
    record: Optional[CellResult],
) -> None:
    """One virtual client. Send, wait, send again, until the deadline.

    Passing record as None is the warmup mode. The requests are still sent and
    the service still does the work, which is the point, and nothing is
    recorded.
    """
    index = offset
    while time.perf_counter() < deadline:
        body = pool[index % len(pool)]
        index += 1
        started = time.perf_counter()
        try:
            response = await client.post(url, json=body)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if response.status_code != 200:
                if record is not None:
                    record.errors += 1
                    if len(record.error_examples) < 3:
                        record.error_examples.append(
                            f"http {response.status_code} {response.text[:160]}"
                        )
                continue
            if record is not None:
                record.latencies_ms.append(elapsed_ms)
                timings = response.json().get("timings", {})
                record.server_total_ms.append(float(timings.get("total_ms", float("nan"))))
                record.server_feature_ms.append(
                    float(timings.get("feature_ms", float("nan")))
                )
                record.server_model_ms.append(float(timings.get("model_ms", float("nan"))))
        except Exception as exc:  # noqa: BLE001 a client side failure is a data point
            if record is not None:
                record.errors += 1
                if len(record.error_examples) < 3:
                    record.error_examples.append(f"{type(exc).__name__} {exc}")


async def run_cell(
    base_url: str,
    pool: Sequence[Dict[str, Any]],
    concurrency: int,
    n_candidates: int,
    duration: float,
    warmup: float,
    timeout: float,
) -> CellResult:
    """Run one cell of the sweep with a warmup phase that is thrown away."""
    import httpx

    url = base_url.rstrip("/") + "/score"
    limits = httpx.Limits(
        max_connections=concurrency + 8, max_keepalive_connections=concurrency + 8
    )
    result = CellResult(concurrency, n_candidates)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        if warmup > 0:
            deadline = time.perf_counter() + warmup
            await asyncio.gather(
                *[
                    _client_loop(client, url, pool, deadline, i, None)
                    for i in range(concurrency)
                ]
            )
        result.load_average = load_average()
        started = time.perf_counter()
        deadline = started + duration
        await asyncio.gather(
            *[
                _client_loop(client, url, pool, deadline, i, result)
                for i in range(concurrency)
            ]
        )
        result.wall_seconds = time.perf_counter() - started
    return result


# ---------------------------------------------------------------------------
# Service lifecycle
# ---------------------------------------------------------------------------


def service_health(base_url: str, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    """Return the health document, or None when nothing is answering."""
    import httpx

    try:
        response = httpx.get(base_url.rstrip("/") + "/health", timeout=timeout)
    except Exception:  # noqa: BLE001 nothing there is an answer, not an error
        return None
    if response.status_code != 200:
        return None
    return response.json()


def spawn_service(
    base_url: str, threads: int, wait_seconds: float = 600.0
) -> subprocess.Popen:
    """Start scripts/serve.py as a subprocess and wait for it to report healthy.

    Starting the service from inside the load test is what makes one command
    reproduce the whole report. The first start can be slow because it may have
    to build the serving bundle, which trains a checkpoint when none exists, so
    the wait is generous and it prints what it is waiting for.
    """
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    command = [
        sys.executable,
        os.path.join(_REPO_ROOT, "scripts", "serve.py"),
        "--host",
        str(host),
        "--port",
        str(port),
        "--threads",
        str(threads),
        "--log-level",
        "warning",
    ]
    print(f"starting a service for this run. {' '.join(command)}")
    process = subprocess.Popen(command, cwd=_REPO_ROOT)

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"the service exited during startup with code {process.returncode}. "
                "Run python scripts/serve.py on its own to see why."
            )
        if service_health(base_url) is not None:
            print("the service is healthy and the sweep can start.")
            return process
        time.sleep(1.0)

    process.terminate()
    raise TimeoutError(
        f"the service did not become healthy within {wait_seconds:.0f} seconds."
    )


def load_average() -> Optional[List[float]]:
    """Return the one, five, and fifteen minute load averages when the host has them.

    This is recorded on purpose. A latency measurement taken while the machine
    is running something else is not a measurement of the service, and the only
    way a reader can tell is if the number is in the report.
    """
    try:
        return [round(v, 2) for v in os.getloadavg()]
    except (AttributeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def find_knee(cells: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Locate the concurrency past which the tail costs more than the throughput buys.

    The rule is a ratio of ratios. Stepping from one concurrency level to the
    next multiplies throughput by some gain and multiplies p99 by some cost. As
    long as the gain exceeds the cost, the extra load is buying work. When the
    cost exceeds the gain, the extra load is buying queue depth, and the last
    level before that is the knee.

    This is deliberately a local rule rather than a curve fit. A fitted
    scalability model would produce a smooth answer from five points, which
    would look more authoritative than five points deserve.
    """
    ordered = sorted(cells, key=lambda c: c["concurrency"])
    steps: List[Dict[str, Any]] = []
    knee: Optional[int] = None
    for previous, current in zip(ordered, ordered[1:]):
        prev_tput = float(previous["throughput_rps"])
        cur_tput = float(current["throughput_rps"])
        prev_p99 = float(previous["latency_ms"]["p99"])
        cur_p99 = float(current["latency_ms"]["p99"])
        if not (prev_tput > 0 and prev_p99 > 0):
            continue
        gain = cur_tput / prev_tput
        cost = cur_p99 / prev_p99
        efficiency = gain / cost if cost > 0 else float("inf")
        steps.append(
            {
                "from_concurrency": previous["concurrency"],
                "to_concurrency": current["concurrency"],
                "throughput_gain": gain,
                "p99_cost": cost,
                "efficiency": efficiency,
            }
        )
        if knee is None and efficiency < 1.0:
            knee = int(previous["concurrency"])

    if knee is None:
        return {
            "knee_concurrency": None,
            "reached": False,
            "note": (
                "the sweep never reached a knee. At every step the throughput "
                "gain was at least as large as the p99 cost, so this service has "
                "headroom above the highest concurrency tested and the sweep "
                "would have to be extended to find the operating point"
            ),
            "steps": steps,
        }
    return {
        "knee_concurrency": knee,
        "reached": True,
        "note": (
            f"past {knee} concurrent clients the p99 grows faster than the "
            "throughput does, so additional load buys queue depth rather than "
            "work"
        ),
        "steps": steps,
    }


def slo_summary(cells: Sequence[Dict[str, Any]], slo_ms: float) -> Dict[str, Any]:
    """Find the highest throughput that held the p99 under the budget.

    A cell only counts when it served every request. Throughput bought with
    errors is not throughput.
    """
    passing = [
        c
        for c in cells
        if c["errors"] == 0 and float(c["latency_ms"]["p99"]) <= slo_ms
    ]
    if not passing:
        best = min(cells, key=lambda c: float(c["latency_ms"]["p99"])) if cells else None
        return {
            "slo_ms": slo_ms,
            "met": False,
            "max_sustainable_rps": None,
            "at_concurrency": None,
            "note": (
                "no cell in this sweep held the p99 under the budget. The closest "
                f"was {best['latency_ms']['p99']:.1f} ms at concurrency "
                f"{best['concurrency']} with {best['n_candidates']} candidates"
                if best
                else "no cells were measured"
            ),
        }
    best = max(passing, key=lambda c: float(c["throughput_rps"]))
    return {
        "slo_ms": slo_ms,
        "met": True,
        "max_sustainable_rps": float(best["throughput_rps"]),
        "max_sustainable_ads_per_second": float(best["throughput_ads_per_second"]),
        "at_concurrency": int(best["concurrency"]),
        "at_candidates": int(best["n_candidates"]),
        "p99_ms": float(best["latency_ms"]["p99"]),
        "note": (
            f"{best['throughput_rps']:.1f} requests per second at concurrency "
            f"{best['concurrency']} with {best['n_candidates']} candidates per "
            f"request, holding a p99 of {best['latency_ms']['p99']:.1f} ms under "
            f"the {slo_ms:.0f} ms budget"
        ),
    }


# ---------------------------------------------------------------------------
# Charts and report
# ---------------------------------------------------------------------------


def plot_throughput(results: Dict[str, Any], path: str) -> str:
    """Throughput against concurrency, one line per candidate set size."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for size in results["candidate_sizes"]:
        cells = [c for c in results["cells"] if c["n_candidates"] == size]
        cells.sort(key=lambda c: c["concurrency"])
        ax.plot(
            [c["concurrency"] for c in cells],
            [c["throughput_rps"] for c in cells],
            marker="o",
            label=f"{size} candidates per request",
        )
    knees = results.get("knees", {})
    for size, knee in knees.items():
        if knee.get("reached"):
            ax.axvline(
                knee["knee_concurrency"],
                linestyle=":",
                alpha=0.5,
                color="gray",
            )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("closed loop concurrency (virtual clients)")
    ax.set_ylabel("throughput (requests per second)")
    ax.set_title(f"Throughput against concurrency\n{results['hardware_label']}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_p99(results: Dict[str, Any], path: str) -> str:
    """Tail latency against concurrency, with the budget drawn on it."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for size in results["candidate_sizes"]:
        cells = [c for c in results["cells"] if c["n_candidates"] == size]
        cells.sort(key=lambda c: c["concurrency"])
        ax.plot(
            [c["concurrency"] for c in cells],
            [c["latency_ms"]["p99"] for c in cells],
            marker="o",
            label=f"{size} candidates per request",
        )
    slo = results["slo"]["slo_ms"]
    ax.axhline(slo, color="crimson", linestyle="--", linewidth=1.2)
    ax.text(
        1.0,
        slo,
        f" {slo:.0f} ms p99 budget",
        color="crimson",
        va="bottom",
        fontsize=8,
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("closed loop concurrency (virtual clients)")
    ax.set_ylabel("p99 end to end latency (ms)")
    ax.set_title(f"Tail latency against concurrency\n{results['hardware_label']}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_distribution(
    results: Dict[str, Any], samples: Dict[str, List[float]], path: str
) -> str:
    """The latency distribution itself, as a complementary cumulative curve.

    A histogram of a latency distribution wastes its resolution on the bulk,
    which is the part nobody argues about. The complementary cumulative
    distribution plotted on a log vertical axis gives the tail a whole decade per
    order of magnitude, so the difference between a p99 and a p999 is legible
    instead of being one pixel above the axis.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for label, values in samples.items():
        if not values:
            continue
        arr = np.sort(np.asarray(values, dtype=np.float64))
        survival = 1.0 - np.arange(1, arr.size + 1) / arr.size
        ax.plot(arr, np.clip(survival, 1.0 / arr.size, None), label=label)
    slo = results["slo"]["slo_ms"]
    ax.axvline(slo, color="crimson", linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("end to end latency (ms)")
    ax.set_ylabel("fraction of requests slower than x")
    ax.set_title(
        f"Latency distribution tail\n{results['hardware_label']}"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _fmt(value: Any, spec: str = ".1f") -> str:
    """Format a number for a report cell, or the not available marker."""
    if value is None:
        return NOT_AVAILABLE
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return NOT_AVAILABLE
    return format(number, spec)


def write_markdown(results: Dict[str, Any], path: str) -> str:
    """Write the human readable load test report."""
    lines: List[str] = []
    lines.append("# Serving Load Test")
    lines.append("")
    lines.append(
        "Closed loop load test of the ClickRank ranking service. Every number "
        "on this page was measured on the machine named below and applies to "
        "that machine, that backend, and that configuration only."
    )
    lines.append("")

    lines.append("## Configuration")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Measured at (UTC) | {results['measured_at_utc']} |")
    lines.append(f"| Hardware | {results['hardware_label']} |")
    lines.append(f"| Logical cores | {results['logical_cores']} |")
    lines.append(f"| Backend | {results['backend_label']} |")
    lines.append(f"| Model | {results['model_name']} |")
    lines.append(f"| Service thread pool | {results['thread_pool_size']} |")
    lines.append(f"| Service workers | {results['service_workers']} |")
    lines.append(f"| Load generator | closed loop, asyncio and httpx |")
    lines.append(f"| Warmup per cell | {results['warmup_seconds']:.1f} s, discarded |")
    lines.append(f"| Measured per cell | {results['duration_seconds']:.1f} s |")
    lines.append(f"| p99 budget | {results['slo']['slo_ms']:.0f} ms |")
    load = results.get("load_average_at_start")
    lines.append(
        f"| Host load average at start | {load if load else NOT_AVAILABLE} |"
    )
    if results.get("label"):
        lines.append(f"| Run label | {results['label']} |")
    lines.append("")
    if results.get("host_was_busy"):
        lines.append(
            "**These numbers were taken on a busy host.** The one minute load "
            f"average at the start of the sweep was {load[0]:.1f} on "
            f"{results['logical_cores']} logical cores, which means other work "
            "was competing for the same cores as the service. Every latency on "
            "this page is inflated by that contention and should be read as an "
            "upper bound rather than as the service's own number. Rerun on an "
            "idle machine before quoting anything from it."
        )
        lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append(
        "Latency is end to end at the client, so it includes json serialization, "
        "the loopback network, the queue inside the service, the feature "
        "pipeline, and the model call. Throughput is completed requests divided "
        "by the measured wall time of the cell. Ad scores per second multiplies "
        "that by the candidate set size, which is the number a capacity plan "
        "actually consumes."
    )
    lines.append("")
    lines.append(
        "| Candidates | Concurrency | RPS | Ad scores/s | p50 ms | p95 ms | p99 ms | p999 ms | Errors | Feature ms | Model ms | Host load |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for cell in sorted(
        results["cells"], key=lambda c: (c["n_candidates"], c["concurrency"])
    ):
        latency = cell["latency_ms"]
        server = cell["server_side_ms"]
        lines.append(
            f"| {cell['n_candidates']} | {cell['concurrency']} | "
            f"{_fmt(cell['throughput_rps'])} | "
            f"{_fmt(cell['throughput_ads_per_second'], '.0f')} | "
            f"{_fmt(latency['p50'], '.2f')} | {_fmt(latency['p95'], '.2f')} | "
            f"{_fmt(latency['p99'], '.2f')} | {_fmt(latency['p999'], '.2f')} | "
            f"{_fmt(cell['error_rate'], '.4f')} | "
            f"{_fmt(server['feature_mean'], '.2f')} | "
            f"{_fmt(server['model_mean'], '.2f')} | "
            f"{_fmt(cell['host_load_average'][0] if cell.get('host_load_average') else None)} |"
        )
    lines.append("")

    lines.append("## Against the Budget")
    lines.append("")
    slo = results["slo"]
    lines.append(
        f"The budget is a p99 of {slo['slo_ms']:.0f} ms end to end for one "
        "auction. The derivation is in `docs/SERVING.md` and it is a slice of a "
        "roughly one hundred millisecond ad request, of which retrieval, the "
        "auction, and the network hops take the rest."
    )
    lines.append("")
    note = slo["note"]
    lines.append(f"**{note[:1].upper()}{note[1:]}.**")
    lines.append("")

    lines.append("## The Knee")
    lines.append("")
    lines.append(
        "The knee is the concurrency past which p99 climbs faster than "
        "throughput improves. It is the operating point a capacity plan is built "
        "on, because beyond it extra load buys queue depth rather than work."
    )
    lines.append("")
    lines.append(
        "| Candidates | Knee concurrency | Reading |"
    )
    lines.append("| --- | --- | --- |")
    for size in results["candidate_sizes"]:
        knee = results["knees"][str(size)]
        where = (
            str(knee["knee_concurrency"])
            if knee.get("reached")
            else "not reached in this sweep"
        )
        lines.append(f"| {size} | {where} | {knee['note']} |")
    lines.append("")
    lines.append("Step by step, where efficiency is the throughput gain divided by the p99 cost.")
    lines.append("")
    lines.append(
        "| Candidates | Step | Throughput gain | p99 cost | Efficiency |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for size in results["candidate_sizes"]:
        for step in results["knees"][str(size)]["steps"]:
            lines.append(
                f"| {size} | {step['from_concurrency']} to {step['to_concurrency']} | "
                f"{_fmt(step['throughput_gain'], '.2f')}x | "
                f"{_fmt(step['p99_cost'], '.2f')}x | "
                f"{_fmt(step['efficiency'], '.2f')} |"
            )
    lines.append("")

    lines.append("## How to Read These Numbers")
    lines.append("")
    lines.append(
        "This is a closed loop test. A fixed number of virtual clients each wait "
        "for a response before sending the next request, so the offered load is "
        "a consequence of the service's speed rather than an independent "
        "variable. An open loop test, which fires at a fixed rate whatever the "
        "service is doing, would report a worse tail from the same service. The "
        "two are not interchangeable and this page is the first kind."
    )
    lines.append("")
    lines.append(
        "Because of that, these tails are a floor rather than an estimate. A slow "
        "response suppresses the requests a client would otherwise have sent "
        "during it, and the suppressed requests are precisely the ones that would "
        "have been slowest. That is coordinated omission. It is not corrected "
        "here, it is stated, and the achieved concurrency column in the json "
        "exists so a cell where the clients rather than the service were the "
        "bottleneck can be spotted."
    )
    lines.append("")
    lines.append(
        "The feature and model columns are server side means reported by the "
        "service itself, and they are the reason the end to end number is what it "
        "is. They do not sum to the end to end latency, because the difference "
        "between them is queueing, json handling, and the loopback hop."
    )
    lines.append("")
    lines.append(
        "Every cell ran for a fixed wall time rather than a fixed request count, "
        "so a slow cell and a fast cell both get the same amount of clock and the "
        "percentiles of the fast cell rest on more samples rather than on a "
        "longer window."
    )
    lines.append("")

    lines.append("## Charts")
    lines.append("")
    for title, filename in results["charts"].items():
        lines.append(f"![{title}]({os.path.basename(filename)})")
        lines.append("")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_sweep(
    base_url: str,
    concurrency_levels: Sequence[int],
    candidate_sizes: Sequence[int],
    duration: float,
    warmup: float,
    timeout: float,
    rows: Sequence[Dict[str, Any]],
    seed: int,
) -> Tuple[List[CellResult], Dict[str, List[float]]]:
    """Run every cell of the sweep in order and return the results."""
    cells: List[CellResult] = []
    samples: Dict[str, List[float]] = {}
    for size in candidate_sizes:
        # One pool per candidate size, reused across the concurrency levels so
        # the payload is not a variable between them.
        pool = build_requests(rows, size, n_requests=64, seed=seed + size)
        for concurrency in concurrency_levels:
            print(
                f"  cell. {size} candidates, {concurrency} concurrent clients, "
                f"{warmup:.0f} s warmup then {duration:.0f} s measured."
            )
            cell = await run_cell(
                base_url, pool, concurrency, size, duration, warmup, timeout
            )
            record = cell.as_dict()
            print(
                f"    {record['completed']} requests, "
                f"{_fmt(record['throughput_rps'])} rps, p50 "
                f"{_fmt(record['latency_ms']['p50'], '.2f')} ms, p99 "
                f"{_fmt(record['latency_ms']['p99'], '.2f')} ms, "
                f"{record['errors']} errors."
            )
            cells.append(cell)
            samples[f"{size} candidates at concurrency {concurrency}"] = list(
                cell.latencies_ms
            )
    return cells, samples


def main() -> None:
    """Run the sweep and write the json, the markdown, and the charts."""
    args = parse_args()
    concurrency_levels = _int_list(args.concurrency)
    candidate_sizes = _int_list(args.candidates)

    health = service_health(args.url)
    process: Optional[subprocess.Popen] = None
    if health is None:
        if args.spawn is False:
            raise SystemExit(
                f"nothing is answering at {args.url} and --no-spawn was passed. "
                "Start the service with python scripts/serve.py first."
            )
        process = spawn_service(args.url, threads=args.threads)
        health = service_health(args.url)
    elif args.spawn:
        print(
            f"a service is already answering at {args.url}, so --spawn is "
            "ignored and the running one is used."
        )

    if health is None:
        raise SystemExit(f"the service at {args.url} never became healthy.")

    backend = health.get("backend", {})
    bundle = health.get("bundle", {})
    hardware = health.get("hardware", collect_hardware_record())
    host = hardware.get("host", {})
    hardware_label = (
        f"{host.get('cpu_model', NOT_AVAILABLE)}, "
        f"{host.get('logical_cores', NOT_AVAILABLE)} logical cores, "
        f"{backend.get('label', NOT_AVAILABLE)}"
    )

    print(f"loading payload rows. {args.rows} rows from the synthetic generator.")
    rows = build_payload_rows(args.rows, args.seed)

    print(
        f"sweeping {len(candidate_sizes)} candidate sizes by "
        f"{len(concurrency_levels)} concurrency levels against {args.url}."
    )
    started_load = load_average()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cores = host.get("logical_cores")
    if started_load and isinstance(cores, int) and started_load[0] > 0.5 * cores:
        print(
            f"warning. The one minute load average is {started_load[0]:.1f} on "
            f"{cores} logical cores, so this host is busy with other work. Every "
            "latency this run reports will be inflated by that contention. The "
            "report will say so, and the sweep should be rerun on an idle "
            "machine before any number from it is quoted."
        )

    try:
        cells, samples = asyncio.run(
            run_sweep(
                args.url,
                concurrency_levels,
                candidate_sizes,
                args.duration,
                args.warmup,
                args.timeout,
                rows,
                args.seed,
            )
        )
    finally:
        if process is not None:
            print("stopping the service this run started.")
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()

    cell_records = [c.as_dict() for c in cells]
    knees = {
        str(size): find_knee([c for c in cell_records if c["n_candidates"] == size])
        for size in candidate_sizes
    }

    results: Dict[str, Any] = {
        "measured_at_utc": started_at,
        "label": args.label,
        "hardware_label": hardware_label,
        "hardware": hardware,
        "logical_cores": host.get("logical_cores", NOT_AVAILABLE),
        "backend_key": backend.get("selected", NOT_AVAILABLE),
        "backend_label": backend.get("label", NOT_AVAILABLE),
        "backend_probe": backend.get("probed", []),
        "thread_pool_size": backend.get("thread_pool_size", NOT_AVAILABLE),
        "model_name": bundle.get("model_name", NOT_AVAILABLE),
        "bundle": bundle,
        "generator": "closed loop",
        "duration_seconds": args.duration,
        "warmup_seconds": args.warmup,
        "concurrency_levels": list(concurrency_levels),
        "candidate_sizes": list(candidate_sizes),
        "service_workers": 1 if process is not None else NOT_AVAILABLE,
        "load_average_at_start": started_load,
        "load_average_at_end": load_average(),
        "cells": cell_records,
        "knees": knees,
        "slo": slo_summary(cell_records, args.slo_ms),
        "host_was_busy": bool(
            started_load
            and isinstance(host.get("logical_cores"), int)
            and started_load[0] > 0.5 * int(host["logical_cores"])
        ),
        "coordinated_omission": (
            "this is a closed loop generator, so a slow response suppresses the "
            "requests that would have been sent during it and the reported tail "
            "is a floor on the real tail rather than an estimate of it"
        ),
    }

    os.makedirs(args.output, exist_ok=True)
    charts = {
        "Throughput against concurrency": os.path.join(
            args.output, "load_test_throughput.png"
        ),
        "Tail latency against concurrency": os.path.join(
            args.output, "load_test_p99.png"
        ),
        "Latency distribution tail": os.path.join(
            args.output, "load_test_latency_distribution.png"
        ),
    }
    results["charts"] = charts
    plot_throughput(results, charts["Throughput against concurrency"])
    plot_p99(results, charts["Tail latency against concurrency"])
    # The distribution chart shows one candidate size across every concurrency
    # level, because overlaying every cell would be fifteen curves and legible
    # as none of them.
    focus = candidate_sizes[len(candidate_sizes) // 2]
    plot_distribution(
        results,
        {k: v for k, v in samples.items() if k.startswith(f"{focus} candidates")},
        charts["Latency distribution tail"],
    )

    json_path = os.path.join(args.output, "load_test.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(jsonable(results), handle, indent=2)
        handle.write("\n")
    markdown_path = write_markdown(results, os.path.join(args.output, "load_test.md"))

    print("")
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    for path in charts.values():
        print(f"wrote {path}")
    print("")
    print(f"budget. {results['slo']['note']}.")
    for size in candidate_sizes:
        knee = knees[str(size)]
        where = (
            f"concurrency {knee['knee_concurrency']}"
            if knee.get("reached")
            else "not reached in this sweep"
        )
        print(f"knee at {size} candidates. {where}.")


if __name__ == "__main__":
    main()
