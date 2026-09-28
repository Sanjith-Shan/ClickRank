"""GPU power, utilization, and energy sampling for the inference benchmark.

Latency and throughput answer how fast. Power answers how much it costs to be
that fast, and in a serving fleet the second question is the one that sets the
bill. This module samples the gpu on a background thread while a benchmark runs
and reports mean power, peak power, total energy in joules, and the derived
inferences per joule, which is the efficiency number that actually compares two
precisions or two runtimes on equal terms.

Sampling goes through NVML. Energy is the trapezoid integral of the sampled
power curve over the wall clock of the measured region, which is the honest way
to turn a series of instantaneous watt readings into joules. When NVML is not
importable, when there is no driver, or when the device does not expose a power
sensor, every field reads not available and the reason is carried in the record
rather than being silently dropped.

Usage follows the standard start and stop shape.

    sampler = PowerSampler()
    sampler.start()
    ... run the workload ...
    sampler.stop()
    record = sampler.summarize(n_inferences=total_rows_scored)
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

from src.inference.common import NOT_AVAILABLE


class PowerSampler:
    """Sample gpu power and utilization on a background thread.

    The sampler is safe to construct and to start on a machine with no gpu. In
    that case it records nothing, the summary says not available, and the reason
    names what was missing. Nothing here raises for an absent device.

    Parameters
    ----------
    interval_s : float
        Seconds between samples. The default of 20 milliseconds is fast enough
        to catch the shape of a short burst without perturbing the workload.
    device_index : int
        Which gpu to sample.
    """

    def __init__(self, interval_s: float = 0.02, device_index: int = 0) -> None:
        self.interval_s = float(interval_s)
        self.device_index = int(device_index)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pynvml = None
        self._handle = None
        self._reason = "the power sampler was never started"
        self._times: List[float] = []
        self._watts: List[float] = []
        self._sm_util: List[float] = []
        self._mem_util: List[float] = []
        self._mem_used: List[int] = []
        self._t0: Optional[float] = None
        self._t1: Optional[float] = None

    @property
    def available(self) -> bool:
        """True when at least one power sample was actually collected."""
        return len(self._watts) > 0

    def _open(self) -> bool:
        """Initialize NVML and grab a device handle, recording why if it fails."""
        try:
            import pynvml
        except Exception as exc:  # noqa: BLE001
            self._reason = (
                "pynvml is not installed, so gpu power could not be sampled "
                f"({exc}). Install nvidia-ml-py on a cuda host to fill this in."
            )
            return False
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception as exc:  # noqa: BLE001
            self._reason = (
                f"NVML could not open gpu {self.device_index} ({exc}), so there "
                "is no NVIDIA driver visible to this process"
            )
            return False
        try:
            pynvml.nvmlDeviceGetPowerUsage(handle)
        except Exception as exc:  # noqa: BLE001
            self._reason = (
                f"gpu {self.device_index} does not expose a power sensor to NVML "
                f"({exc})"
            )
            try:
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
            return False
        self._pynvml = pynvml
        self._handle = handle
        self._reason = ""
        return True

    def _close(self) -> None:
        """Shut NVML down, ignoring any error on the way out."""
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
        self._pynvml = None
        self._handle = None

    def _loop(self) -> None:
        """Background sampling loop. Runs until the stop event is set."""
        pynvml = self._pynvml
        handle = self._handle
        while not self._stop_event.is_set():
            now = time.perf_counter()
            try:
                milliwatts = pynvml.nvmlDeviceGetPowerUsage(handle)
                self._times.append(now)
                self._watts.append(float(milliwatts) / 1000.0)
            except Exception:  # noqa: BLE001 a transient NVML error must not kill the run
                pass
            try:
                rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self._sm_util.append(float(rates.gpu))
                self._mem_util.append(float(rates.memory))
            except Exception:  # noqa: BLE001
                pass
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self._mem_used.append(int(mem.used))
            except Exception:  # noqa: BLE001
                pass
            self._stop_event.wait(self.interval_s)

    def start(self) -> "PowerSampler":
        """Begin sampling. Returns self so the call can be chained."""
        self._times = []
        self._watts = []
        self._sm_util = []
        self._mem_util = []
        self._mem_used = []
        self._stop_event.clear()
        self._t0 = time.perf_counter()
        self._t1 = None
        if not self._open():
            return self
        self._thread = threading.Thread(
            target=self._loop, name="clickrank-power-sampler", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> "PowerSampler":
        """Stop sampling and join the background thread."""
        self._t1 = time.perf_counter()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._close()
        return self

    def __enter__(self) -> "PowerSampler":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def summarize(self, n_inferences: Optional[int] = None) -> Dict[str, Any]:
        """Return the power record for the sampled region.

        Energy is the trapezoid integral of the sampled watt curve. When fewer
        than two samples landed, the record falls back to mean power times the
        wall clock duration and says so in the method field, because a single
        sample cannot describe a curve.

        Parameters
        ----------
        n_inferences : int, optional
            Number of rows scored inside the sampled region. When given, the
            record carries inferences per joule, which is the efficiency figure
            that lets two precisions be compared without reference to wall clock.
        """
        duration = None
        if self._t0 is not None and self._t1 is not None:
            duration = float(self._t1 - self._t0)

        if not self.available:
            return {
                "available": False,
                "reason": self._reason or "no power samples were collected",
                "samples": 0,
                "duration_s": duration,
                "mean_watts": None,
                "peak_watts": None,
                "energy_joules": None,
                "energy_method": NOT_AVAILABLE,
                "mean_sm_utilization_pct": None,
                "mean_memory_utilization_pct": None,
                "peak_memory_used_bytes": None,
                "inferences_per_joule": None,
            }

        watts = list(self._watts)
        times = list(self._times)
        mean_watts = sum(watts) / len(watts)
        peak_watts = max(watts)

        if len(watts) >= 2:
            energy = 0.0
            for i in range(1, len(watts)):
                dt = times[i] - times[i - 1]
                energy += 0.5 * (watts[i] + watts[i - 1]) * dt
            method = "trapezoid integral of the sampled power curve"
        elif duration is not None:
            energy = mean_watts * duration
            method = "mean power times wall clock, only one sample landed"
        else:
            energy = None
            method = NOT_AVAILABLE

        per_joule = None
        if energy and n_inferences:
            per_joule = float(n_inferences) / float(energy)

        return {
            "available": True,
            "reason": "",
            "samples": len(watts),
            "duration_s": duration,
            "mean_watts": mean_watts,
            "peak_watts": peak_watts,
            "energy_joules": energy,
            "energy_method": method,
            "mean_sm_utilization_pct": (
                sum(self._sm_util) / len(self._sm_util) if self._sm_util else None
            ),
            "mean_memory_utilization_pct": (
                sum(self._mem_util) / len(self._mem_util) if self._mem_util else None
            ),
            "peak_memory_used_bytes": max(self._mem_used) if self._mem_used else None,
            "inferences_per_joule": per_joule,
        }


def unavailable_power_record(reason: str) -> Dict[str, Any]:
    """Return a power record for a lane that never ran, with the reason stated."""
    return {
        "available": False,
        "reason": reason,
        "samples": 0,
        "duration_s": None,
        "mean_watts": None,
        "peak_watts": None,
        "energy_joules": None,
        "energy_method": NOT_AVAILABLE,
        "mean_sm_utilization_pct": None,
        "mean_memory_utilization_pct": None,
        "peak_memory_used_bytes": None,
        "inferences_per_joule": None,
    }
