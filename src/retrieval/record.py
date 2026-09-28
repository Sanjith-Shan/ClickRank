"""Every measurement goes to results/*.jsonl with its machine, load and data.

A row without the machine it ran on and how busy that machine was is not a
result. Each row carries the dataset label from RetrievalData.label(), which
says SYNTHETIC when it is, so a synthetic figure can never be mistaken for a
real one when the file is read later.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict

from src.inference.hardware import cpu_model


def _mem_gb() -> float:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2)
        return round(int(out.stdout.strip()) / 2**30, 1)
    except Exception:
        try:
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
        except Exception:
            return float("nan")


def machine() -> Dict[str, Any]:
    import numpy as np
    import torch

    rec = {
        "cpu": cpu_model(),
        "os": f"{platform.system()} {platform.release()}",
        "logical_cores": os.cpu_count(),
        "ram_gb": _mem_gb(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
    }
    try:
        import faiss

        rec["faiss"] = faiss.__version__
        rec["faiss_threads"] = faiss.omp_get_max_threads()
    except ImportError:
        rec["faiss"] = "not installed"
    return rec


def load() -> Dict[str, Any]:
    """The 1, 5 and 15 minute load averages at the moment of writing."""
    try:
        a, b, c = os.getloadavg()
        return {"load_1m": round(a, 2), "load_5m": round(b, 2), "load_15m": round(c, 2)}
    except OSError:
        return {}


def write(path: str, row: Dict[str, Any], *, dataset: str) -> Dict[str, Any]:
    """Append one row to a jsonl file, stamped with time, machine, load and data."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    full = {
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset,
        **row,
        "machine": machine(),
        "load": load(),
    }
    with open(path, "a") as fh:
        fh.write(json.dumps(full, default=_default) + "\n")
    return full


def _default(o):
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def results_dir(root: str, synthetic: bool) -> str:
    """Synthetic runs write under results/synthetic so the two never mix."""
    return os.path.join(root, "synthetic", "retrieval") if synthetic else os.path.join(root, "retrieval")
