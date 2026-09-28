#!/usr/bin/env python
"""Closed loop load test of the two stage recommend service.

N virtual clients each send a request, wait for the answer, and send the next,
for D seconds after a warmup that is thrown away. Requests replay real test
day (user, placement) pairs sampled from the log, so the mix of known users,
heavy users and cold starts is the log's own. The report is achieved requests
per second, end to end p50, p95 and p99 as the client saw them, and the server
side stage breakdown from each response's timings.

Closed loop means the offered load follows the service's own speed, and a slow
response suppresses the requests that would have been sent during it, so the
tail here is a floor on the real tail (coordinated omission), the same caveat
scripts/run_load_test.py states. The clients run on the same machine as the
server and compete with it for cores. Both facts are recorded with the result.

    python scripts/load_test_two_stage.py --source synthetic --spawn --clients 4 --duration 10
    python scripts/load_test_two_stage.py --url http://127.0.0.1:PORT --source taobao

Writes one row to results/retrieval/serving_load.jsonl, or to
results/synthetic/retrieval/serving_load.jsonl for synthetic data, with the
dataset label, the machine and the load average.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.retrieval import record  # noqa: E402
from src.serving.two_stage import STAGES  # noqa: E402


def request_pool(source: str, data_dir: str, size: int, k: int, n: int, seed: int) -> tuple:
    """Sample (user, placement) pairs from the test day of the parquet cache."""
    cache = os.path.join(data_dir, "cache")
    imp = pd.read_parquet(os.path.join(cache, "impressions.parquet"), columns=["user", "day", "pid"])
    notes = json.load(open(os.path.join(cache, "notes.json")))
    pids = notes.get("pids", [])
    test_day = int(imp["day"].max())
    t = imp[imp["day"] == test_day]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(t), size=min(size, len(t)), replace=False)
    users = t["user"].to_numpy()[pick]
    codes = t["pid"].to_numpy()[pick]
    pool = [{"user_id": int(u), "pid": pids[int(c)] if int(c) < len(pids) else int(c), "k": k, "n": n}
            for u, c in zip(users, codes)]
    return pool, source == "synthetic"


async def client_loop(client, url, pool, deadline, offset, stride, out) -> None:
    i = offset
    while time.perf_counter() < deadline:
        body = pool[i % len(pool)]
        i += stride
        t0 = time.perf_counter()
        try:
            r = await client.post(url, json=body)
            ms = (time.perf_counter() - t0) * 1000.0
            if r.status_code != 200:
                if out is not None:
                    out["errors"] += 1
                continue
            if out is not None:
                js = r.json()
                out["lat_ms"].append(ms)
                out["cold"] += int(js.get("cold_start", False))
                for s in STAGES:
                    out["stages"].setdefault(s, []).append(js["timings_us"].get(s, float("nan")))
        except Exception:  # noqa: BLE001 a client side failure is a data point
            if out is not None:
                out["errors"] += 1


async def run(url: str, pool, clients: int, duration: float, warmup: float) -> Dict[str, Any]:
    import httpx

    target = url.rstrip("/") + "/v1/recommend"
    limits = httpx.Limits(max_connections=clients + 4, max_keepalive_connections=clients + 4)
    out = {"lat_ms": [], "stages": {}, "errors": 0, "cold": 0}
    async with httpx.AsyncClient(timeout=30.0, limits=limits) as client:
        if warmup > 0:
            dl = time.perf_counter() + warmup
            await asyncio.gather(*[client_loop(client, target, pool, dl, i, clients, None) for i in range(clients)])
        load_before = record.load()
        t0 = time.perf_counter()
        dl = t0 + duration
        await asyncio.gather(*[client_loop(client, target, pool, dl, i, clients, out) for i in range(clients)])
        wall = time.perf_counter() - t0
    out["wall_s"] = wall
    out["load_before"] = load_before
    return out


def pct(a: List[float]) -> Dict[str, float]:
    x = np.asarray(a, dtype=np.float64)
    if len(x) == 0:
        return {}
    return {"p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "mean": float(x.mean())}


def spawn(args, port_file: str) -> subprocess.Popen:
    cmd = [sys.executable, os.path.join(_REPO_ROOT, "scripts", "serve_two_stage.py"),
           "--source", args.source, "--results", args.results, "--index", args.index,
           "--ranker", args.ranker, "--port", "0", "--port-file", port_file]
    if args.data:
        cmd += ["--data", args.data]
    if args.pool:
        cmd += ["--pool", str(args.pool)]
    print("starting", " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, cwd=_REPO_ROOT)


def wait_ready(url: str, proc, seconds: float = 900.0) -> Dict[str, Any]:
    import httpx

    deadline = time.time() + seconds
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"the service exited with code {proc.returncode}")
        try:
            r = httpx.get(url + "/healthz", timeout=2.0)
            if r.status_code == 200:
                return r.json()
        except Exception:  # noqa: BLE001 not up yet
            pass
        time.sleep(0.5)
    raise TimeoutError("the service did not come up")


def main() -> None:
    ap = argparse.ArgumentParser(description="Closed loop load test of /v1/recommend.")
    ap.add_argument("--source", choices=["taobao", "avazu", "synthetic"], default="synthetic")
    ap.add_argument("--data", default=None)
    ap.add_argument("--results", default="results")
    ap.add_argument("--url", default=None, help="a running service; omit with --spawn")
    ap.add_argument("--spawn", action="store_true", help="start the service on a free port for this run")
    ap.add_argument("--index", default="flat")
    ap.add_argument("--ranker", default="deepfm")
    ap.add_argument("--pool", type=int, default=None, help="server request thread pool size")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=3.0)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--requests", type=int, default=5000, help="size of the replay pool")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.url and not args.spawn:
        ap.error("give --url or --spawn")

    data_dir = args.data or {"taobao": "data/taobao", "synthetic": "data/synthetic_ids"}[args.source]
    pool, synthetic = request_pool(args.source, data_dir, args.requests, args.k, args.n, args.seed)

    proc = None
    port_file = None
    try:
        if args.spawn:
            fd, port_file = tempfile.mkstemp(prefix="clickrank_port_")
            os.close(fd)
            os.remove(port_file)
            proc = spawn(args, port_file)
            while not os.path.exists(port_file) or not open(port_file).read().strip():
                if proc.poll() is not None:
                    raise RuntimeError(f"the service exited with code {proc.returncode}")
                time.sleep(0.2)
            url = f"http://127.0.0.1:{int(open(port_file).read().strip())}"
        else:
            url = args.url.rstrip("/")
        health = wait_ready(url, proc)
        res = asyncio.run(run(url, pool, args.clients, args.duration, args.warmup))
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if port_file and os.path.exists(port_file):
            os.remove(port_file)

    done = len(res["lat_ms"])
    row = {
        "mode": "closed_loop",
        "clients": args.clients,
        "duration_s": round(res["wall_s"], 3),
        "k": args.k,
        "n": args.n,
        "requests_ok": done,
        "errors": res["errors"],
        "cold_start_requests": res["cold"],
        "achieved_rps": done / res["wall_s"] if res["wall_s"] else float("nan"),
        "end_to_end_ms": pct(res["lat_ms"]),
        "server_stage_us": {s: pct(v) for s, v in res["stages"].items()},
        "service": {k: health.get(k) for k in ("index", "index_bytes", "ranker", "corpus_ads",
                                               "torch_threads", "faiss_threads")},
        "load_before_run": res["load_before"],
        "caveats": "closed loop, tail is a floor; clients share the machine with the server",
    }
    label = health.get("dataset", "unknown")
    if synthetic and "SYNTHETIC" not in label:
        label = f"SYNTHETIC ({label})"
    path = os.path.join(record.results_dir(args.results, synthetic), "serving_load.jsonl")
    record.write(path, row, dataset=label)
    print(json.dumps({"dataset": label, **row}, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
