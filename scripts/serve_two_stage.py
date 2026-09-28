#!/usr/bin/env python
"""Start the two stage recommend service (retrieve with the towers, rank with a ranker).

Needs the artifacts that scripts/run_retrieval.py and scripts/run_rankers.py
write. The startup prints what it loaded: the dataset label (SYNTHETIC when it
is), the index and its size, the ranker, and the thread settings, and with
--parity it refuses to report ready until served scores match the offline
ranker on logged impressions.

    python scripts/serve_two_stage.py --source synthetic --port 0
    python scripts/serve_two_stage.py --source taobao --index "hnsw M=32 efConstruction=200 efSearch=128" --parity 20

--port 0 picks a free port and prints it. A single worker is the default, for
the reason docs/SERVING.md gives: one latency claim, one worker count.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.serving.two_stage import TwoStageConfig, create_app, load_bundle, parity_check  # noqa: E402


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the two stage recommend endpoint.")
    ap.add_argument("--source", choices=["taobao", "avazu", "synthetic"], default="synthetic")
    ap.add_argument("--data", default=None)
    ap.add_argument("--results", default="results")
    ap.add_argument("--index", default="flat", help='"flat" or e.g. "hnsw M=32 efConstruction=200 efSearch=128"')
    ap.add_argument("--ranker", default="deepfm", choices=["deepfm", "dcn", "din", "din-mean"])
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--faiss-threads", type=int, default=1)
    ap.add_argument("--pool", type=int, default=None, help="request thread pool size")
    ap.add_argument("--parity", type=int, default=0, help="impressions for the startup parity check")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0 picks a free port")
    ap.add_argument("--port-file", default=None, help="write the bound port here once chosen")
    ap.add_argument("--log-level", default="warning")
    args = ap.parse_args()

    cfg = TwoStageConfig(source=args.source, data_dir=args.data, results_dir=args.results,
                         index=args.index, ranker=args.ranker, torch_threads=args.torch_threads,
                         faiss_threads=args.faiss_threads, parity_samples=args.parity)
    bundle = load_bundle(cfg)
    print("loaded", json.dumps(bundle.info))
    parity = None
    if args.parity:
        parity = parity_check(bundle, samples=args.parity)
        print("parity", json.dumps(parity))
        if not parity["passed"]:
            print("parity failed, the service will report not ready on /readyz")
    app = create_app(bundle, parity=parity, thread_pool_size=args.pool)

    port = args.port or free_port(args.host)
    if args.port_file:
        with open(args.port_file, "w") as fh:
            fh.write(str(port))
    print(f"serving on http://{args.host}:{port}", flush=True)

    import uvicorn

    uvicorn.run(app, host=args.host, port=port, log_level=args.log_level)


if __name__ == "__main__":
    main()
