#!/usr/bin/env python
"""Start the ClickRank ranking service.

This is the online lane. It loads one serving bundle, selects the fastest
backend the machine can actually construct, warms it, and serves /score,
/health, and /metrics over http.

The startup sequence is deliberately loud. It prints the hardware record, the
bundle provenance including the sha256 of the weights and of the fitted feature
pipeline, every backend it tried with the reason each unavailable one was
skipped, and the backend it settled on. An operator should never have to guess
which runtime is answering requests, because a service that quietly fell back
from a compiled engine to eager PyTorch looks identical from the outside until
someone reads the latency graph.

Run from the repository root.

    python scripts/serve.py
    python scripts/serve.py --build-only
    python scripts/serve.py --backend openvino-cpu-fp32 --threads 16
    python scripts/serve.py --port 8080 --workers 4

A single worker is the default because the measurements in docs/SERVING.md were
taken against a single worker and mixing worker counts into one latency claim
would make it meaningless. Real horizontal scaling is the --workers flag and it
is the right lever, since scoring is cpu bound and one Python process cannot use
more than a core's worth of interpreter at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Insert the repository root onto sys.path so that "import src" works when this
# script is run directly from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.serving.artifact import (  # noqa: E402  import after the sys.path insert
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BUNDLE_DIR,
    build_bundle,
)
from src.serving.runtime import (  # noqa: E402
    PREFERENCE_ORDER,
    REDUCED_PRECISION_ORDER,
)
from src.serving.service import ServiceConfig, build_service  # noqa: E402

# The module path uvicorn imports when it is asked to run several workers. A
# worker process cannot inherit an already built app object, so it rebuilds one
# from this factory using the environment variables set below.
_APP_FACTORY = "src.serving.entrypoint:app"


def parse_args() -> argparse.Namespace:
    """Parse the command line flags for the service."""
    parser = argparse.ArgumentParser(
        description="Serve the ClickRank ranker over http."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument(
        "--bundle-dir",
        default=DEFAULT_BUNDLE_DIR,
        help="Directory holding the serving bundle. Built when it is not there.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory holding the checkpoint and the exported graph.",
    )
    parser.add_argument(
        "--model",
        default="deepfm",
        help="Which trained ranker to serve. One of deepfm or dcn.",
    )
    parser.add_argument(
        "--data-path",
        default="data/criteo.csv",
        help="Criteo file used when a bundle has to be built. Falls back to synthetic.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100000,
        help="Rows used when a bundle has to be built.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic data when a bundle has to be built.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help=(
            "Pin one backend key instead of probing the preference order. The "
            f"fp32 order is {', '.join(PREFERENCE_ORDER)}."
        ),
    )
    parser.add_argument(
        "--allow-reduced-precision",
        action="store_true",
        help=(
            "Let the probe consider the fp16 and int8 backends "
            f"({', '.join(REDUCED_PRECISION_ORDER)}) ahead of the fp32 order. "
            "These change the model output, so this is off by default."
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=4096,
        help="Largest candidate set a single request may carry.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help=(
            "Size of the request thread pool. This is the queueing parameter of "
            "the service and it is reported on /health."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Uvicorn worker processes. Scoring is cpu bound, so this is the "
            "lever that actually scales throughput on a multi core host."
        ),
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the discarded warmup batches. Only useful for a fast smoke start.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build or rebuild the serving bundle, print the manifest, and exit.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the bundle even when one is already present.",
    )
    parser.add_argument(
        "--log-level", default="info", help="Uvicorn log level."
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ServiceConfig:
    """Translate the parsed flags into a service configuration."""
    return ServiceConfig(
        bundle_dir=args.bundle_dir,
        artifact_dir=args.artifact_dir,
        model_name=args.model,
        data_path=args.data_path,
        sample_size=args.sample_size,
        synthetic=args.synthetic,
        backend=args.backend,
        allow_reduced_precision=args.allow_reduced_precision,
        max_candidates=args.max_candidates,
        thread_pool_size=args.threads,
        warmup=not args.no_warmup,
        verbose=True,
    )


def export_environment(args: argparse.Namespace) -> None:
    """Publish the configuration for uvicorn worker processes to read.

    Multiple workers means multiple processes, and a process cannot inherit a
    Python object across a fork that uvicorn manages itself. The worker
    entrypoint rebuilds the service from these variables, which keeps one source
    of truth for the configuration rather than two argument parsers.
    """
    os.environ["ADRANK_BUNDLE_DIR"] = str(args.bundle_dir)
    os.environ["ADRANK_ARTIFACT_DIR"] = str(args.artifact_dir)
    os.environ["ADRANK_MODEL"] = str(args.model)
    os.environ["ADRANK_DATA_PATH"] = str(args.data_path)
    os.environ["ADRANK_SAMPLE_SIZE"] = str(args.sample_size)
    os.environ["ADRANK_SYNTHETIC"] = "1" if args.synthetic else "0"
    os.environ["ADRANK_BACKEND"] = str(args.backend or "")
    os.environ["ADRANK_ALLOW_REDUCED_PRECISION"] = (
        "1" if args.allow_reduced_precision else "0"
    )
    os.environ["ADRANK_MAX_CANDIDATES"] = str(args.max_candidates)
    os.environ["ADRANK_THREADS"] = str(args.threads)
    os.environ["ADRANK_WARMUP"] = "0" if args.no_warmup else "1"


def main() -> None:
    """Build the service and hand it to uvicorn."""
    args = parse_args()

    if args.rebuild or args.build_only:
        bundle = build_bundle(
            bundle_dir=args.bundle_dir,
            artifact_dir=args.artifact_dir,
            model_name=args.model,
            data_path=args.data_path,
            sample_size=args.sample_size,
            synthetic=args.synthetic,
            force_export=args.rebuild,
            verbose=True,
        )
        if args.build_only:
            print(json.dumps(bundle.describe(), indent=2, sort_keys=True))
            return

    import uvicorn

    if args.workers and args.workers > 1:
        export_environment(args)
        print(
            f"starting {args.workers} uvicorn workers on {args.host}:{args.port}. "
            "Each worker loads the bundle and probes backends on its own, so the "
            "startup banner appears once per worker."
        )
        uvicorn.run(
            _APP_FACTORY,
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level,
        )
        return

    service = build_service(config_from_args(args))
    print(
        f"serving {service.bundle.describe()['model_name']} on "
        f"http://{args.host}:{args.port} with the {service.engine.backend_key} "
        f"backend and a thread pool of {args.threads}."
    )
    uvicorn.run(
        service.app, host=args.host, port=args.port, log_level=args.log_level
    )


if __name__ == "__main__":
    main()
