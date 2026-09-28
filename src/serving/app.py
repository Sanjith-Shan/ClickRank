"""The FastAPI scoring service.

Three endpoints and no more. /score ranks one auction, /health says what is
loaded and what it is running on, and /metrics exposes the counters and the
latency histograms. Anything else a serving stack needs, such as routing,
retries, or authentication, belongs in the layer in front of this one rather
than inside a ranker.

The scoring endpoint is a plain def rather than an async def, which is a
deliberate choice and not an oversight. Scoring is cpu bound. Featurizing a
candidate set hashes several dozen strings per row in Python, and running the
model is a numpy and runtime call. Putting that work directly on the event loop
would block every other request behind it for the duration and turn the service
into a strictly serial queue with an async facade. Declaring the endpoint
synchronous hands it to the Starlette thread pool, so requests genuinely queue
and genuinely overlap wherever the underlying work releases the global
interpreter lock, which is what makes the concurrency sweep in
scripts/run_load_test.py measure something real.

The thread pool size is set explicitly at startup rather than left at the
framework default. It is the queueing parameter of this service. Too few threads
and the tail latency is dominated by waiting for a worker, too many and the
threads fight over the interpreter lock and the tail latency is dominated by
context switching instead. Either way it is a number an operator has to be able
to see and set, so it is a flag and it is reported on /health.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from src.serving.metrics import ServiceMetrics
from src.serving.runtime import ScoringEngine
from src.serving.schemas import (
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
    ScoredAd,
    ScoreTimings,
)

API_TITLE: str = "ClickRank ranking service"
API_VERSION: str = "1.0.0"

API_DESCRIPTION: str = (
    "Scores a candidate set of ads for one auction and returns it ranked by "
    "calibrated click probability. The model, the fitted feature pipeline, and "
    "the calibration all come from one serving bundle, which is the same bundle "
    "the batch scoring job loads."
)


def create_app(
    engine: ScoringEngine,
    metrics: Optional[ServiceMetrics] = None,
    hardware: Optional[Dict[str, Any]] = None,
    thread_pool_size: Optional[int] = None,
) -> FastAPI:
    """Build the FastAPI app around an already constructed scoring engine.

    The engine is constructed before the app rather than inside a startup hook
    so that a bundle that will not load or a backend that will not build fails
    the process rather than producing a server that answers /health with an
    error. A ranking service that is up and cannot rank is worse than one that
    is down, because the load balancer keeps sending it traffic.
    """
    metrics = metrics if metrics is not None else ServiceMetrics()
    hardware = hardware if hardware is not None else {}
    started_at = time.time()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Set the request thread pool size, which is the queueing parameter."""
        if thread_pool_size:
            try:
                import anyio.to_thread

                anyio.to_thread.current_default_thread_limiter().total_tokens = int(
                    thread_pool_size
                )
            except Exception as exc:  # noqa: BLE001 a default pool still serves
                print(
                    f"could not set the thread pool size to {thread_pool_size} "
                    f"({exc}), so the framework default is in use."
                )
        yield

    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.engine = engine
    app.state.metrics = metrics
    app.state.hardware = hardware
    app.state.thread_pool_size = thread_pool_size

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Report the loaded backend, the hardware, and the bundle provenance.

        This is the endpoint that makes the honesty rule enforceable. It names
        the backend that was actually selected, every backend that was tried and
        the reason it was skipped, the sha256 of the weights and of the feature
        pipeline, and the machine underneath. An operator who believes the
        service is running a compiled engine can check rather than assume.
        """
        snapshot = metrics.snapshot()
        described = engine.describe()
        described["backend"]["thread_pool_size"] = thread_pool_size
        return HealthResponse(
            status="ok",
            backend=described["backend"],
            hardware=hardware,
            bundle=described["bundle"],
            uptime_seconds=round(time.time() - started_at, 3),
            requests_total=int(snapshot["requests_total"]),
        )

    @app.get("/metrics")
    def metrics_endpoint(
        format: str = Query(
            default="prometheus",
            pattern="^(prometheus|json)$",
            description="prometheus for the text exposition format, json for the raw view.",
        )
    ):
        """Expose the request counters and the three latency histograms."""
        if format == "json":
            return JSONResponse(content=_jsonable(metrics.snapshot()))
        return PlainTextResponse(
            content=metrics.render_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.post("/score", response_model=ScoreResponse)
    def score(request: ScoreRequest) -> ScoreResponse:
        """Score and rank one auction's candidate set.

        The whole candidate set goes through the model as a single batch. That
        is what an auction asks for and it is also why the batch dimension is
        meaningful here rather than arbitrary. Ranking happens server side
        because the service already holds the probabilities and the caller would
        otherwise sort them again, and because returning a ranked list makes the
        top_k truncation possible without sending the losers over the wire.
        """
        try:
            probs, timing = engine.score_rows(request.rows())
        except ValueError as exc:
            metrics.observe_failure()
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 a failed request is a 500 and a metric
            metrics.observe_failure()
            raise HTTPException(
                status_code=500, detail=f"scoring failed. {type(exc).__name__} {exc}"
            ) from exc

        metrics.observe_request(
            total_seconds=timing.total_seconds,
            feature_seconds=timing.feature_seconds,
            model_seconds=timing.model_seconds,
            n_candidates=len(request.candidates),
        )

        # Descending probability, with the request order breaking ties so the
        # ranking is deterministic for a caller that sends the same auction
        # twice. numpy argsort is stable on the default kind only for stable
        # kinds, so the stable kind is asked for explicitly.
        values = np.asarray(probs, dtype=np.float64)
        order = np.argsort(-values, kind="stable")
        limit = request.top_k or len(order)
        ranked: List[ScoredAd] = [
            ScoredAd(
                ad_id=request.candidates[int(idx)].ad_id,
                rank=position + 1,
                p_click=float(values[int(idx)]),
            )
            for position, idx in enumerate(order[:limit])
        ]

        return ScoreResponse(
            request_id=request.request_id,
            backend=engine.backend_key,
            model_name=str(engine.bundle.manifest.get("model", {}).get("display_name", "")),
            n_candidates=len(request.candidates),
            ranked=ranked,
            timings=ScoreTimings(
                feature_ms=round(timing.feature_seconds * 1000.0, 4),
                model_ms=round(timing.model_seconds * 1000.0, 4),
                total_ms=round(timing.total_seconds * 1000.0, 4),
            ),
        )

    @app.get("/")
    def root(request: Request) -> Dict[str, Any]:
        """Name the service and point at the three endpoints that matter."""
        return {
            "service": API_TITLE,
            "version": API_VERSION,
            "endpoints": ["/score", "/health", "/metrics", "/docs"],
            "backend": engine.backend_key,
        }

    return app


def _jsonable(value: Any) -> Any:
    """Replace the non finite floats a fresh histogram produces with null.

    A histogram with no observations reports a mean and quantiles of NaN, which
    is the honest value and is not valid json. It goes out as null rather than
    as a zero, for the same reason the reports in this project print a not
    available marker instead of an empty cell.
    """
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return None if (np.isnan(value) or np.isinf(value)) else value
    return value
