"""The serving layer for the ClickRank rankers.

The inference package answers how fast a trained ranker can run. This package
answers whether it can be served, which is a different question with a different
answer. A single process timing loop measures a model. A service measures a
system, and a system has a feature pipeline in front of the model, a queue in
front of that, and a tail latency that only appears once several requests are in
flight at once.

- features. Persists the fitted feature pipeline and replays it at request time.
  This is the module that prevents training and serving skew, and it is the
  reason the rest of the package exists in this order.
- calibration. The served probability calibrator, fitted on validation and
  applied only when it measurably improves calibration.
- artifact. The serving bundle, which is the manifest that binds the weights,
  the exported graph, the fitted pipeline, and the calibration into one thing
  with fingerprints on every member.
- runtime. Backend selection through the src.inference registry in a documented
  preference order, and the scoring engine both lanes run on.
- metrics. Request counters and latency histograms, split into the feature half
  and the model half.
- schemas. The typed request and response contracts, strict about unknown
  fields.
- app. The FastAPI application with /score, /health, and /metrics.
- service. The single function that assembles all of the above.
- batch. The batch scoring lane, which loads the same bundle and calls the same
  engine as the online lane.

Everything here runs on a cpu only machine with no accelerator, which is the
machine it was developed and measured on, and every backend that could not be
constructed is reported with the reason rather than silently skipped.
"""

from __future__ import annotations

from src.serving.artifact import ServingBundle, build_bundle, load_bundle
from src.serving.metrics import ServiceMetrics
from src.serving.runtime import PREFERENCE_ORDER, ScoringEngine, select_backend
from src.serving.service import Service, ServiceConfig, build_service

__all__ = [
    "PREFERENCE_ORDER",
    "Service",
    "ServiceConfig",
    "ServiceMetrics",
    "ScoringEngine",
    "ServingBundle",
    "build_bundle",
    "build_service",
    "load_bundle",
    "select_backend",
]
