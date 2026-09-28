"""The two stage recommend service: retrieve with the towers, rank with a ranker.

One endpoint does the work. POST /v1/recommend takes a user, a placement and an
hour, runs the user tower, searches the FAISS index for the top K ads, scores
those K with the ranker, and returns the best n with their click probability
and the time each stage took. The request path is TwoStagePipeline from
src/retrieval/pipeline.py, the same class the offline measurement in
scripts/run_two_stage.py times, so the served numbers and the measured numbers
come from one code path rather than two that are hoped to agree.

The service holds a snapshot. The encoder, the click history and the corpus
embeddings are built from the log once at startup, and every request sees the
user's state as of the end of the training days (enc.cutoff_ts), which is the
state the offline evaluation freezes too. A production system would stream new
clicks into the history and re-embed changed ads. This one does not, and says so.

Cold start. A user id the log has never seen is not an error. It is served
from an extra user row appended at load time whose every field is the out of
vocabulary code and whose click history is empty, so the towers place the user
from nothing and the ranker scores with unknown profile fields. That is the
same thing the encoder does for a user first seen on the test day, and the
response says it happened. A placement the log has never seen gets the out of
vocabulary placement code in the same way.

Parity. The repo's serving layer refuses to start when the features it builds
on the request path differ from the ones training saw. The same idea is here as
parity_check: for a sample of logged test day impressions whose ad the pipeline
retrieves, the served click probability must equal the offline ranker score on
the frozen rows_dataset row for that impression. The check runs at startup when
asked and /readyz stays false until it passes.

Endpoints are plain def, not async def, for the reason src/serving/app.py
gives: the work is cpu bound, and a synchronous endpoint runs on the thread
pool instead of blocking the event loop. Torch and FAISS are each set to one
thread by default, which is one request per worker thread.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from src.serving.metrics import Histogram

# Guard on the retrieval depth a request may ask for. This bounds the ranking
# work one request can demand, the same way MAX_CANDIDATES bounds /score.
MAX_K: int = 1000
MAX_N: int = 100
BEIJING_OFFSET_S: int = 8 * 3600
STAGES = ("user_embed", "search", "feature_build", "rank", "total")


# ---------------------------------------------------------------------------
# The artifact bundle


@dataclass
class TwoStageConfig:
    source: str = "synthetic"
    data_dir: Optional[str] = None
    results_dir: str = "results"
    index: str = "flat"              # "flat", or e.g. "hnsw M=32 efConstruction=200 efSearch=128"
    ranker: str = "deepfm"           # deepfm, dcn, din or din-mean
    torch_threads: int = 1
    faiss_threads: int = 1
    parity_samples: int = 0          # 0 skips the startup parity check


@dataclass
class TwoStageBundle:
    """Everything a request needs, loaded once."""

    config: TwoStageConfig
    label: str
    enc: Any
    hist: Any
    spec: Any
    tower: Any
    tower_cfg: Dict[str, Any]
    index: Any
    ranker: Any
    pipeline: Any
    user_lookup: pd.Index
    pid_lookup: Dict[str, int]
    cold_user_row: int
    load_seconds: float
    info: Dict[str, Any] = field(default_factory=dict)

    def user_row(self, user_id: Union[int, str]) -> tuple:
        """(row, cold) for a raw user id. Unknown ids get the cold start row."""
        try:
            key = int(user_id)
        except (TypeError, ValueError):
            return self.cold_user_row, True
        pos = int(self.user_lookup.get_indexer([key])[0])
        if pos < 0:
            return self.cold_user_row, True
        return pos, False

    def pid_code(self, pid: Union[int, str]) -> tuple:
        """(code, known) for a raw placement string or an integer code."""
        code = self.pid_lookup.get(str(pid))
        if code is not None:
            return code, True
        try:
            code = int(pid)
        except (TypeError, ValueError):
            return -1, False
        if 0 <= code < self.enc.n_pid:
            return code, True
        return -1, False


def _default_data_dir(source: str) -> str:
    return {"taobao": "data/taobao", "synthetic": "data/synthetic_ids"}[source]


def _parse_index(spec: str) -> tuple:
    parts = spec.split()
    kind = parts[0]
    kv = {k: int(v) for k, v in (p.split("=") for p in parts[1:])}
    search = {k: kv.pop(k) for k in ("nprobe", "efSearch") if k in kv}
    return kind, kv, search


def load_index(spec: str, corpus: np.ndarray, art_dir: str):
    """Build the serving index, or load it when the sweep already saved it.

    run_index_sweep.py writes each approximate index it built under artifacts/
    as kind_param<value>.faiss. Reusing that file avoids a minutes long HNSW
    build at every service start on the real corpus.
    """
    from src.retrieval import index as ix

    kind, build, search = _parse_index(spec)
    saved = os.path.join(art_dir, f"{kind}_{'_'.join(f'{k}{v}' for k, v in build.items())}.faiss")
    if kind != "flat" and os.path.exists(saved):
        t0 = time.perf_counter()
        raw = ix.faiss.read_index(saved)
        idx = ix.Index(kind, raw, build, time.perf_counter() - t0, search)
        idx.params["loaded_from"] = os.path.basename(saved)
        return idx
    idx = ix.build_index(kind, corpus, **build)
    for k, v in search.items():
        idx.set_search_param(k, v)
    return idx


def _load_ranker(art: str, name: str, spec, enc):
    from src.models.dcn import DCNModule
    from src.models.deepfm import DeepFMModule
    from src.models.din import DIN_CONFIG, DINModel
    from src.retrieval.ranking import DIN_AD_COLS
    from src.train.config import get_config

    if name in ("deepfm", "dcn"):
        meta = spec.meta(False)
        c = get_config(name)
        mod = (DeepFMModule(meta, c["embed_dim"], c["hidden"], c["dropout"]) if name == "deepfm"
               else DCNModule(meta, c["embed_dim"], c["cross_layers"], c["hidden"], c["dropout"]))
    elif name in ("din", "din-mean"):
        c = dict(DIN_CONFIG)
        if name == "din-mean":
            c["pooling"] = "mean"
        mod = DINModel(spec.history_len, spec.din_query_positions, enc.ad_feat, DIN_AD_COLS).build(
            spec.meta(True), c)
    else:
        raise ValueError(f"unknown ranker {name!r}")
    mod.load_state_dict(torch.load(os.path.join(art, f"ranker_{name}.pt"), map_location="cpu"))
    return mod.eval()


def load_bundle(config: TwoStageConfig) -> TwoStageBundle:
    """Load the log, rebuild the encoder, and assemble the serving pipeline.

    The encoder is rebuilt from the dataset's parquet cache rather than
    unpickled, because encode() is deterministic and the rebuilt vocabularies
    are the ones the towers and rankers were trained against. On the real
    Taobao log this holds the impression arrays in memory, a few GB, which is
    the price of serving from the same objects the offline run uses.
    """
    from src.retrieval import index as ix
    from src.retrieval import record
    from src.retrieval.data import load
    from src.retrieval.features import ClickHistory, encode
    from src.retrieval.pipeline import TwoStagePipeline
    from src.retrieval.ranking import RankingSpec

    t0 = time.perf_counter()
    torch.set_num_threads(int(config.torch_threads))
    ix.set_threads(int(config.faiss_threads))

    data_dir = config.data_dir or _default_data_dir(config.source)
    data = load(config.source, data_dir)
    label = data.label()
    pids = list(data.notes.get("pids", []))
    art = os.path.join(record.results_dir(config.results_dir, data.synthetic), "artifacts")
    enc = encode(data)
    del data

    # The cold start row. Appended to the encoder's user table and to the
    # tower's copy of it, with every field at the out of vocabulary code 0.
    # It has no clicks, so its history is empty and its prior count is 0.
    cold_row = enc.n_user_rows
    enc.user_feat = np.vstack([enc.user_feat, np.zeros((1, enc.user_feat.shape[1]), dtype=enc.user_feat.dtype)])

    hist = ClickHistory(enc)
    spec = RankingSpec(enc, hist)
    tower, tcfg = _load_tower_with_cold(art, enc)
    corpus = np.load(os.path.join(art, "corpus.npy"))
    index = load_index(config.index, corpus, art)
    ranker = _load_ranker(art, config.ranker, spec, enc)
    pipe = TwoStagePipeline(tower, index, ranker, spec, enc, hist, tcfg["history_len"],
                            ranker_uses_history=config.ranker.startswith("din"))
    bundle = TwoStageBundle(
        config=config, label=label, enc=enc, hist=hist, spec=spec, tower=tower, tower_cfg=tcfg,
        index=index, ranker=ranker, pipeline=pipe,
        user_lookup=pd.Index(np.asarray(enc.user_raw_ids)),
        pid_lookup={p: i for i, p in enumerate(pids)},
        cold_user_row=cold_row, load_seconds=time.perf_counter() - t0,
    )
    bundle.info = {
        "dataset": label,
        "artifacts": art,
        "index": index.name,
        "index_bytes": index.memory_bytes(),
        "ranker": config.ranker,
        "corpus_ads": int(enc.n_ads),
        "known_users": int(cold_row),
        "torch_threads": int(config.torch_threads),
        "faiss_threads": int(config.faiss_threads),
        "state_as_of_ts": int(enc.cutoff_ts),
        "load_seconds": round(bundle.load_seconds, 3),
    }
    return bundle


def _load_tower_with_cold(art: str, enc):
    """Load the tower against the encoder's user table, cold start row included.

    The checkpoint's user_feat buffer has one row fewer than the extended
    table, so it is dropped from the state dict and the buffer comes from the
    encoder instead. The two agree on every other row by construction, since
    both came from the same deterministic encode().
    """
    from src.retrieval import two_tower as tt

    ck = torch.load(os.path.join(art, "two_tower.pt"), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    state = dict(ck["state_dict"])
    saved_users = state.pop("user_feat")
    if not torch.equal(saved_users, torch.as_tensor(enc.user_feat[:-1], dtype=torch.long)):
        raise RuntimeError("the checkpoint's user table does not match the rebuilt encoder, "
                           "so the artifacts and the data disagree")
    m = tt.TwoTowerRetriever(enc.ad_vocab_sizes, enc.user_vocab_sizes, enc.ad_feat, enc.user_feat,
                             embed_dim=cfg["embed_dim"], id_dim=cfg["id_dim"], small_dim=cfg["small_dim"],
                             hidden=cfg["hidden"], use_history=cfg["use_history"])
    missing, unexpected = m.load_state_dict(state, strict=False)
    if unexpected or [k for k in missing if k != "user_feat"]:
        raise RuntimeError(f"tower checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    return m.eval(), cfg


# ---------------------------------------------------------------------------
# Parity


def parity_check(bundle: TwoStageBundle, samples: int = 20, k: int = MAX_K, seed: int = 0,
                 tol: float = 1e-5) -> Dict[str, Any]:
    """Served pCTR against the offline frozen row score, for logged impressions.

    Picks test day impressions, serves each one's (user, placement, hour) with
    retrieval depth k, and for every one whose ad was retrieved compares the
    served probability of that ad with the ranker's score on rows_dataset in
    frozen mode. Impressions whose ad was not retrieved cannot be compared and
    are counted separately.
    """
    from src.retrieval.ranking import hour_of_day, rows_dataset, score_module

    enc = bundle.enc
    test = np.flatnonzero(enc.split_mask("test"))
    rng = np.random.default_rng(seed)
    pick = rng.choice(test, size=min(samples * 5, len(test)), replace=False)
    k = min(int(k), enc.n_ads)
    compared, not_retrieved, max_diff = 0, 0, 0.0
    worst = None
    uses_h = bundle.config.ranker.startswith("din")
    for i in pick:
        if compared >= samples:
            break
        u, a = int(enc.imp_user[i]), int(enc.imp_ad[i])
        pid = int(enc.imp_pid[i])
        hour = int(hour_of_day(np.array([enc.imp_ts[i]]))[0])
        r = bundle.pipeline.request(u, pid, hour, int(enc.cutoff_ts), k, n=k)
        hit = np.flatnonzero(r.ads == a)
        if len(hit) == 0:
            not_retrieved += 1
            continue
        served = float(r.scores[hit[0]])
        ds = rows_dataset(bundle.spec, enc, bundle.hist, np.array([i]), frozen=True, with_history=uses_h)
        offline = float(score_module(bundle.ranker, ds, torch.device("cpu"))[0])
        d = abs(served - offline)
        if d > max_diff:
            max_diff, worst = d, {"impression": int(i), "served": served, "offline": offline}
        compared += 1
    return {"passed": compared > 0 and max_diff <= tol, "compared": compared,
            "not_retrieved": not_retrieved, "max_abs_diff": max_diff, "tolerance": tol,
            "k": k, "worst": worst}


# ---------------------------------------------------------------------------
# Metrics


class TwoStageMetrics:
    """Request counters and one latency histogram per pipeline stage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests_total = 0
        self.requests_failed_total = 0
        self.cold_start_total = 0
        self.unknown_pid_total = 0
        self.candidates_ranked_total = 0
        self.stage = {s: Histogram(f"clickrank_two_stage_{s}_seconds") for s in STAGES}

    def observe(self, stage_ns: Dict[str, int], cold: bool, unknown_pid: bool, k: int) -> None:
        with self._lock:
            self.requests_total += 1
            self.cold_start_total += int(cold)
            self.unknown_pid_total += int(unknown_pid)
            self.candidates_ranked_total += int(k)
            for s in STAGES:
                if s in stage_ns:
                    self.stage[s].observe(stage_ns[s] / 1e9)

    def observe_failure(self) -> None:
        with self._lock:
            self.requests_total += 1
            self.requests_failed_total += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "requests_total": self.requests_total,
                "requests_failed_total": self.requests_failed_total,
                "cold_start_total": self.cold_start_total,
                "unknown_pid_total": self.unknown_pid_total,
                "candidates_ranked_total": self.candidates_ranked_total,
                "stages": {s: h.snapshot() for s, h in self.stage.items()},
            }

    def render_prometheus(self) -> str:
        with self._lock:
            lines = []
            for name, value, help_text in (
                ("clickrank_two_stage_requests_total", self.requests_total, "Recommend requests received."),
                ("clickrank_two_stage_requests_failed_total", self.requests_failed_total,
                 "Recommend requests that could not be served."),
                ("clickrank_two_stage_cold_start_total", self.cold_start_total,
                 "Requests served from the cold start user row."),
                ("clickrank_two_stage_unknown_pid_total", self.unknown_pid_total,
                 "Requests whose placement was not in the log."),
                ("clickrank_two_stage_candidates_ranked_total", self.candidates_ranked_total,
                 "Retrieved candidates scored by the ranker."),
            ):
                lines += [f"# HELP {name} {help_text}", f"# TYPE {name} counter", f"{name} {value}"]
            for s, h in self.stage.items():
                lines += h.prometheus_lines(f"Two stage {s} latency in seconds.")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Schemas


class RecommendRequest(BaseModel):
    model_config = {"extra": "forbid"}

    user_id: Union[int, str] = Field(..., description="Raw user id from the log. Unknown ids are cold starts.")
    pid: Union[int, str] = Field(..., description="Raw placement string such as 430548_1007, or its code.")
    hour: Optional[int] = Field(None, ge=0, le=23, description="Hour of day, Beijing time. Server clock if absent.")
    k: int = Field(100, ge=1, le=MAX_K, description="Retrieval depth, the candidates handed to the ranker.")
    n: int = Field(10, ge=1, le=MAX_N, description="How many ranked ads to return.")


class RecommendedAd(BaseModel):
    ad_id: int
    rank: int
    p_click: float


class RecommendResponse(BaseModel):
    user_id: Union[int, str]
    cold_start: bool
    pid_known: bool
    hour: int
    k: int
    retrieved: int
    ads: List[RecommendedAd]
    timings_us: Dict[str, float]


def beijing_hour(now: Optional[float] = None) -> int:
    t = time.time() if now is None else now
    return int(((int(t) + BEIJING_OFFSET_S) // 3600) % 24)


# ---------------------------------------------------------------------------
# App


def create_app(bundle: TwoStageBundle, metrics: Optional[TwoStageMetrics] = None,
               parity: Optional[Dict[str, Any]] = None, thread_pool_size: Optional[int] = None) -> FastAPI:
    """Build the app around a loaded bundle.

    The bundle is loaded before the app, as src/serving/app.py does with its
    engine, so a bundle that will not load fails the process instead of
    producing a server that is up and cannot recommend. parity is the result
    of parity_check when it was run. /readyz requires it to have passed when
    present.
    """
    metrics = metrics or TwoStageMetrics()
    started = time.time()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if thread_pool_size:
            try:
                import anyio.to_thread

                anyio.to_thread.current_default_thread_limiter().total_tokens = int(thread_pool_size)
            except Exception as exc:  # noqa: BLE001 the default pool still serves
                print(f"could not set the thread pool size to {thread_pool_size} ({exc})")
        yield

    app = FastAPI(title="ClickRank two stage recommend service", version="1.0.0", lifespan=lifespan)
    app.state.bundle = bundle
    app.state.metrics = metrics
    app.state.parity = parity

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        """Liveness. The process is up and the bundle is loaded."""
        return {"status": "ok", "uptime_seconds": round(time.time() - started, 3), **bundle.info}

    @app.get("/readyz")
    def readyz():
        """Readiness. Loaded, and the parity check passed if it was run."""
        ok = parity is None or bool(parity.get("passed"))
        body = {"ready": ok, "parity": parity}
        return JSONResponse(content=_jsonable(body), status_code=200 if ok else 503)

    @app.get("/metrics")
    def metrics_endpoint(format: str = Query(default="prometheus", pattern="^(prometheus|json)$")):
        if format == "json":
            return JSONResponse(content=_jsonable(metrics.snapshot()))
        return PlainTextResponse(content=metrics.render_prometheus(),
                                 media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.post("/v1/recommend", response_model=RecommendResponse)
    def recommend(req: RecommendRequest) -> RecommendResponse:
        row, cold = bundle.user_row(req.user_id)
        pid, pid_known = bundle.pid_code(req.pid)
        hour = beijing_hour() if req.hour is None else int(req.hour)
        k = min(int(req.k), bundle.enc.n_ads)
        try:
            r = bundle.pipeline.request(row, pid, hour, int(bundle.enc.cutoff_ts), k, n=int(req.n))
        except Exception as exc:  # noqa: BLE001 a failed request is a 500 and a metric
            metrics.observe_failure()
            raise HTTPException(status_code=500, detail=f"recommend failed. {type(exc).__name__} {exc}") from exc
        metrics.observe(r.stage_ns, cold, not pid_known, len(r.retrieved))
        raw = bundle.enc.ad_raw_ids
        return RecommendResponse(
            user_id=req.user_id, cold_start=cold, pid_known=pid_known, hour=hour, k=k,
            retrieved=int(len(r.retrieved)),
            ads=[RecommendedAd(ad_id=int(raw[a]), rank=i + 1, p_click=float(s))
                 for i, (a, s) in enumerate(zip(r.ads, r.scores))],
            timings_us={s: round(v / 1e3, 1) for s, v in r.stage_ns.items()},
        )

    return app


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def build_from_env() -> FastAPI:
    """uvicorn factory: read the config from CLICKRANK_TS_* environment variables."""
    env = os.environ
    cfg = TwoStageConfig(
        source=env.get("CLICKRANK_TS_SOURCE", "synthetic"),
        data_dir=env.get("CLICKRANK_TS_DATA") or None,
        results_dir=env.get("CLICKRANK_TS_RESULTS", "results"),
        index=env.get("CLICKRANK_TS_INDEX", "flat"),
        ranker=env.get("CLICKRANK_TS_RANKER", "deepfm"),
        torch_threads=int(env.get("CLICKRANK_TS_TORCH_THREADS", "1")),
        faiss_threads=int(env.get("CLICKRANK_TS_FAISS_THREADS", "1")),
        parity_samples=int(env.get("CLICKRANK_TS_PARITY", "0")),
    )
    bundle = load_bundle(cfg)
    parity = parity_check(bundle, samples=cfg.parity_samples) if cfg.parity_samples else None
    pool = env.get("CLICKRANK_TS_POOL")
    return create_app(bundle, parity=parity, thread_pool_size=int(pool) if pool else None)
