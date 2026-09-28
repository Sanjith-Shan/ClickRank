"""FAISS indexes over the ad tower's embeddings, exact and approximate.

Every index here scores by inner product. The towers L2 normalise their output,
so inner product is cosine similarity and the ranking matches the one the model
was trained with.

Four kinds are wrapped:

flat      IndexFlatIP. Scores the query against every corpus vector. This is
          the exact baseline every approximate index is measured against.
ivf_flat  IndexIVFFlat. k-means splits the corpus into nlist cells and a query
          scans only the nprobe cells whose centroids score highest. Vectors
          are stored uncompressed, so the only loss is cells not visited.
hnsw      IndexHNSWFlat. A layered proximity graph walked greedily. M is the
          number of neighbours per node, efConstruction the beam width while
          building, efSearch the beam width while searching.
ivf_pq    IndexIVFPQ. The IVF layout with each vector product quantised to m
          codes of nbits bits, the compressed option when memory matters more
          than the last points of recall.

nprobe and efSearch are search time settings, so a sweep builds each index once
and then walks its search settings. default_sweep is laid out that way.

Sizes come from faiss.serialize_index, which is the index as it would be
written to disk. That is an honest figure for what serving has to hold, not an
estimate from vector counts.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import faiss

KMEANS_SEED = 1234


def set_threads(n: int) -> None:
    """Set the number of OpenMP threads FAISS uses for search and training."""
    faiss.omp_set_num_threads(int(n))


def _as_f32(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32)


def _train_sample(vectors: np.ndarray, n: int, seed: int) -> np.ndarray:
    """A seeded random subset for k-means training, or everything if small."""
    if n >= len(vectors):
        return vectors
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(vectors), size=n, replace=False))
    return vectors[idx]


class Index:
    """One built FAISS index plus what a results row needs to say about it.

    kind and params describe how it was built. search_params holds the current
    search time settings (nprobe, efSearch) and name renders both, so a row in
    results/ reads like "ivf_flat nlist=1024 nprobe=16".
    """

    def __init__(self, kind: str, index, params: Dict[str, object], build_seconds: float,
                 search_params: Optional[Dict[str, object]] = None):
        self.kind = kind
        self.index = index
        self.params = dict(params)
        self.build_seconds = float(build_seconds)
        self.search_params: Dict[str, object] = {}
        for k, v in (search_params or {}).items():
            self.set_search_param(k, v)

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    @property
    def name(self) -> str:
        parts = [self.kind]
        parts += [f"{k}={v}" for k, v in self.params.items()]
        parts += [f"{k}={v}" for k, v in self.search_params.items()]
        return " ".join(parts)

    def set_search_param(self, name: str, value) -> None:
        """Change a search time setting without rebuilding the index.

        nprobe applies to the IVF kinds and efSearch to HNSW. Anything else is
        an error, since a silently ignored setting would make a sweep lie.
        """
        if name == "nprobe" and self.kind in ("ivf_flat", "ivf_pq"):
            self.index.nprobe = int(value)
        elif name == "efSearch" and self.kind == "hnsw":
            self.index.hnsw.efSearch = int(value)
        else:
            raise ValueError(f"{name!r} is not a search parameter of {self.kind}")
        self.search_params[name] = int(value)

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Top k by inner product. Returns (scores, ids), both (Q, k).

        When fewer than k vectors are reachable FAISS pads ids with -1, which
        the metrics ignore.
        """
        q = _as_f32(queries)
        if q.ndim == 1:
            q = q[None, :]
        return self.index.search(q, int(k))

    def memory_bytes(self) -> int:
        """Size of the serialised index in bytes."""
        return int(len(faiss.serialize_index(self.index)))


def build_index(kind: str, vectors: np.ndarray, **params) -> Index:
    """Build one index of the given kind over vectors.

    Build parameters by kind:
      flat      none
      ivf_flat  nlist, train_size (default min(N, 256 * nlist)), nprobe
      hnsw      M (32), efConstruction (200), efSearch (64)
      ivf_pq    nlist, m (16), nbits (8), train_size, nprobe
    k-means is seeded so the same corpus gives the same cells.
    """
    x = _as_f32(vectors)
    n, d = x.shape
    t0 = time.perf_counter()
    search = {}
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
        index.add(x)
        built = {}
    elif kind in ("ivf_flat", "ivf_pq"):
        nlist = int(params.get("nlist", max(1, int(4 * math.sqrt(n)))))
        train_size = int(params.get("train_size", min(n, 256 * nlist)))
        quantizer = faiss.IndexFlatIP(d)
        if kind == "ivf_flat":
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            built = {"nlist": nlist}
        else:
            m = int(params.get("m", 16))
            nbits = int(params.get("nbits", 8))
            index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
            built = {"nlist": nlist, "m": m, "nbits": nbits}
        index.cp.seed = KMEANS_SEED
        index.train(_train_sample(x, train_size, KMEANS_SEED))
        index.add(x)
        # The Python wrapper keeps a reference to the quantizer on the index,
        # so it lives as long as the index does.
        search["nprobe"] = int(params.get("nprobe", 1))
    elif kind == "hnsw":
        M = int(params.get("M", 32))
        efc = int(params.get("efConstruction", 200))
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = efc
        index.add(x)
        built = {"M": M, "efConstruction": efc}
        search["efSearch"] = int(params.get("efSearch", 64))
    else:
        raise ValueError(f"unknown index kind {kind!r}")
    seconds = time.perf_counter() - t0
    return Index(kind, index, built, seconds, search)


def _pow2(x: float) -> int:
    return int(2 ** round(math.log2(max(x, 1.0))))


def default_sweep(n_vectors: int) -> List[Tuple[str, Dict[str, object], List[Dict[str, object]]]]:
    """The M2 sweep as (kind, build params, list of search settings).

    IVF uses nlist near 4 * sqrt(N) rounded to a power of two, plus one step
    either side, and nprobe from 1 up to a quarter of nlist. HNSW uses M of 16
    and 32 with efSearch from 16 to 512. IVF-PQ is built at the middle nlist
    with 16 and 32 byte codes. The flat index is first, since every recall
    figure is measured against it.
    """
    n = max(int(n_vectors), 1)
    base = _pow2(4 * math.sqrt(n))
    # A cell should hold a few dozen vectors at least, or k-means has nothing
    # to cluster and FAISS warns about it.
    cap = max(1, n // 39)
    nlists = sorted({min(v, cap) for v in (base // 2, base, base * 2) if v >= 1})
    sweep: List[Tuple[str, Dict[str, object], List[Dict[str, object]]]] = [("flat", {}, [{}])]
    for nl in nlists:
        probes = [p for p in (1, 4, 16, 64, 256) if p <= max(1, nl // 4)] or [1]
        sweep.append(("ivf_flat", {"nlist": nl}, [{"nprobe": p} for p in probes]))
    efs = [16, 32, 64, 128, 256, 512]
    for M in (16, 32):
        sweep.append(("hnsw", {"M": M, "efConstruction": 200}, [{"efSearch": e} for e in efs]))
    mid = nlists[len(nlists) // 2]
    probes = [p for p in (4, 16, 64) if p <= max(1, mid // 4)] or [1]
    for m in (16, 32):
        sweep.append(("ivf_pq", {"nlist": mid, "m": m, "nbits": 8}, [{"nprobe": p} for p in probes]))
    return sweep


def _percentiles_us(ns: np.ndarray) -> Dict[str, float]:
    us = ns.astype(np.float64) / 1e3
    return {
        "p50_us": float(np.percentile(us, 50)),
        "p95_us": float(np.percentile(us, 95)),
        "p99_us": float(np.percentile(us, 99)),
        "mean_us": float(us.mean()),
    }


def time_search(index: Index, queries: np.ndarray, k: int, batch_size: int = 1,
                repeats: int = 1, warmup: int = 10) -> Dict[str, float]:
    """Time searches of queries at top k.

    batch_size 1 calls search once per query, which is what one request costs,
    and reports per query percentiles in microseconds. A larger batch_size
    splits the queries into batches and reports the per batch percentiles plus
    throughput in queries per second, the offline scoring view. The first
    warmup calls are not timed.
    """
    q = _as_f32(queries)
    nq = len(q)
    bs = max(1, int(batch_size))
    for i in range(min(warmup, nq)):
        index.search(q[i:i + 1], k)
    samples = []
    total_ns = 0
    for _ in range(max(1, repeats)):
        for s in range(0, nq, bs):
            batch = q[s:s + bs]
            t0 = time.perf_counter_ns()
            index.search(batch, k)
            dt = time.perf_counter_ns() - t0
            samples.append(dt)
            total_ns += dt
    out = _percentiles_us(np.asarray(samples))
    out["batch_size"] = bs
    out["queries"] = nq * max(1, repeats)
    out["qps"] = float(out["queries"] / (total_ns / 1e9)) if total_ns else float("nan")
    return out
