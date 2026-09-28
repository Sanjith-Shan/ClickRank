"""M2, part one: FAISS indexes over the corpus, recall against exact, and hit rate.

Reads the artifacts run_retrieval.py wrote (corpus embeddings, the frozen test
queries and what each test user clicked) and writes, under results/retrieval:

hit_rate.jsonl      exact search hit rate at K in {50, 100, 500} over every test
                    (user, clicked ad) pair, next to the popularity and chance
                    baselines at the same K.
index_sweep.jsonl   one row per index and search setting: build seconds, bytes,
                    recall@K against exact, hit rate@K, and single query search
                    latency at each K with one FAISS thread.

Recall is against IndexFlatIP on the same embeddings. It says how much of the
exact top K an approximate index returns. Hit rate says whether retrieval puts
the ad the user went on to click in front of the ranker at all. They answer
different questions and both are reported.

Usage:
    python scripts/run_index_sweep.py --source taobao
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval import index as ix  # noqa: E402
from src.retrieval import metrics as rm  # noqa: E402
from src.retrieval import record  # noqa: E402
from src.retrieval.data import load  # noqa: E402
from src.retrieval.features import encode  # noqa: E402

KS = (50, 100, 500)


def main() -> None:
    ap = argparse.ArgumentParser(description="FAISS index sweep and hit rate.")
    ap.add_argument("--source", choices=["taobao", "avazu", "synthetic"], required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--results", default="results")
    ap.add_argument("--recall-queries", type=int, default=20000,
                    help="queries used for recall and hit rate in the sweep")
    ap.add_argument("--latency-queries", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=0, help="FAISS threads for batch search, 0 = all")
    ap.add_argument("--quick", action="store_true", help="a small sweep, for smoke tests")
    args = ap.parse_args()

    data_dir = args.data or {"taobao": "data/taobao", "synthetic": "data/synthetic_ids"}.get(args.source)
    data = load(args.source, data_dir)
    label = data.label()
    out = record.results_dir(args.results, data.synthetic)
    art = os.path.join(out, "artifacts")
    corpus = np.load(os.path.join(art, "corpus.npy"))
    queries = np.load(os.path.join(art, "queries.npy"))
    with open(os.path.join(art, "queries.pkl"), "rb") as fh:
        q = pickle.load(fh)
    clicked = q["clicked"]
    n_ads = len(corpus)
    print(f"{label}: corpus {n_ads:,} x {corpus.shape[1]}, test queries {len(queries):,}")

    enc = encode(data)
    del data
    train_clicks = enc.imp_ad[enc.split_mask("train") & (enc.imp_clk == 1)]

    batch_threads = args.threads or os.cpu_count()
    ix.set_threads(batch_threads)

    # Exact search over every test query. This is both the recall reference and
    # the headline hit rate.
    flat = ix.build_index("flat", corpus)
    t0 = time.time()
    _, exact_ids = flat.search(queries, max(KS))
    exact_s = time.time() - t0
    hr = rm.hit_rate(exact_ids, clicked, KS)
    pop = rm.popularity_hit_rate(train_clicks, n_ads, clicked, KS)
    row = {
        "index": flat.name,
        "queries": int(len(queries)),
        "click_pairs": int(sum(len(c) for c in clicked)),
        "corpus_ads": int(n_ads),
        "hit_rate": {str(k): v for k, v in hr.items()},
        "hit_rate_popularity": {str(k): v for k, v in pop.items()},
        "hit_rate_chance": {str(k): rm.random_baseline_rate(k, n_ads) for k in KS},
        "candidate_reduction": {str(k): n_ads / k for k in KS},
        "batch_search_seconds_all_queries": exact_s,
        "faiss_threads": batch_threads,
        "history": "frozen at end of training days",
    }
    record.write(os.path.join(out, "hit_rate.jsonl"), row, dataset=label)
    print({k: round(v["rate"], 4) for k, v in hr.items()}, "popularity",
          {k: round(v["rate"], 4) for k, v in pop.items()})

    # The sweep runs on a seeded subset of the queries, so each setting costs
    # seconds rather than minutes. The subset's exact results come from above.
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(len(queries), size=min(args.recall_queries, len(queries)), replace=False))
    qs = queries[sub]
    ex = exact_ids[sub]
    cl = [clicked[i] for i in sub]
    lat_q = qs[: args.latency_queries]

    sweep = ix.default_sweep(n_ads)
    if args.quick:
        sweep = [s for s in sweep if s[0] in ("flat", "ivf_flat", "hnsw")][:3]
        sweep = [(k, b, s[:2]) for k, b, s in sweep]

    for kind, build_params, settings in sweep:
        ix.set_threads(batch_threads)
        idx = flat if kind == "flat" else ix.build_index(kind, corpus, **build_params)
        mem = idx.memory_bytes()
        for setting in settings or [{}]:
            for name, value in setting.items():
                idx.set_search_param(name, value)
            ix.set_threads(batch_threads)
            t0 = time.time()
            _, ids = idx.search(qs, max(KS))
            batch_s = time.time() - t0
            recall = {str(k): rm.recall_at_k(ids[:, :k], ex[:, :k], k) for k in KS}
            hits = rm.hit_rate(ids, cl, KS)
            # Per request latency: one query per call, one FAISS thread, the
            # way a serving process that handles one request per core runs.
            ix.set_threads(1)
            lat = {str(k): ix.time_search(idx, lat_q, k, batch_size=1) for k in KS}
            r = {
                "index": idx.name,
                "kind": kind,
                "build_params": build_params,
                "search_params": dict(idx.search_params),
                "build_seconds": idx.build_seconds,
                "bytes": mem,
                "recall_vs_exact": recall,
                "hit_rate": {k: v["rate"] for k, v in hits.items()},
                "recall_queries": int(len(qs)),
                "batch_search_seconds": batch_s,
                "batch_threads": batch_threads,
                "latency_1thread": lat,
            }
            record.write(os.path.join(out, "index_sweep.jsonl"), r, dataset=label)
            print(f"{idx.name:42s} bytes={mem/2**20:7.1f}MB recall@100={recall['100']:.4f} "
                  f"hit@100={hits[100]['rate']:.4f} p50@100={lat['100']['p50_us']:.0f}us")
        if kind != "flat":
            ix.faiss.write_index(idx.index, os.path.join(art, f"{kind}_{'_'.join(f'{k}{v}' for k, v in build_params.items())}.faiss"))


if __name__ == "__main__":
    main()
