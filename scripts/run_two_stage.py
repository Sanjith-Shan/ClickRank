"""M2, part three: two stage ranking against exhaustive ranking, and stage latency.

Needs run_retrieval.py (towers, corpus embeddings) and run_rankers.py (rankers,
test day scores). Writes under results/retrieval:

two_stage_quality.jsonl  Impression level. Every test day impression is scored
    by the ranker (exhaustive). Two stage at K keeps the ranker's score when
    the ad is in the user's top K retrieved and otherwise drops the ad. For AUC
    and GAUC a dropped ad ranks below every retrieved one. For NE a dropped ad
    gets one constant probability, the ranker's own mean prediction over the
    dropped impressions, so no label is used to set it. Reported for exact
    retrieval and for the serving index.

final_rank.jsonl  Request level. For a seeded sample of test day clicks, the
    request is (user, placement, hour) at the click. Exhaustive scores all
    corpus ads and records the clicked ad's rank. Two stage retrieves K,
    scores those, and records the rank if the clicked ad was retrieved.

stage_latency.jsonl  Per request on this CPU with one torch thread and one
    FAISS thread: user features and tower, index search, candidate feature
    build, ranking, and the end to end two stage request, against scoring the
    whole corpus. Exhaustive is also timed with every core, since that is its
    most favourable setting.

Usage:
    python scripts/run_two_stage.py --source taobao --serving-index "hnsw M=32 efSearch=..."
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import compute_auc, normalized_entropy  # noqa: E402
from src.models.dcn import DCNModule  # noqa: E402
from src.models.deepfm import DeepFMModule  # noqa: E402
from src.models.din import DIN_CONFIG, DINModule  # noqa: E402
from src.retrieval import index as ix  # noqa: E402
from src.retrieval import metrics as rm  # noqa: E402
from src.retrieval import record  # noqa: E402
from src.retrieval import two_tower as tt  # noqa: E402
from src.retrieval.data import load  # noqa: E402
from src.retrieval.features import ClickHistory, encode  # noqa: E402
from src.retrieval.latency import StageTimer, summarise  # noqa: E402
from src.retrieval.pipeline import TwoStagePipeline  # noqa: E402
from src.retrieval.ranking import DIN_AD_COLS, RankingSpec, hour_of_day  # noqa: E402
from src.train.config import get_config  # noqa: E402

KS = (50, 100, 500)


def load_tower(art: str, enc) -> tuple:
    ck = torch.load(os.path.join(art, "two_tower.pt"), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = tt.TwoTowerRetriever(enc.ad_vocab_sizes, enc.user_vocab_sizes, enc.ad_feat, enc.user_feat,
                             embed_dim=cfg["embed_dim"], id_dim=cfg["id_dim"], small_dim=cfg["small_dim"],
                             hidden=cfg["hidden"], use_history=cfg["use_history"])
    m.load_state_dict(ck["state_dict"])
    return m.eval(), cfg


def load_ranker(art: str, name: str, spec: RankingSpec, enc) -> torch.nn.Module:
    if name in ("deepfm", "dcn"):
        meta = spec.meta(False)
        c = get_config(name)
        mod = (DeepFMModule(meta, c["embed_dim"], c["hidden"], c["dropout"]) if name == "deepfm"
               else DCNModule(meta, c["embed_dim"], c["cross_layers"], c["hidden"], c["dropout"]))
    else:
        from src.models.din import DINModel

        c = dict(DIN_CONFIG)
        if name == "din-mean":
            c["pooling"] = "mean"
        mod = DINModel(spec.history_len, spec.din_query_positions, enc.ad_feat, DIN_AD_COLS).build(
            spec.meta(True), c)
    mod.load_state_dict(torch.load(os.path.join(art, f"ranker_{name}.pt"), map_location="cpu"))
    return mod.eval()


def find_index(corpus: np.ndarray, spec_str: str):
    """Parse 'kind k=v k=v' into a built index with its search settings."""
    parts = spec_str.split()
    kind, kv = parts[0], dict(p.split("=") for p in parts[1:])
    kv = {k: int(v) for k, v in kv.items()}
    search = {k: kv.pop(k) for k in ("nprobe", "efSearch") if k in kv}
    idx = ix.build_index(kind, corpus, **kv)
    for k, v in search.items():
        idx.set_search_param(k, v)
    return idx


def retrieved_mask(enc, users_of_imp: np.ndarray, ads_of_imp: np.ndarray, user_list: np.ndarray,
                   ids: np.ndarray, k: int) -> np.ndarray:
    """For each impression, whether its ad is in its user's top k."""
    n_ads = np.int64(enc.n_ads)
    pos = np.searchsorted(user_list, users_of_imp)
    keys = np.sort((user_list[:, None].astype(np.int64) * n_ads + ids[:, :k].astype(np.int64)).ravel())
    imp_keys = users_of_imp.astype(np.int64) * n_ads + ads_of_imp.astype(np.int64)
    loc = np.searchsorted(keys, imp_keys)
    loc = np.minimum(loc, len(keys) - 1)
    assert (user_list[pos] == users_of_imp).all()
    return keys[loc] == imp_keys


def quality(y, p, groups, keep) -> dict:
    """Two stage metrics for one keep mask. Dropped ads rank last and share one probability."""
    floor = float(p[~keep].mean()) if (~keep).any() else 0.0
    # Rank based metrics: dropped ads sit below every kept one, tied together.
    ranked = np.where(keep, p + 1.0, 0.0)
    calibrated = np.where(keep, p, floor)
    g = rm.fast_group_auc(y, ranked, groups)
    return {
        "auc": compute_auc(y, ranked),
        "gauc_by_user": g["gauc"],
        "ne": normalized_entropy(y, calibrated),
        "impressions_kept": float(keep.mean()),
        "clicks_kept": float(keep[y > 0].mean()) if (y > 0).any() else float("nan"),
        "dropped_probability": floor,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Two stage versus exhaustive ranking.")
    ap.add_argument("--source", choices=["taobao", "avazu", "synthetic"], required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", default="deepfm,dcn")
    ap.add_argument("--latency-models", default="deepfm,dcn,din")
    ap.add_argument("--serving-index", default="hnsw M=32 efConstruction=200 efSearch=512")
    ap.add_argument("--final-rank-requests", type=int, default=300)
    ap.add_argument("--latency-requests", type=int, default=500)
    ap.add_argument("--exhaustive-latency-requests", type=int, default=20)
    ap.add_argument("--skip", default="", help="comma list of phases to skip: quality,rank,latency")
    args = ap.parse_args()
    skip = set(s for s in args.skip.split(",") if s)

    data_dir = args.data or {"taobao": "data/taobao", "synthetic": "data/synthetic_ids"}.get(args.source)
    data = load(args.source, data_dir)
    label = data.label()
    out = record.results_dir(args.results, data.synthetic)
    art = os.path.join(out, "artifacts")
    enc = encode(data)
    del data
    hist = ClickHistory(enc)
    spec = RankingSpec(enc, hist)
    tower, tcfg = load_tower(art, enc)
    corpus = np.load(os.path.join(art, "corpus.npy"))
    n_ads = enc.n_ads
    torch.set_num_threads(os.cpu_count())
    ix.set_threads(os.cpu_count())

    test_idx = np.load(os.path.join(art, "test_idx.npy"))
    y = enc.imp_clk[test_idx].astype(np.float32)
    iu = enc.imp_user[test_idx]
    ia = enc.imp_ad[test_idx]

    print("building serving index", args.serving_index)
    serving = find_index(corpus, args.serving_index)
    flat = ix.build_index("flat", corpus)

    # ---------------------------------------------------------------- quality
    if "quality" not in skip:
        users = np.unique(iu)
        uh = hist.history(users, np.full(len(users), enc.cutoff_ts), tcfg["history_len"])
        t0 = time.time()
        uvec = tt.embed_users(tower, users, uh, torch.device("cpu"))
        print(f"embedded {len(users):,} test day users in {time.time() - t0:.1f}s")
        ids = {}
        for nm, idx in (("exact", flat), ("serving", serving)):
            t0 = time.time()
            ids[nm] = idx.search(uvec, max(KS))[1]
            print(f"{nm} search for all test users {time.time() - t0:.1f}s")
        for model in [m for m in args.models.split(",") if m]:
            p = np.load(os.path.join(art, f"test_scores_{model}.npy"))
            base = {
                "auc": compute_auc(y, p),
                "gauc_by_user": rm.fast_group_auc(y, p, iu)["gauc"],
                "ne": normalized_entropy(y, p),
            }
            rows = {"exhaustive": base}
            for nm in ("exact", "serving"):
                for k in KS:
                    keep = retrieved_mask(enc, iu, ia, users, ids[nm], k)
                    rows[f"{nm}@{k}"] = quality(y, p, iu, keep)
            record.write(os.path.join(out, "two_stage_quality.jsonl"), {
                "ranker": model, "serving_index": serving.name, "test_impressions": int(len(y)),
                "test_users": int(len(users)), "corpus_ads": int(n_ads),
                "results": rows,
                "definitions": "dropped ads rank below all retrieved for AUC and GAUC; for NE they "
                               "share the ranker's mean prediction over dropped impressions",
            }, dataset=label)
            print(model, json.dumps({k: {m: round(v, 4) for m, v in r.items()} for k, r in rows.items()}))

    # ------------------------------------------------------------- final rank
    click_idx = test_idx[y > 0]
    rng = np.random.default_rng(7)
    _, first = np.unique(enc.imp_user[click_idx], return_index=True)
    one_per_user = click_idx[first]
    req = np.sort(rng.choice(one_per_user, size=min(args.final_rank_requests, len(one_per_user)),
                             replace=False))
    if "rank" not in skip:
        for model in [m for m in args.models.split(",") if m]:
            ranker = load_ranker(art, model, spec, enc)
            pipe = TwoStagePipeline(tower, serving, ranker, spec, enc, hist, tcfg["history_len"],
                                    ranker_uses_history=model.startswith("din"))
            ex_ranks, ts_ranks, retrieved_at = [], {k: [] for k in KS}, {k: 0 for k in KS}
            t0 = time.time()
            for n, i in enumerate(req):
                u, a = int(enc.imp_user[i]), int(enc.imp_ad[i])
                pid, hour = int(enc.imp_pid[i]), int(hour_of_day(np.array([enc.imp_ts[i]]))[0])
                s = pipe.exhaustive_scores(u, pid, hour, enc.cutoff_ts)
                ex_ranks.append(rm.rank_of(s[a], s))
                hvec = tt.embed_users(tower, np.array([u]),
                                      hist.history(np.array([u]), np.array([enc.cutoff_ts]),
                                                   tcfg["history_len"]), torch.device("cpu"))
                cand = serving.search(hvec, max(KS))[1][0]
                for k in KS:
                    ck = cand[:k]
                    if a in ck:
                        retrieved_at[k] += 1
                        sk = s[ck]  # same ranker, same features, so the scores are the same rows
                        ts_ranks[k].append(rm.rank_of(s[a], sk))
                    else:
                        ts_ranks[k].append(float("inf"))
                if n % 50 == 0:
                    print(f"  {model} request {n}/{len(req)} {time.time() - t0:.0f}s")
            ex = np.asarray(ex_ranks)

            def within(r, n):
                return float(np.mean(np.asarray(r) <= n))

            row = {
                "ranker": model, "requests": int(len(req)), "serving_index": serving.name,
                "corpus_ads": int(n_ads),
                "exhaustive": {"median_rank": float(np.median(ex)), "top1": within(ex, 1),
                               "top10": within(ex, 10), "top50": within(ex, 50), "top100": within(ex, 100)},
                "two_stage": {str(k): {"retrieved": retrieved_at[k] / len(req),
                                       "top1": within(ts_ranks[k], 1), "top10": within(ts_ranks[k], 10),
                                       "top50": within(ts_ranks[k], 50)} for k in KS},
                "request": "one test day click per sampled user; features frozen at end of training",
            }
            record.write(os.path.join(out, "final_rank.jsonl"), row, dataset=label)
            print(json.dumps(row, indent=1))

    # ---------------------------------------------------------------- latency
    if "latency" not in skip:
        torch.set_num_threads(1)
        ix.set_threads(1)
        lat_req = req[: args.latency_requests] if len(req) >= args.latency_requests else np.sort(
            rng.choice(one_per_user, size=min(args.latency_requests, len(one_per_user)), replace=False))
        indexes = {"flat": flat, "serving": serving}
        for model in [m for m in args.latency_models.split(",") if m]:
            ranker = load_ranker(art, model, spec, enc)
            for iname, idx in indexes.items():
                pipe = TwoStagePipeline(tower, idx, ranker, spec, enc, hist, tcfg["history_len"],
                                        ranker_uses_history=model.startswith("din"))
                for k in KS:
                    timer = StageTimer()
                    for w in lat_req[:10]:
                        pipe.request(int(enc.imp_user[w]), int(enc.imp_pid[w]), 12, int(enc.cutoff_ts), k)
                    for i in lat_req:
                        r = pipe.request(int(enc.imp_user[i]), int(enc.imp_pid[i]),
                                         int(hour_of_day(np.array([enc.imp_ts[i]]))[0]), int(enc.cutoff_ts), k)
                        for st, ns in r.stage_ns.items():
                            timer.add(st, ns)
                    record.write(os.path.join(out, "stage_latency.jsonl"), {
                        "mode": "two_stage", "ranker": model, "index": idx.name, "k": k,
                        "torch_threads": 1, "faiss_threads": 1, "stages": timer.summary(),
                    }, dataset=label)
                    print(f"{model} {iname} K={k}: total p50 "
                          f"{timer.summary()['total']['p50_us'] / 1000:.2f} ms")
            if model.startswith("din"):
                continue  # scoring every ad with DIN is a minute a request; not measured
            pipe = TwoStagePipeline(tower, None, ranker, spec, enc, hist, tcfg["history_len"])
            for threads in (1, os.cpu_count()):
                torch.set_num_threads(threads)
                timer = StageTimer()
                pipe.exhaustive(int(enc.imp_user[lat_req[0]]), 0, 12, int(enc.cutoff_ts))
                for i in lat_req[: args.exhaustive_latency_requests]:
                    r = pipe.exhaustive(int(enc.imp_user[i]), int(enc.imp_pid[i]),
                                        int(hour_of_day(np.array([enc.imp_ts[i]]))[0]), int(enc.cutoff_ts))
                    for st, ns in r.stage_ns.items():
                        timer.add(st, ns)
                record.write(os.path.join(out, "stage_latency.jsonl"), {
                    "mode": "exhaustive", "ranker": model, "k": int(n_ads), "torch_threads": threads,
                    "stages": timer.summary(),
                }, dataset=label)
                print(f"{model} exhaustive threads={threads}: p50 {timer.summary()['total']['p50_us'] / 1e3:.0f} ms")
            torch.set_num_threads(1)


if __name__ == "__main__":
    main()
