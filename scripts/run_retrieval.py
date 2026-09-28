"""M0 and M1: load the id bearing dataset, train the two tower model, export the corpus.

What it writes, under results/retrieval (or results/synthetic/retrieval when the
data is synthetic):

data_summary.jsonl       the counts the loader found, next to the dataset card
two_tower_curve.jsonl    one row per epoch: loss, seconds, held out hit rate
two_tower_sanity.jsonl   clicked ads versus random ads, and the popularity and
                         chance baselines at the same K
artifacts/               tower weights, corpus embeddings (one vector per ad),
                         and the frozen test queries. The encoder is
                         rebuilt from the parquet cache, which is deterministic. Git ignored.

Usage:
    python scripts/run_retrieval.py --source taobao --data data/taobao
    python scripts/run_retrieval.py --source synthetic --data data/synthetic_ids
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

from src.retrieval import record  # noqa: E402
from src.retrieval import two_tower as tt  # noqa: E402
from src.retrieval.data import check_against_card, load  # noqa: E402
from src.retrieval.features import encode  # noqa: E402
from src.train.trainer import get_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the two tower retrieval model.")
    ap.add_argument("--source", choices=["taobao", "avazu", "synthetic"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--history-len", type=int, default=20)
    ap.add_argument("--no-history", action="store_true")
    ap.add_argument("--no-logq", action="store_true", help="switch off the sampling bias correction")
    ap.add_argument("--eval-users", type=int, default=20000)
    ap.add_argument("--tag", default="", help="suffix for an ablation run, keeps artifacts apart")
    args = ap.parse_args()

    t0 = time.time()
    data = load(args.source, args.data)
    label = data.label()
    out = record.results_dir(args.results, data.synthetic)
    art = os.path.join(out, "artifacts" + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(art, exist_ok=True)
    print(f"loaded {label} in {time.time() - t0:.1f}s")

    summary = data.summary()
    if args.source == "taobao":
        summary["card_check"] = check_against_card(data)
    print(json.dumps(summary, indent=1, default=str))
    if not args.tag:
        record.write(os.path.join(out, "data_summary.jsonl"), {"summary": summary}, dataset=label)

    t1 = time.time()
    enc = encode(data)
    print(f"encoded in {time.time() - t1:.1f}s: {enc.notes}")
    del data

    device = get_device(args.device)
    cfg = tt.TrainConfig(
        embed_dim=args.embed_dim,
        history_len=args.history_len,
        use_history=not args.no_history,
        temperature=args.temperature,
        logq_correction=not args.no_logq,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        eval_users=args.eval_users,
    )
    print(f"device {device}, config {cfg}")
    model, hist, curve = tt.train(enc, cfg, device)
    variant = args.tag or "main"
    for row in curve:
        record.write(os.path.join(out, "two_tower_curve.jsonl"),
                     {"variant": variant, "config": cfg.__dict__, "device": str(device), **row},
                     dataset=label)

    # Export every corpus ad once. This is the candidate corpus for M2.
    t2 = time.time()
    corpus = tt.embed_corpus(model, enc.n_ads, device)
    export_s = time.time() - t2
    np.save(os.path.join(art, "corpus.npy"), corpus)

    # The full held out query set, frozen at the end of the training days.
    users, qh, clicked = tt.test_queries(enc, hist, cfg.history_len)
    qvecs = tt.embed_users(model, users, qh, device)
    np.save(os.path.join(art, "queries.npy"), qvecs)
    with open(os.path.join(art, "queries.pkl"), "wb") as fh:
        pickle.dump({"users": users, "history": qh, "clicked": clicked}, fh)

    sanity = tt.clicked_vs_random(model, users, qh, clicked, enc.n_ads, device)

    ks = (50, 100, 500)
    hr = tt.exact_hit_rate(qvecs, corpus, clicked, ks=ks)
    train_clicks = enc.imp_ad[enc.split_mask("train") & (enc.imp_clk == 1)]
    counts = np.bincount(train_clicks, minlength=enc.n_ads)
    popular = np.argsort(-counts, kind="stable")
    pairs = sum(len(c) for c in clicked)
    pop = {k: sum(int(np.isin(c, popular[:k]).sum()) for c in clicked) / max(pairs, 1) for k in ks}
    chance = {k: k / enc.n_ads for k in ks}

    row = {
        "variant": variant,
        "clicked_above_random": sanity,
        "test_users": int(len(users)),
        "test_click_pairs": int(pairs),
        "corpus_ads": int(enc.n_ads),
        "hit_rate_exact": {str(k): v for k, v in hr.items()},
        "hit_rate_popularity": {str(k): v for k, v in pop.items()},
        "hit_rate_chance": {str(k): v for k, v in chance.items()},
        "corpus_export_seconds": export_s,
        "history": "frozen at end of training days for every test query",
        "config": cfg.__dict__,
    }
    record.write(os.path.join(out, "two_tower_sanity.jsonl"), row, dataset=label)
    print(json.dumps(row, indent=1, default=str))

    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, os.path.join(art, "two_tower.pt"))
    print(f"done in {time.time() - t0:.1f}s, artifacts in {art}")


if __name__ == "__main__":
    main()
