"""M2, part two: train the rankers on the retrieval dataset and score the test day.

DeepFM and DCN are the repo's existing models with a new feature spec
(src/retrieval/ranking.py). DIN adds target attention over the user's click
history, and din-mean is the same model with the attention replaced by a mean,
which is the ablation that says whether attention earns its cost.

Training rows are a seeded uniform sample of the training days, all but the
last training day. Validation for early stopping is a sample of the last
training day. The test set is every impression on the test day, with features
frozen at the end of the training days. GAUC is grouped by real user id, which
Criteo cannot do.

Writes results/retrieval/ranker_eval.jsonl, one row per model, and the model
weights under artifacts/.

Usage:
    python scripts/run_rankers.py --source taobao --models deepfm,dcn,din,din-mean
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import compute_auc, compute_logloss, normalized_entropy  # noqa: E402
from src.retrieval import record  # noqa: E402
from src.retrieval.data import load  # noqa: E402
from src.retrieval.features import ClickHistory, encode  # noqa: E402
from src.retrieval.metrics import fast_group_auc  # noqa: E402
from src.retrieval.ranking import RankingSpec, fit_ranker, rows_dataset, score  # noqa: E402
from src.train.trainer import get_device, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate the rankers on the retrieval data.")
    ap.add_argument("--source", choices=["taobao", "avazu", "synthetic"], required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", default="deepfm,dcn,din,din-mean")
    ap.add_argument("--train-rows", type=int, default=8_000_000)
    ap.add_argument("--val-rows", type=int, default=1_000_000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--history-len", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    data_dir = args.data or {"taobao": "data/taobao", "synthetic": "data/synthetic_ids"}.get(args.source)
    data = load(args.source, data_dir)
    label = data.label()
    out = record.results_dir(args.results, data.synthetic)
    art = os.path.join(out, "artifacts")
    os.makedirs(art, exist_ok=True)
    enc = encode(data)
    last_train_day = data.train_days[1]
    del data

    hist = ClickHistory(enc)
    spec = RankingSpec(enc, hist, history_len=args.history_len)
    rng = np.random.default_rng(args.seed)
    train_pool = np.flatnonzero(enc.split_mask("train") & (enc.imp_day < last_train_day))
    val_pool = np.flatnonzero(enc.imp_day == last_train_day)
    test_idx = np.flatnonzero(enc.split_mask("test"))
    tr = np.sort(rng.choice(train_pool, size=min(args.train_rows, len(train_pool)), replace=False))
    va = np.sort(rng.choice(val_pool, size=min(args.val_rows, len(val_pool)), replace=False))
    print(f"{label}: train rows {len(tr):,} of {len(train_pool):,}, val {len(va):,}, test {len(test_idx):,}")

    device = get_device(args.device)
    y_test = enc.imp_clk[test_idx].astype(np.float32)
    groups = enc.imp_user[test_idx]
    np.save(os.path.join(art, "test_idx.npy"), test_idx)

    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        with_h = name.startswith("din")
        t0 = time.time()
        train_ds = rows_dataset(spec, enc, hist, tr, frozen=False, with_history=with_h)
        val_ds = rows_dataset(spec, enc, hist, va, frozen=False, with_history=with_h)
        build_s = time.time() - t0
        set_seed(args.seed)
        t1 = time.time()
        model = fit_ranker(name, train_ds, val_ds, spec, enc, overrides={"epochs": args.epochs},
                           device=device)
        train_s = time.time() - t1
        del train_ds, val_ds

        test_ds = rows_dataset(spec, enc, hist, test_idx, frozen=True, with_history=with_h)
        t2 = time.time()
        p = score(model, test_ds)
        score_s = time.time() - t2
        del test_ds
        g = fast_group_auc(y_test, p, groups)
        row = {
            "model": name,
            "auc": compute_auc(y_test, p),
            "logloss": compute_logloss(y_test, p),
            "ne": normalized_entropy(y_test, p),
            "gauc_by_user": g["gauc"],
            "gauc_detail": g,
            "mean_pred": float(p.mean()),
            "test_ctr": float(y_test.mean()),
            "train_rows": int(len(tr)),
            "val_rows": int(len(va)),
            "test_rows": int(len(test_idx)),
            "epochs_max": args.epochs,
            "params": int(sum(x.numel() for x in model.module.parameters())),
            "feature_build_seconds": build_s,
            "train_seconds": train_s,
            "test_score_seconds": score_s,
            "device": str(device),
            "features": spec.field_names + (["history"] if with_h else []),
            "test_features": "frozen at end of training days",
        }
        record.write(os.path.join(out, "ranker_eval.jsonl"), row, dataset=label)
        print(f"{name}: AUC {row['auc']:.4f} NE {row['ne']:.4f} GAUC {row['gauc_by_user']:.4f} "
              f"train {train_s:.0f}s")
        np.save(os.path.join(art, f"test_scores_{name}.npy"), p)
        torch.save(model.module.state_dict(), os.path.join(art, f"ranker_{name}.pt"))
        with open(os.path.join(art, f"ranker_{name}_meta.pkl"), "wb") as fh:
            pickle.dump({"name": name, "history_len": args.history_len, "with_history": with_h,
                         "epochs": args.epochs}, fh)
        del model


if __name__ == "__main__":
    main()
