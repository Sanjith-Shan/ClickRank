"""Model freshness, the staleness curve and the update strategy comparison.

Reproduces the data freshness experiment of He et al., "Practical Lessons from
Predicting Clicks on Ads at Facebook" (ADKDD 2014, Section 5), with the DeepFM
ranker on the Taobao log. The logic is in src/retrieval/freshness.py.

Staleness. One DeepFM per training day d, trained from scratch on --train-rows
rows from the --window-days days ending at d, scored on the test day. NE is
reported relative to the freshest model.

Update strategy. A base model on days 1 to --base-days, then no update, a one
pass warm start on each new day, and a full retrain on every training day, all
scored on the test day with their training wall clock and rows processed.

Every row goes to results/retrieval/freshness.jsonl (results/synthetic/... for
synthetic data) with its machine and load, then one summary row per experiment
with the mean and std across seeds.

Usage:
    python scripts/run_freshness.py --source taobao --device mps --seeds 0,1
    python scripts/run_freshness.py --source synthetic --quick
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval import freshness as F  # noqa: E402
from src.retrieval import record  # noqa: E402
from src.retrieval.data import load  # noqa: E402
from src.retrieval.features import ClickHistory, encode  # noqa: E402
from src.retrieval.ranking import RankingSpec  # noqa: E402
from src.train.trainer import get_device  # noqa: E402

QUICK = {"train_rows": 20_000, "val_rows": 5_000, "epochs": 1, "test_rows": 50_000, "batch_size": 512}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="DeepFM freshness on the retrieval data.")
    ap.add_argument("--source", choices=["taobao", "synthetic"], required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--results", default="results")
    ap.add_argument("--train-rows", type=int, default=None,
                    help="training rows per day (default 1,500,000, or 20,000 with --quick)")
    ap.add_argument("--val-rows", type=int, default=None)
    ap.add_argument("--test-rows", type=int, default=None, help="sample the test day (default all rows)")
    ap.add_argument("--epochs", type=int, default=None, help="epochs for models trained from scratch")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="training batch size (default the ranker's 4096, or 512 with --quick)")
    ap.add_argument("--finetune-epochs", type=int, default=1)
    ap.add_argument("--finetune-lr", type=float, default=None)
    ap.add_argument("--window-days", type=int, default=1)
    ap.add_argument("--base-days", type=int, default=4)
    ap.add_argument("--experiments", default="staleness,update")
    ap.add_argument("--stale-seeds", default=None,
                    help="seeds for the staleness curve (default the first of --seeds)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seeds", default="0,1", help="comma separated, used for the update comparison")
    ap.add_argument("--quick", action="store_true", help="small rows, one epoch, one seed, sampled test day")
    args = ap.parse_args(argv)

    q = QUICK if args.quick else {}
    cfg = F.FreshnessConfig(
        train_rows=args.train_rows or q.get("train_rows", 1_500_000),
        val_rows=args.val_rows or q.get("val_rows", 200_000),
        epochs=args.epochs or q.get("epochs", 2),
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        window_days=args.window_days,
        base_days=args.base_days,
        test_rows=args.test_rows or q.get("test_rows"),
    )
    batch_size = args.batch_size or q.get("batch_size")
    if batch_size:
        cfg.overrides["batch_size"] = int(batch_size)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.quick:
        seeds = seeds[:1]
    stale_seeds = [int(s) for s in args.stale_seeds.split(",")] if args.stale_seeds else seeds[:1]
    exps = {e.strip() for e in args.experiments.split(",") if e.strip()}

    t_all = time.time()
    data_dir = args.data or {"taobao": "data/taobao", "synthetic": "data/synthetic_ids"}[args.source]
    data = load(args.source, data_dir)
    label = data.label()
    out = record.results_dir(args.results, data.synthetic)
    path = os.path.join(out, "freshness.jsonl")
    enc = encode(data)
    del data
    hist = ClickHistory(enc)
    spec = RankingSpec(enc, hist)
    device = get_device(args.device)
    test = F.build_test(spec, enc, hist, cfg.test_rows)
    load_s = time.time() - t_all
    print(f"{label}: device {device}, test rows {test.n:,}, setup {load_s:.0f}s")

    run_info = {"device": str(device), "quick": bool(args.quick), "model": F.MODEL,
                "features": spec.field_names, "test_features": "frozen at end of the last training day",
                "config": {**{k: v for k, v in vars(cfg).items() if k != "overrides"},
                           "batch_size": cfg.overrides.get("batch_size", 4096)}}

    def emit(row):
        record.write(path, {**row, **run_info}, dataset=label)

    stale_rows, update_rows = [], []
    for seed in sorted(set(stale_seeds) | set(seeds)):
        sampler = F.DaySampler(enc, seed, cfg.train_rows, cfg.val_rows)
        print(f"seed {seed}: {sampler.train_rows:,} train rows per day, {sampler.val_per_day:,} val per day")
        if "staleness" in exps and seed in stale_seeds:
            stale_rows += F.staleness_curve(spec, enc, hist, sampler, test, cfg, device, seed, emit)
        if "update" in exps and seed in seeds:
            update_rows += F.update_strategies(spec, enc, hist, sampler, test, cfg, device, seed, emit)

    for row in F.summarise(stale_rows, update_rows):
        emit({**row, "total_wall_seconds": time.time() - t_all})
        if row["experiment"] == "update_summary":
            g, c = row["warm_gap_recovered_ne"], row["compute_ratio_rows"]
            print(f"warm start recovers {100 * g['mean']:.0f}% of the NE freshness gap "
                  f"at {c['mean']:.1f}x fewer training rows (n={g['n']} seeds)")
    print(f"wrote {path} in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
