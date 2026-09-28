"""TorchRec DLRM on one GPU: the sharding plan, and scoring sharded against unsharded.

Needs the DLRM weights run_benchmark.py wrote (--models dlrm), the same data file,
and Linux with a CUDA GPU and TorchRec. Rebuilds the identical held out rows,
then

1. loads the trained DLRM unsharded and scores the test rows (AUC must match the
   benchmark's),
2. asks TorchRec's EmbeddingShardingPlanner for a plan on a one GPU topology and
   wraps the model in DistributedModelParallel, which swaps every embedding table
   for fbgemm's fused table batched embedding kernel, then loads the same weights
   and scores the same rows,
3. times both, plus fp16 autocast on the unsharded model, per batch at several
   batch sizes with CUDA events.

With one GPU the planner has one device to place every table on, so the plan is
trivial. What changes is the kernel, and that is what the timing measures.

Writes <output>/torchrec.jsonl, one row per measurement, each with the machine.

Usage:
    python scripts/run_torchrec.py --data-path data/criteo.csv --sample-size 2000000 \
        --output results/gpu_a100_dlrm
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import build_datasets  # noqa: E402
from src.data.split import temporal_split  # noqa: E402
from src.evaluation.metrics import compute_auc, compute_logloss  # noqa: E402
from src.models.dlrm import DLRMModule, to_kjt  # noqa: E402
from src.train.config import SEED, get_config  # noqa: E402
from src.train.trainer import _build_cat_tensor, set_seed  # noqa: E402


def machine() -> dict:
    import torchrec

    info = {
        "gpu": torch.cuda.get_device_name(0),
        "driver_cuda": torch.version.cuda,
        "torch": torch.__version__,
        "torchrec": getattr(torchrec, "__version__", "?"),
        "python": platform.python_version(),
        "host": platform.node(),
    }
    try:
        import fbgemm_gpu

        info["fbgemm_gpu"] = getattr(fbgemm_gpu, "__version__", "?")
    except Exception:  # noqa: BLE001
        pass
    return info


def write(path: str, row: dict) -> None:
    row = {"when": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "dataset": "criteo", **row}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row)[:300])


@torch.no_grad()
def score(fn, num: torch.Tensor, cat: torch.Tensor, bs: int = 16384) -> np.ndarray:
    out = []
    for i in range(0, len(num), bs):
        out.append(torch.sigmoid(fn(num[i:i + bs], cat[i:i + bs]).float()).cpu())
    return torch.cat(out).numpy().astype(np.float64)


@torch.no_grad()
def time_batches(fn, num: torch.Tensor, cat: torch.Tensor, bs: int, iters: int = 100, warmup: int = 20) -> dict:
    n = len(num)
    starts = [(i * bs) % max(n - bs, 1) for i in range(iters + warmup)]
    for s in starts[:warmup]:
        fn(num[s:s + bs], cat[s:s + bs])
    torch.cuda.synchronize()
    ms = []
    for s in starts[warmup:]:
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn(num[s:s + bs], cat[s:s + bs])
        b.record()
        torch.cuda.synchronize()
        ms.append(a.elapsed_time(b))
    ms = np.asarray(ms)
    p50 = float(np.median(ms))
    return {"batch_size": bs, "iters": iters, "p50_ms": p50, "p99_ms": float(np.percentile(ms, 99)),
            "mean_ms": float(ms.mean()), "predictions_per_sec": bs / (p50 / 1000.0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-path", default="data/criteo.csv")
    ap.add_argument("--sample-size", type=int, default=2_000_000)
    ap.add_argument("--output", default="results/gpu_a100_dlrm")
    ap.add_argument("--batch-sizes", default="256,1024,4096,16384")
    args = ap.parse_args()

    import torch.distributed as dist
    from torchrec.distributed.model_parallel import DistributedModelParallel
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder

    from src.data.loader import load_raw

    set_seed(SEED)
    dev = torch.device("cuda")
    out = os.path.join(args.output, "torchrec.jsonl")
    mach = machine()

    df = load_raw(args.data_path, sample_size=args.sample_size)
    train_df, val_df, test_df = temporal_split(df)
    _, _, test_ds, meta = build_datasets(train_df, val_df, test_df)
    y = np.asarray(test_ds.label, dtype=np.float64).reshape(-1)
    num = torch.as_tensor(np.asarray(test_ds.numerical, dtype=np.float32), device=dev)
    # The same (categorical, crosses) field tensor the shared trainer feeds every model.
    cat = torch.as_tensor(_build_cat_tensor(test_ds), dtype=torch.long, device=dev)

    cfg = get_config("dlrm")
    state = torch.load(os.path.join(args.output, "DLRM.pt"), map_location="cpu")

    # 1. Unsharded, as trained.
    plain = DLRMModule(meta, cfg["embed_dim"], cfg["bottom"], cfg["top"])
    plain.load_state_dict(state)
    plain = plain.to(dev).eval()
    p_plain = score(plain, num, cat)
    write(out, {"what": "unsharded_quality", "auc": compute_auc(y, p_plain), "logloss": compute_logloss(y, p_plain),
                "test_rows": int(len(y)), "machine": mach})

    # 2. Planner and DistributedModelParallel on a one GPU topology.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    dist.init_process_group("nccl", rank=0, world_size=1)
    sharded_src = DLRMModule(meta, cfg["embed_dim"], cfg["bottom"], cfg["top"])
    planner = EmbeddingShardingPlanner(topology=Topology(world_size=1, compute_device="cuda"))
    sharders = [EmbeddingBagCollectionSharder()]
    plan = planner.collective_plan(sharded_src.dlrm, sharders, dist.GroupMember.WORLD)
    per_table = {}
    for mod_path, mod_plan in plan.plan.items():
        for table, p in mod_plan.items():
            per_table[table] = {"sharding_type": p.sharding_type, "compute_kernel": p.compute_kernel,
                                "ranks": list(p.ranks or [])}
    kinds = sorted({(v["sharding_type"], v["compute_kernel"]) for v in per_table.values()})
    write(out, {"what": "sharding_plan", "tables": len(per_table), "distinct_choices": kinds,
                "per_table": per_table, "world_size": 1, "machine": mach})

    dmp = DistributedModelParallel(module=sharded_src.dlrm, device=dev, plan=plan, sharders=sharders)
    loaded = True
    try:
        inner = {k[len("dlrm."):]: v for k, v in state.items() if k.startswith("dlrm.")}
        dmp.load_state_dict(inner)
    except Exception as exc:  # noqa: BLE001 report, never silently continue
        loaded = False
        print(f"could not load trained weights into the sharded model: {exc}")
    dmp.eval()
    keys = plain.keys

    def sharded_fn(n, c):
        return dmp(n, to_kjt(c, keys)).squeeze(-1)

    if loaded:
        p_sh = score(sharded_fn, num, cat)
        write(out, {"what": "sharded_quality", "auc": compute_auc(y, p_sh), "logloss": compute_logloss(y, p_sh),
                    "max_abs_diff_vs_unsharded": float(np.abs(p_sh - p_plain).max()),
                    "test_rows": int(len(y)), "machine": mach})
    else:
        write(out, {"what": "sharded_quality", "error": "weights did not load, timing only", "machine": mach})

    # 3. Timing on the same rows.
    def fp16_fn(n, c):
        with torch.autocast("cuda", dtype=torch.float16):
            return plain(n, c)

    for bs in [int(b) for b in args.batch_sizes.split(",")]:
        for name, fn in (("unsharded_fp32", plain), ("unsharded_fp16_autocast", fp16_fn), ("sharded_fbgemm_fp32", sharded_fn)):
            write(out, {"what": "latency", "variant": name, **time_batches(fn, num, cat, bs), "machine": mach})

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
