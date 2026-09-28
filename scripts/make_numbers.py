"""Render NUMBERS.md for the retrieval stage from the records under results/retrieval.

Every figure in the generated sections is read from a results file, and each table names
the file it came from. Rows labelled SYNTHETIC are skipped, so a synthetic run can never
reach NUMBERS.md. Hand written prose lives between the markers in NUMBERS.md and is kept.

Usage:
    python scripts/make_numbers.py
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "retrieval")
OUT = os.path.join(ROOT, "NUMBERS.md")
BEGIN, END = "<!-- generated:begin -->", "<!-- generated:end -->"


def rows(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        r = json.loads(line)
        if r.get("dataset") != "taobao":
            continue
        out.append(r)
    return out


def latest(rs, key=lambda r: None):
    """Last row for each key, since a rerun appends rather than overwrites."""
    seen = {}
    for r in rs:
        seen[key(r)] = r
    return list(seen.values())


def pct(x):
    return f"{100 * x:.1f}%"


def machine(r):
    m, ld = r.get("machine", {}), r.get("load", {})
    return f"{m.get('cpu', '?')}, load {ld.get('load_1m', '?')}"


def section_data(lines):
    rs = rows("data_summary.jsonl")
    if not rs:
        return
    s = rs[-1]["summary"]
    lines += [
        "## Data",
        "",
        "From `results/retrieval/data_summary.jsonl`. Counts match the dataset card.",
        "",
        "| Impressions | Clicks | CTR | Users | Ads in corpus | Train (days 1 to 7) | Test (day 8) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| {s['impressions']:,} | {s['clicks']:,} | {pct(s['ctr'])} | {s['users_in_log']:,} | "
        f"{s['ads_in_corpus']:,} | {s['train_impressions']:,} | {s['test_impressions']:,} |",
        "",
    ]


def section_retrieval(lines):
    rs = rows("hit_rate.jsonl")
    if not rs:
        return
    r = rs[-1]
    ks = ("50", "100", "500")
    lines += [
        "## Retrieval hit rate on the test day",
        "",
        f"From `results/retrieval/hit_rate.jsonl`. The share of the {r['click_pairs']:,} test day "
        f"(user, clicked ad) pairs whose ad is in that user's top K of {r['corpus_ads']:,} ads, "
        "by exact search on the tower embeddings. Popularity returns the K most clicked ads of the "
        "training days to every user.",
        "",
        "| K | Two tower | Popularity | Chance | Candidates scored, fewer than all ads |",
        "| --- | --- | --- | --- | --- |",
    ]
    for k in ks:
        lines.append(
            f"| {k} | {pct(r['hit_rate'][k]['rate'])} | {pct(r['hit_rate_popularity'][k]['rate'])} | "
            f"{100 * int(k) / r['corpus_ads']:.3f}% | {r['corpus_ads'] / int(k):,.0f}x |"
        )
    san = rows("two_tower_sanity.jsonl")
    if san:
        lines += ["", f"Sanity check, `results/retrieval/two_tower_sanity.jsonl`. A user's clicked ad scores "
                      f"above a random ad for {pct(san[-1]['clicked_above_random'])} of test pairs.", ""]
    else:
        lines.append("")


def section_sweep(lines):
    rs = latest(rows("index_sweep.jsonl"), key=lambda r: r["index"])
    if not rs:
        return
    lines += [
        "## FAISS index sweep",
        "",
        "From `results/retrieval/index_sweep.jsonl`. Recall is against exact search (`IndexFlatIP`) "
        "on the same embeddings over 20,000 test day users. Latency is one query at a time on one "
        "thread. The sweep ran beside other training jobs, so the load column matters, and the "
        "clean latencies are the two stage ones below.",
        "",
        "| Index | Size | Recall@50 | Recall@100 | Recall@500 | Hit rate@100 | p50 @100, 1 thread | Machine |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rs:
        rc = r["recall_vs_exact"]
        lat = r["latency_1thread"]["100"]["p50_us"]
        lines.append(
            f"| {r['index']} | {r['bytes'] / 1e6:.1f} MB | {rc['50']:.3f} | {rc['100']:.3f} | {rc['500']:.3f} | "
            f"{pct(r['hit_rate']['100'])} | {lat / 1000:.2f} ms | {machine(r)} |"
        )
    lines.append("")


def section_rankers(lines):
    rs = latest(rows("ranker_eval.jsonl"), key=lambda r: r["model"])
    if not rs:
        return
    lines += [
        "## Rankers on the test day",
        "",
        "From `results/retrieval/ranker_eval.jsonl`. Every test day impression, scored exhaustively.",
        "",
        "| Model | AUC | GAUC by user | NE |",
        "| --- | --- | --- | --- |",
    ]
    for r in rs:
        g = r.get("gauc_by_user", r.get("gauc"))
        g = f"{g:.4f}" if isinstance(g, float) else "n/a"
        lines.append(f"| {r['model']} | {r['auc']:.4f} | {g} | {r['ne']:.4f} |")
    lines.append("")


def section_two_stage(lines):
    rs = latest(rows("two_stage_quality.jsonl"), key=lambda r: r["ranker"])
    if rs:
        lines += [
            "## Two stage against exhaustive ranking",
            "",
            "From `results/retrieval/two_stage_quality.jsonl`. Exhaustive scores every test day "
            "impression. Two stage at K keeps the ranker's score only when the ad is in the user's "
            "top K retrieved. A dropped ad ranks below every retrieved one for AUC and GAUC, and for "
            "NE it gets the ranker's mean prediction over dropped impressions.",
            "",
            "| Ranker | Setting | AUC | GAUC by user | NE |",
            "| --- | --- | --- | --- | --- |",
        ]
        for r in rs:
            for name, m in r["results"].items():
                lines.append(f"| {r['ranker']} | {name} | {m['auc']:.4f} | {m['gauc_by_user']:.4f} | {m['ne']:.4f} |")
        lines += ["", f"Serving index: `{rs[0]['serving_index']}`.", ""]

    fr = latest(rows("final_rank.jsonl"), key=lambda r: r["ranker"])
    if fr:
        lines += [
            "### Where the clicked ad lands",
            "",
            "From `results/retrieval/final_rank.jsonl`. One test day click per sampled user. "
            "Exhaustive ranks the clicked ad among every ad in the corpus. Two stage ranks it among "
            "the K retrieved, and counts a miss when retrieval did not return it.",
            "",
            "| Ranker | Requests | Exhaustive median rank | Exhaustive top 50 | K | Retrieved | Two stage top 10 | Two stage top 50 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in fr:
            ex = r["exhaustive"]
            for k, t in r["two_stage"].items():
                lines.append(
                    f"| {r['ranker']} | {r['requests']} | {ex['median_rank']:,.0f} | {pct(ex['top50'])} | {k} | "
                    f"{pct(t['retrieved'])} | {pct(t['top10'])} | {pct(t['top50'])} |"
                )
        lines.append("")

    lat = rows("stage_latency.jsonl")
    if lat:
        two = latest([r for r in lat if r["mode"] == "two_stage"], key=lambda r: (r["ranker"], r["index"], r["k"]))
        ex = latest([r for r in lat if r["mode"] == "exhaustive"], key=lambda r: (r["ranker"], r["torch_threads"]))
        lines += [
            "### Latency per request",
            "",
            "From `results/retrieval/stage_latency.jsonl`. One torch thread and one FAISS thread for "
            "two stage. Exhaustive is shown on one thread and on every core, its best case.",
            "",
            "| Ranker | Mode | Index | K | User tower p50 | Search p50 | Rank p50 | Total p50 | Total p99 | Machine |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]

        def st(r, name, q="p50_us"):
            for key in r["stages"]:
                if name in key:
                    return f"{r['stages'][key][q] / 1000:.2f} ms"
            return ""

        for r in two:
            lines.append(
                f"| {r['ranker']} | two stage | {r['index']} | {r['k']} | {st(r, 'user')} | {st(r, 'search')} | "
                f"{st(r, 'rank')} | {st(r, 'total')} | {st(r, 'total', 'p99_us')} | {machine(r)} |"
            )
        for r in ex:
            lines.append(
                f"| {r['ranker']} | exhaustive, {r['torch_threads']} threads | none | {r['k']:,} | {st(r, 'user')} | | "
                f"{st(r, 'rank')} | {st(r, 'total')} | {st(r, 'total', 'p99_us')} | {machine(r)} |"
            )
        lines.append("")


def section_freshness(lines):
    rs = rows("freshness.jsonl")
    stale = latest([r for r in rs if r["experiment"] == "staleness_summary"], key=lambda r: r["train_end_day"])
    if not stale:
        stale = latest([r for r in rs if r["experiment"] == "staleness"], key=lambda r: r["train_end_day"])
    upd = [r for r in rs if r["experiment"] == "update_summary"]
    if not stale and not upd:
        return
    lines += ["## Freshness", "", "From `results/retrieval/freshness.jsonl`. DeepFM, test day 8.", ""]
    if stale:
        lines += ["| Trained on day | Days stale | AUC | NE | NE vs freshest |", "| --- | --- | --- | --- | --- |"]
        for r in sorted(stale, key=lambda r: r["train_end_day"]):
            auc = r.get("auc", r.get("auc_mean"))
            ne = r.get("ne", r.get("ne_mean"))
            rel = r.get("ne_rel_to_freshest", r.get("ne_rel_to_freshest_mean"))
            rel = f"{100 * rel:+.2f}%" if isinstance(rel, float) else ""
            lines.append(f"| {r['train_end_day']} | {r['staleness_days']} | {auc:.4f} | {ne:.4f} | {rel} |")
        lines.append("")
    if upd:
        u = upd[-1]
        lines += ["Update strategy, summary row:", "", "```", json.dumps({k: v for k, v in u.items()
                  if k not in ("machine", "load", "when")}, indent=1), "```", ""]


def main() -> None:
    lines = [BEGIN, ""]
    for fn in (section_data, section_retrieval, section_sweep, section_rankers, section_two_stage, section_freshness):
        fn(lines)
    lines.append(END)
    gen = "\n".join(lines)
    if os.path.exists(OUT):
        cur = open(OUT).read()
        if BEGIN in cur and END in cur:
            head, rest = cur.split(BEGIN, 1)
            tail = rest.split(END, 1)[1]
            open(OUT, "w").write(head + gen + tail)
            print(f"updated {OUT}")
            return
    open(OUT, "w").write("# Numbers\n\n" + gen + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
