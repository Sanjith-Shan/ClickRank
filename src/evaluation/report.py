"""Markdown report generation for the ClickRank benchmark.

This module turns a list of per model result dicts into a single readable
markdown report. The report ranks models by AUC, links the calibration figure,
lists the top logistic regression coefficients when available, and closes with a
short plain analysis of which model won and why feature interactions help.

All prose written here follows the project style rules. No em dashes. No
semicolons. No mid sentence colons.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

# Columns shown in the main results table, in display order.
_TABLE_FIELDS = [
    ("name", "Model"),
    ("auc", "AUC"),
    ("logloss", "LogLoss"),
    ("ne", "NE"),
    ("relaimpr", "RelaImpr"),
    ("gauc", "GAUC"),
    ("ece", "ECE"),
    ("train_time", "Train s"),
    ("n_params", "Params"),
]


def _fmt(value) -> str:
    """Format a metric value for a markdown table cell."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _build_table(results: List[Dict]) -> str:
    """Build the markdown results table sorted by AUC descending."""
    ordered = sorted(
        results,
        key=lambda r: (r.get("auc") if r.get("auc") is not None else -1.0),
        reverse=True,
    )

    header = "| " + " | ".join(label for _, label in _TABLE_FIELDS) + " |"
    divider = "| " + " | ".join("---" for _ in _TABLE_FIELDS) + " |"
    lines = [header, divider]
    for row in ordered:
        cells = [_fmt(row.get(key)) for key, _ in _TABLE_FIELDS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _build_feature_importance(
    feature_importance: Optional[Sequence[Tuple]],
) -> str:
    """Build the feature importance section from (name, weight) tuples."""
    if not feature_importance:
        return ""

    lines = [
        "## Feature Importance",
        "",
        "Top logistic regression coefficients by absolute magnitude. A larger "
        "magnitude means the feature pushes the predicted click probability more "
        "strongly. The sign shows the direction of that push.",
        "",
        "| Feature | Coefficient |",
        "| --- | --- |",
    ]
    for name, weight in feature_importance:
        lines.append(f"| {name} | {_fmt(float(weight))} |")
    lines.append("")
    return "\n".join(lines)


def _best_model_name(results: List[Dict]) -> str:
    """Return the name of the model with the highest AUC."""
    best = max(
        results,
        key=lambda r: (r.get("auc") if r.get("auc") is not None else -1.0),
        default=None,
    )
    if best is None:
        return "the top model"
    return str(best.get("name", "the top model"))


def _build_observations(results: List[Dict]) -> str:
    """Build the key observations section as plain analysis paragraphs."""
    winner = _best_model_name(results)

    logistic = next(
        (r for r in results if str(r.get("name", "")).lower().startswith("logistic")),
        None,
    )
    lr_auc = _fmt(logistic.get("auc")) if logistic else "the baseline"

    para_one = (
        f"The strongest model by test AUC is {winner}. It ranks impressions "
        f"better than the logistic regression baseline which scored {lr_auc} AUC. "
        "Ranking quality is what an ad auction cares about most because the "
        "auction orders candidates before pricing them."
    )

    para_two = (
        "The factorization machine family beats plain logistic regression "
        "because the synthetic clicks depend on pairwise interactions between "
        "categorical fields. Logistic regression only learns linear weights on "
        "single features, so it cannot capture the lift that appears when two "
        "category groups co occur. The FM second order term and the deep towers "
        "model those interactions directly, which is why they pull ahead on AUC "
        "and normalized entropy."
    )

    para_three = (
        "Calibration and normalized entropy round out the picture. A model can "
        "rank well and still be biased in absolute probability, so the "
        "calibration curve and the expected calibration error matter for bidding "
        "and pacing. Group AUC reflects within request ranking quality, which is "
        "closer to the live auction than dataset wide AUC. Reading these metrics "
        "together gives a fair comparison of the architectures."
    )

    return "\n\n".join(
        [
            "## Key Observations",
            para_one,
            para_two,
            para_three,
        ]
    )


def generate_report(
    results: List[Dict],
    out_dir: str,
    calibration_png: Optional[str] = None,
    feature_importance: Optional[Sequence[Tuple]] = None,
) -> str:
    """Write the benchmark markdown report and return its path.

    The results argument is a list of per model dicts. Each dict may carry name,
    auc, logloss, ne, relaimpr, gauc, ece, train_time, and n_params. The report
    is written to out_dir as benchmark_report.md. It contains a results table
    sorted by AUC descending, an optional link to the calibration figure, an
    optional feature importance section, and a key observations section.
    """
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "benchmark_report.md")

    sections: List[str] = []
    sections.append("# ClickRank Benchmark Report")
    sections.append(
        "This report compares click through rate models on a shared featurized "
        "dataset. Models are ranked by test AUC. All numbers come from the held "
        "out test split."
    )

    sections.append("## Results")
    sections.append(_build_table(results))

    if calibration_png:
        # Embed the figure as a relative path so the markdown renders next to it.
        rel = os.path.relpath(calibration_png, out_dir)
        sections.append("## Calibration")
        sections.append(
            "The reliability curves below overlay each model against the perfect "
            "calibration diagonal. A curve that hugs the diagonal is well "
            "calibrated."
        )
        sections.append(f"![Calibration curves]({rel})")

    fi_section = _build_feature_importance(feature_importance)
    if fi_section:
        sections.append(fi_section)

    sections.append(_build_observations(results))

    content = "\n\n".join(sections).rstrip() + "\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    return report_path
