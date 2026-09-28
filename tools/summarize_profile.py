#!/usr/bin/env python
"""Turn the CSV that `nsys stats` emits into a readable markdown report.

An Nsight Systems capture is a binary file that only opens in the Nsight
Systems GUI. That is fine for looking at a timeline and useless for everything
else. It cannot be diffed, it cannot be read in a pull request, it cannot be
pasted into a design document, and it does not survive being emailed to somebody
who has not installed the GUI. `nsys stats` exists for that reason. It runs
canned reports over a capture and writes them as CSV.

This script is the last step. It reads those CSV files and writes the same
markdown table style the rest of this project reports in, so a profile becomes
an artifact a person can read rather than a blob somebody has to open a tool to
look at.

It also does one thing that is specific to this workload. `docs/INFERENCE.md`
predicts that this model is memory bandwidth bound on the embedding gather
rather than compute bound on the multilayer perceptron, and the evidence for or
against that is how the GPU time splits between gather kernels and matrix
multiply kernels. So the kernel summary is bucketed by what each kernel is,
alongside the usual per kernel table, and the bucket table is the one that
answers the question the document asked.

Usage.

    nsys stats --report cuda_gpu_kern_sum --format csv --output . profile.nsys-rep
    python tools/summarize_profile.py profile_cuda_gpu_kern_sum.csv --output profile.md

    # or point it at a directory and let it find every report in there
    python tools/summarize_profile.py results/profiles/run1 --output profile.md

Every number this produces is a GPU number and belongs to the GPU that produced
it, so the report carries a hardware label and refuses to print one that says
unknown without saying that it is unknown.

The CSV parsing here has been exercised against files written to the column
layout `nsys stats` documents. It has not been run against output from a real
`nsys` binary, because this project has no NVIDIA GPU to capture on. See
`docs/BENCHMARK_AUTOMATION.md` for what is executed and what is not.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Report identification
# ---------------------------------------------------------------------------

# nsys names its CSV files <base>_<report>.csv. These are the reports this
# script knows how to title. An unrecognized report is still rendered, with its
# file name as the heading, because a table nobody titled is better than a table
# nobody sees.
REPORT_TITLES = {
    "cuda_gpu_kern_sum": "CUDA kernels, by total GPU time",
    "cuda_gpu_mem_time_sum": "Memory operations, by total time",
    "cuda_gpu_mem_size_sum": "Memory operations, by bytes moved",
    "cuda_api_sum": "CUDA API calls, host side",
    "cuda_gpu_trace": "CUDA trace",
    "nvtx_sum": "NVTX ranges",
    "nvtx_pushpop_sum": "NVTX push and pop ranges",
    "nvtx_gpu_proj_sum": "NVTX ranges projected onto the GPU",
    "osrt_sum": "Operating system runtime calls",
    "openmp_sum": "OpenMP regions",
}

# The column that names the thing each row is about, in order of preference.
NAME_COLUMNS = ("Name", "Range", "Operation", "Kernel Name", "Function")

# Columns that hold a duration. The unit is in the header, which is where the
# unit conversion below reads it from.
_TIME_COLUMN = re.compile(r"^(Total Time|Time|Avg|Med|Min|Max|StdDev|Duration)\s*\((\w+)\)$")
_PERCENT_COLUMN = re.compile(r"^(Time|Total Time)\s*\(%\)$")

_TO_MS = {
    "ns": 1e-6,
    "us": 1e-3,
    "µs": 1e-3,
    "ms": 1.0,
    "s": 1e3,
    "sec": 1e3,
}


# ---------------------------------------------------------------------------
# Kernel classification
# ---------------------------------------------------------------------------

# Buckets in priority order. The first pattern that matches a kernel name wins,
# so the more specific patterns come first. These are substring matches against
# the lowercased kernel name and they are heuristics, not a ground truth. A
# kernel that lands in the wrong bucket is a mislabelled row and not a wrong
# measurement, and the per kernel table below always shows the raw names so a
# reader can check.
KERNEL_BUCKETS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "embedding gather",
        (
            "embedding",
            "gather",
            "index_select",
            "indexselect",
            "take",
            "sparse",
            "lookup",
        ),
    ),
    (
        "matrix multiply",
        (
            "gemm",
            "cutlass",
            "matmul",
            "tensorop",
            "implicit_gemm",
            "dot_kernel",
            "fully_connected",
            "myelin",
        ),
    ),
    (
        "reduction and normalization",
        ("reduce", "softmax", "layernorm", "batchnorm", "norm_kernel", "sum_kernel"),
    ),
    (
        "elementwise and activation",
        (
            "elementwise",
            "pointwise",
            "vectorized_",
            "relu",
            "sigmoid",
            "tanh",
            "add_kernel",
            "mul_kernel",
            "bias",
            "activation",
            "cast",
        ),
    ),
    (
        "data movement",
        (
            "memcpy",
            "memset",
            "transpose",
            "permute",
            "reformat",
            "concat",
            "catarray",
            "cat_",
            "copy",
            "clone",
            "contiguous",
        ),
    ),
)

OTHER_BUCKET = "other"

# What each bucket says about the prediction in docs/INFERENCE.md.
BUCKET_MEANING = {
    "embedding gather": (
        "scattered reads into the embedding tables. This is the part the "
        "bandwidth bound prediction says should dominate"
    ),
    "matrix multiply": (
        "the multilayer perceptron. This is the part a compute bound model "
        "would be dominated by and the part that fp16 and int8 speed up"
    ),
    "reduction and normalization": "reductions across the feature or batch axis",
    "elementwise and activation": (
        "activations, bias adds, and casts. Cheap in arithmetic and paid for in "
        "bandwidth, so a large share here is itself a bandwidth signal"
    ),
    "data movement": (
        "copies, transposes, and layout changes. Pure bandwidth with no "
        "arithmetic attached"
    ),
    OTHER_BUCKET: "kernels this script could not classify from their names",
}


def classify_kernel(name: str) -> str:
    """Put a kernel name into one of the buckets above."""
    lowered = name.lower()
    for bucket, patterns in KERNEL_BUCKETS:
        for pattern in patterns:
            if pattern in lowered:
                return bucket
    return OTHER_BUCKET


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


@dataclass
class Table:
    """One nsys report, loaded from CSV."""

    path: str
    report: str
    headers: List[str] = field(default_factory=list)
    rows: List[Dict[str, str]] = field(default_factory=list)

    @property
    def title(self) -> str:
        return REPORT_TITLES.get(self.report, self.report.replace("_", " "))

    @property
    def name_column(self) -> Optional[str]:
        for candidate in NAME_COLUMNS:
            if candidate in self.headers:
                return candidate
        # Fall back to the last column, which is where nsys puts the name in
        # every summary report it ships.
        return self.headers[-1] if self.headers else None

    @property
    def percent_column(self) -> Optional[str]:
        for header in self.headers:
            if _PERCENT_COLUMN.match(header):
                return header
        return None

    @property
    def total_time_column(self) -> Optional[str]:
        for header in self.headers:
            match = _TIME_COLUMN.match(header)
            if match and match.group(1) in ("Total Time", "Duration", "Time"):
                return header
        return None


def report_name_for(path: str) -> str:
    """Work out which nsys report a CSV file holds, from its file name.

    nsys writes `<base>_<report>.csv`, so the report is the longest known
    report name that the file name ends with. Falling back to the stem keeps an
    unknown or renamed file readable rather than dropping it.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    for report in sorted(REPORT_TITLES, key=len, reverse=True):
        if stem.endswith(report):
            return report
    return stem


def load_table(path: str) -> Table:
    """Read one nsys CSV report."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        text = handle.read()
    return parse_table(text, path=path, report=report_name_for(path))


def parse_table(text: str, path: str = "<stdin>", report: str = "report") -> Table:
    """Parse nsys CSV text into a Table.

    nsys sometimes prefixes a report with blank lines and a title line before
    the header, depending on the version and on whether the output went to a
    file or to stdout, so the header row is found rather than assumed to be
    first. The header is the first row that has more than one field and holds
    one of the columns every summary report has.
    """
    lines = text.splitlines()
    start = 0
    for index, line in enumerate(lines):
        if "," not in line:
            continue
        fields = next(csv.reader([line]))
        if len(fields) < 2:
            continue
        stripped = [f.strip() for f in fields]
        if any(_PERCENT_COLUMN.match(f) or _TIME_COLUMN.match(f) for f in stripped) or any(
            f in NAME_COLUMNS for f in stripped
        ):
            start = index
            break

    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    headers = [h.strip() for h in (reader.fieldnames or [])]
    rows: List[Dict[str, str]] = []
    for raw in reader:
        row = {
            (key.strip() if key else ""): (value.strip() if isinstance(value, str) else "")
            for key, value in raw.items()
            if key is not None
        }
        if not any(row.values()):
            continue
        rows.append(row)
    return Table(path=path, report=report, headers=headers, rows=rows)


def discover(paths: Sequence[str]) -> List[str]:
    """Expand directories into the CSV files inside them."""
    found: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            for entry in sorted(os.listdir(path)):
                if entry.lower().endswith(".csv"):
                    found.append(os.path.join(path, entry))
        elif os.path.isfile(path):
            found.append(path)
        else:
            raise FileNotFoundError(f"no file or directory at {path}")
    return found


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def to_float(value: str) -> Optional[float]:
    """Parse a number out of an nsys cell, tolerating thousands separators."""
    if value is None:
        return None
    cleaned = value.replace(",", "").replace("%", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def column_ms(header: str, value: str) -> Optional[float]:
    """Convert a duration cell to milliseconds using the unit in its header."""
    match = _TIME_COLUMN.match(header)
    number = to_float(value)
    if match is None or number is None:
        return None
    factor = _TO_MS.get(match.group(2).lower())
    if factor is None:
        return None
    return number * factor


def fmt_ms(value: Optional[float]) -> str:
    if value is None:
        return "not available"
    if value >= 1000:
        return f"{value / 1000:,.3f} s"
    if value < 0.001:
        return f"{value * 1000:,.3f} us"
    return f"{value:,.3f} ms"


def fmt_pct(value: Optional[float]) -> str:
    return "not available" if value is None else f"{value:.1f}%"


def fmt_int(value: Optional[float]) -> str:
    return "not available" if value is None else f"{value:,.0f}"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def escape(cell: str) -> str:
    """Make a cell safe to put inside a markdown table."""
    return cell.replace("|", "\\|").replace("\n", " ")


def render_table(table: Table, top: int) -> List[str]:
    """Render one nsys report as a markdown table, truncated to the top rows."""
    lines: List[str] = []
    lines.append(f"### {table.title}")
    lines.append("")
    lines.append(f"Source `{os.path.basename(table.path)}`.")
    lines.append("")

    if not table.rows:
        lines.append("This report is empty. Nothing of this kind was captured.")
        lines.append("")
        return lines

    percent = table.percent_column
    rows = list(table.rows)
    if percent:
        rows.sort(key=lambda row: to_float(row.get(percent, "")) or 0.0, reverse=True)

    shown = rows[:top]
    hidden = rows[top:]

    headers = [h for h in table.headers if h]
    lines.append("| " + " | ".join(escape(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in shown:
        lines.append("| " + " | ".join(escape(row.get(h, "")) for h in headers) + " |")
    lines.append("")

    if hidden:
        hidden_pct = 0.0
        if percent:
            hidden_pct = sum(to_float(row.get(percent, "")) or 0.0 for row in hidden)
        lines.append(
            f"{len(hidden)} further row(s) are not shown, together accounting for "
            f"{hidden_pct:.1f} percent of the time in this report. They are in the "
            f"CSV. Raise --top to see them."
        )
        lines.append("")

    return lines


def bucket_summary(table: Table) -> List[str]:
    """Render the kernel time split by what each kernel is.

    This is the table the prediction in docs/INFERENCE.md is decided by, so it
    comes before the per kernel table in the report and it is the one worth
    reading first.
    """
    name_column = table.name_column
    total_column = table.total_time_column
    percent_column = table.percent_column
    if not name_column:
        return []

    totals: Dict[str, float] = {}
    percents: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    grand_ms = 0.0

    for row in table.rows:
        bucket = classify_kernel(row.get(name_column, ""))
        counts[bucket] = counts.get(bucket, 0) + 1
        if total_column:
            value = column_ms(total_column, row.get(total_column, ""))
            if value is not None:
                totals[bucket] = totals.get(bucket, 0.0) + value
                grand_ms += value
        if percent_column:
            value = to_float(row.get(percent_column, ""))
            if value is not None:
                percents[bucket] = percents.get(bucket, 0.0) + value

    if not counts:
        return []

    lines: List[str] = []
    lines.append("### Where the GPU time went")
    lines.append("")
    lines.append(
        "Kernels are bucketed by what their names say they do. The buckets are "
        "heuristics over kernel names and not a ground truth, so a "
        "misclassified kernel is a mislabelled row rather than a wrong "
        "measurement, and the full per kernel table below carries the raw names."
    )
    lines.append("")
    lines.append("| Bucket | Kernels | Total time | Share | What it is |")
    lines.append("| --- | --- | --- | --- | --- |")

    ordered = sorted(
        counts, key=lambda bucket: percents.get(bucket, totals.get(bucket, 0.0)), reverse=True
    )
    for bucket in ordered:
        share = percents.get(bucket)
        if share is None and grand_ms > 0 and bucket in totals:
            share = 100.0 * totals[bucket] / grand_ms
        lines.append(
            f"| {bucket} | {counts[bucket]} | {fmt_ms(totals.get(bucket))} "
            f"| {fmt_pct(share)} | {BUCKET_MEANING.get(bucket, '')} |"
        )
    lines.append("")

    gather = percents.get("embedding gather", 0.0)
    gemm = percents.get("matrix multiply", 0.0)
    if gather or gemm:
        lines.append(
            f"The gather bucket holds {gather:.1f} percent of the GPU time and the "
            f"matrix multiply bucket holds {gemm:.1f} percent. `docs/INFERENCE.md` "
            f"predicts the first number is the large one, because nearly all the "
            f"parameters of a DLRM style ranker sit in embedding tables that are "
            f"read with no locality while the multilayer perceptron after them is "
            f"small. A profile where the second number is the large one refutes "
            f"that prediction and is the more interesting result."
        )
        lines.append("")

    return lines


def render_report(
    tables: Sequence[Table],
    hardware: str,
    top: int,
    title: str,
    source: str,
) -> str:
    """Render the whole markdown report."""
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(
        "This report was generated from the CSV that `nsys stats` writes out of "
        "an Nsight Systems capture. The capture itself opens in the Nsight "
        "Systems GUI and holds the full timeline. This is the readable summary "
        "of it."
    )
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Hardware | {hardware} |")
    lines.append(f"| Capture | {source} |")
    lines.append(f"| Reports | {len(tables)} |")
    lines.append(f"| Rows shown per report | {top} |")
    lines.append("")

    if hardware.strip().lower() in ("", "unknown", "not available"):
        lines.append(
            "The hardware label is unknown. Every number below is a GPU number "
            "and a GPU number without its device is not a result, so fill the "
            "label in with --hardware before this report is quoted anywhere."
        )
        lines.append("")

    kernel_tables = [t for t in tables if t.report.startswith("cuda_gpu_kern")]
    for table in kernel_tables:
        lines.extend(bucket_summary(table))

    lines.append("## Reports")
    lines.append("")
    for table in tables:
        lines.extend(render_table(table, top))

    lines.append("## Reading this")
    lines.append("")
    lines.append(
        "Nsight Systems answers where the time goes across the whole process. "
        "It is a timeline, so it shows the gaps as well as the work, and on a "
        "small model the gaps are usually the finding. Nsight Compute answers "
        "why one kernel is slow, by replaying that kernel and reading the "
        "hardware counters. Reach for Systems first and for Compute only once "
        "Systems has named the kernel worth asking about. "
        "`docs/BENCHMARK_AUTOMATION.md` covers both in full."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="summarize_profile.py",
        description=(
            "Turn the CSV from nsys stats into a markdown report in the same "
            "style the rest of ClickRank reports in."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="CSV files from nsys stats, or directories holding them.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Where to write the markdown. Default is stdout.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Rows to show per report before truncating. Default %(default)s",
    )
    parser.add_argument(
        "--hardware",
        default="unknown",
        help=(
            "Hardware label stamped on the report. A GPU number without its "
            "device is not a result, so this should always be filled in."
        ),
    )
    parser.add_argument(
        "--title",
        default="Nsight Systems Profile Summary",
        help="Title for the report. Default %(default)s",
    )
    parser.add_argument(
        "--capture",
        default="not recorded",
        help="Path of the nsys-rep capture these reports came from.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    try:
        paths = discover(args.inputs)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not paths:
        print(
            "no csv files were found. Produce them with nsys stats --format csv "
            "--output <dir> <capture>.nsys-rep",
            file=sys.stderr,
        )
        return 1

    tables: List[Table] = []
    for path in paths:
        try:
            table = load_table(path)
        except OSError as exc:
            print(f"could not read {path}. {exc}", file=sys.stderr)
            return 1
        if not table.headers:
            print(f"skipping {path}, no header row was found in it", file=sys.stderr)
            continue
        tables.append(table)

    if not tables:
        print("none of the inputs parsed as an nsys csv report", file=sys.stderr)
        return 1

    markdown = render_report(
        tables,
        hardware=args.hardware,
        top=args.top,
        title=args.title,
        source=args.capture,
    )

    if args.output:
        directory = os.path.dirname(os.path.abspath(args.output))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(markdown)
        print(f"wrote {args.output} from {len(tables)} report(s)")
    else:
        print(markdown)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
