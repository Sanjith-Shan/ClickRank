"""SparkSession construction and environment probing.

Spark needs a JVM. That is the one hard dependency this lane carries that the
rest of ClickRank does not, so every entry point probes for it first and
degrades with a clear message rather than throwing a stack trace. The helpers
here are the single place that knows how to answer whether Spark can run.

The session factory takes its configuration as parameters rather than baking in
laptop values. The defaults are tuned for a single machine with a dozen cores,
which is what a developer runs against, and the same function produces a cluster
session when the caller passes a real master URL and larger shuffle and memory
settings. Nothing in the pipeline reads a hardcoded config, so moving from
`local[*]` to YARN or Kubernetes is a change at the call site only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict, Optional

# Default shuffle partition count for a laptop run. Spark ships with 200, which
# is badly oversized for a few million rows on one machine and turns every
# shuffle into hundreds of tiny tasks whose scheduling overhead dominates the
# actual work. A small multiple of the core count is the right order here.
DEFAULT_SHUFFLE_PARTITIONS = 16

# Broadcast join threshold in bytes. The categorical vocabulary tables this
# pipeline joins against are small on synthetic data and can be large on real
# Criteo, so the threshold is a knob rather than a constant.
DEFAULT_BROADCAST_THRESHOLD = 32 * 1024 * 1024


def java_available() -> bool:
    """Return True when a usable JVM is on the path or under JAVA_HOME.

    Spark shells out to `java` at session start. We probe the same way it does
    so the caller can print a useful message instead of surfacing a Py4J error.
    """
    java_home = os.environ.get("JAVA_HOME")
    candidates = []
    if java_home:
        candidates.append(os.path.join(java_home, "bin", "java"))
    resolved = shutil.which("java")
    if resolved:
        candidates.append(resolved)

    for candidate in candidates:
        try:
            proc = subprocess.run(
                [candidate, "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0:
            return True
    return False


def pyspark_available() -> bool:
    """Return True when the pyspark package can be imported."""
    try:
        import pyspark
    except Exception:
        return False
    return bool(pyspark.__version__)


def spark_available() -> bool:
    """Return True when both pyspark and a JVM are present."""
    return pyspark_available() and java_available()


def unavailable_reason() -> Optional[str]:
    """Return a one line reason Spark cannot run, or None when it can.

    The text is written to be printed straight to the terminal, so it says what
    is missing and what to do about it.
    """
    if not pyspark_available():
        return (
            "pyspark is not installed. install it with "
            "pip install -r requirements-spark.txt"
        )
    if not java_available():
        return (
            "no java runtime was found. Spark needs a JVM, so install a JDK 11 "
            "or 17 and set JAVA_HOME"
        )
    return None


def build_session(
    app_name: str = "ClickRank",
    master: str = "local[*]",
    shuffle_partitions: int = DEFAULT_SHUFFLE_PARTITIONS,
    driver_memory: str = "4g",
    executor_memory: Optional[str] = None,
    broadcast_threshold: int = DEFAULT_BROADCAST_THRESHOLD,
    adaptive: bool = True,
    local_dir: Optional[str] = None,
    extra_conf: Optional[Dict[str, str]] = None,
) -> Any:
    """Build or fetch a SparkSession configured for this workload.

    Parameters
    ----------
    app_name
        Name shown in the Spark UI and in cluster schedulers.
    master
        Spark master URL. `local[*]` uses every core on this machine. Pass a
        real master such as `yarn` or a `spark://` URL to run on a cluster.
    shuffle_partitions
        Post shuffle partition count. This is the single most important knob for
        this pipeline because the categorical aggregation and the vocabulary
        joins are all shuffles.
    driver_memory
        Heap for the driver JVM. In local mode the driver is also the executor,
        so this is the memory the whole run gets.
    executor_memory
        Heap per executor on a cluster. Ignored in local mode, where the driver
        setting governs.
    broadcast_threshold
        Size in bytes below which Spark broadcasts the smaller side of a join
        instead of shuffling both sides.
    adaptive
        Enable adaptive query execution, which coalesces small post shuffle
        partitions and splits skewed ones at runtime.
    local_dir
        Scratch directory for shuffle spill. Point this at fast local storage on
        a cluster.
    extra_conf
        Any additional Spark configuration entries, applied last so a caller can
        override anything above.

    Returns
    -------
    SparkSession
        An active session. Calling this twice returns the same session, since
        `getOrCreate` is what Spark gives us.
    """
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName(app_name).master(master)
    builder = builder.config("spark.sql.shuffle.partitions", str(shuffle_partitions))
    builder = builder.config("spark.driver.memory", driver_memory)
    if executor_memory is not None:
        builder = builder.config("spark.executor.memory", executor_memory)
    builder = builder.config(
        "spark.sql.autoBroadcastJoinThreshold", str(broadcast_threshold)
    )
    builder = builder.config("spark.sql.adaptive.enabled", "true" if adaptive else "false")
    builder = builder.config(
        "spark.sql.adaptive.coalescePartitions.enabled", "true" if adaptive else "false"
    )
    builder = builder.config(
        "spark.sql.adaptive.skewJoin.enabled", "true" if adaptive else "false"
    )
    # Arrow makes the pandas handoff on both ends far cheaper. The pipeline only
    # crosses that boundary at ingestion and at the parity check, but those are
    # exactly the places where a row at a time Python conversion would dominate.
    builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
    # Parquet is the output format, and the legacy rebase modes only matter for
    # dates, which this schema does not carry. Snappy is the right default codec
    # for a file that gets read back many times.
    builder = builder.config("spark.sql.parquet.compression.codec", "snappy")
    # Keep the driver quiet in a benchmark context. The UI is still available.
    builder = builder.config("spark.ui.showConsoleProgress", "false")
    if local_dir is not None:
        builder = builder.config("spark.local.dir", local_dir)

    if extra_conf:
        for key, value in extra_conf.items():
            builder = builder.config(key, value)

    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("ERROR")
    return session
