"""Distributed feature engineering for ClickRank on PySpark.

The pandas feature pipeline in `src.data.preprocess` is the reference
implementation. It is correct, readable, and bounded by the memory of a single
machine, because it materializes the whole frame and then loops over rows in
Python to hash categorical values. This package reimplements the same
transformations as Spark SQL expressions so the work is distributed, streamed
through partitions, and never has to hold the full dataset in one address space.

Parity with the pandas path is the contract. Every transformation here is
written to reproduce the reference output exactly, including the md5 bucketing,
the rare token collapse, the train only standardization statistics, and the
frequency variance ranking that selects which columns get crossed. The parity
test in `tests/test_spark_pipeline.py` is what holds that contract honest.
"""

from __future__ import annotations

from src.spark.session import build_session, java_available, spark_available
from src.spark.hashing import md5_bucket
from src.spark.pipeline import (
    SparkFeaturePipeline,
    SparkPipelineConfig,
    spark_to_dataset,
)
from src.spark.io import (
    read_criteo,
    read_pandas,
    add_row_index,
    add_temporal_split,
    write_features,
)

__all__ = [
    "build_session",
    "java_available",
    "spark_available",
    "md5_bucket",
    "SparkFeaturePipeline",
    "SparkPipelineConfig",
    "spark_to_dataset",
    "read_criteo",
    "read_pandas",
    "add_row_index",
    "add_temporal_split",
    "write_features",
]
