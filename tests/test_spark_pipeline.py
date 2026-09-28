"""Parity tests between the Spark pipeline and the reference pandas pipeline.

The Spark lane exists to do the same work as `src.data.preprocess` on data that
does not fit in one process. That is only a useful claim if the two produce the
same numbers, so parity is the acceptance criterion for this package and these
tests are what enforce it. They compare the two pipelines column by column on the
same synthetic sample. The hashed and crossed columns are integer bucket indices
and must match exactly, because a hash that is off by one bucket is not close, it
is a different feature. The dense and frequency columns are floats and are
compared within a tolerance appropriate to float32.

Everything here skips cleanly when pyspark or a JVM is absent, so `pytest -q`
passes on a machine that has neither.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import pytest

from src.data.loader import generate_synthetic
from src.data.preprocess import FeaturePipeline
from src.data.split import temporal_split
from src.features.categorical import stable_hash
from src.schema import CAT_COLS, NUM_COLS
from src.spark.session import java_available

pytest.importorskip("pyspark", reason="pyspark is not installed.")

pytestmark = pytest.mark.skipif(
    not java_available(),
    reason="no java runtime, so Spark cannot start.",
)

# Small enough that the whole suite stays quick, large enough that the rare
# token cutoff, the missing value path, and the cross ranking all have real
# variation to disagree about.
N_ROWS = 2000
HASH_BUCKETS = 1000
CROSS_BUCKETS = 5000
MIN_COUNT = 5
TOP_K = 5

# float32 carries about seven significant decimal digits. The Spark and numpy
# standard deviation reductions use different summation orders, so the fit
# statistics can differ in the last bits even when both are correct. This
# tolerance is set to that noise floor and no wider.
FLOAT_ATOL = 1e-5
FLOAT_RTOL = 1e-5


@pytest.fixture(scope="module")
def spark():
    """One SparkSession for the whole module.

    Starting a JVM costs several seconds, so the session is built once and every
    test in the file shares it.
    """
    from src.spark.session import build_session

    session = build_session(
        app_name="ClickRankParityTest",
        master="local[2]",
        shuffle_partitions=4,
        driver_memory="2g",
    )
    yield session
    session.stop()


@pytest.fixture(scope="module")
def sample_frame():
    """The synthetic frame both pipelines are run against."""
    return generate_synthetic(N_ROWS, seed=42)


@pytest.fixture(scope="module")
def reference(sample_frame):
    """Fit the pandas pipeline on train and transform all three splits."""
    train_df, val_df, test_df = temporal_split(sample_frame)
    pipeline = FeaturePipeline(
        hash_bucket_size=HASH_BUCKETS,
        cross_bucket_size=CROSS_BUCKETS,
        min_count=MIN_COUNT,
        n_cross_features=TOP_K,
    )
    pipeline.fit(train_df)
    datasets = {
        "train": pipeline.transform(train_df),
        "val": pipeline.transform(val_df),
        "test": pipeline.transform(test_df),
    }
    return pipeline, datasets


@pytest.fixture(scope="module")
def distributed(spark, sample_frame):
    """Fit the Spark pipeline on train and transform the whole frame."""
    from src.spark.io import add_temporal_split, read_pandas
    from src.spark.pipeline import SparkFeaturePipeline, SparkPipelineConfig

    spark_df = read_pandas(spark, sample_frame)
    spark_df, _n_train, _n_val, _n_test = add_temporal_split(
        spark_df, n_rows=len(sample_frame)
    )
    pipeline = SparkFeaturePipeline(
        SparkPipelineConfig(
            hash_bucket_size=HASH_BUCKETS,
            cross_bucket_size=CROSS_BUCKETS,
            min_count=MIN_COUNT,
            n_cross_features=TOP_K,
        )
    )
    pipeline.fit(spark_df.filter(spark_df["split"] == "train"))
    featurized = pipeline.transform(spark_df).cache()
    featurized.count()
    yield pipeline, featurized
    featurized.unpersist()
    pipeline.unpersist()


def test_md5_bucket_matches_stable_hash(spark):
    """The Spark md5 fold reproduces `stable_hash` on every probe value.

    This is checked directly rather than inferred from the pipeline output,
    because it is the single place where the port could not be a mechanical
    translation. Spark has no 128 bit integer, so the digest is folded modulo the
    bucket count instead of being read as one number. The probes cover the empty
    string, the rare token, hex like Criteo values, unicode, and long strings, at
    three bucket sizes including one that is not a power of two.
    """
    from pyspark.sql import functions as F

    from src.spark.hashing import md5_bucket

    probes = [
        "",
        "__rare__",
        "C1=__nan__",
        "C1=0000000a",
        "C26=fffffffe",
        "C1=0000000a&C2=0000000b",
        "a much longer value than any criteo field would ever carry",
        "unicode éèê and emoji \U0001f600",
        "0",
        "00000000",
    ]
    frame = spark.createDataFrame([(value,) for value in probes], ["value"])

    for buckets in (7, HASH_BUCKETS, CROSS_BUCKETS):
        computed = (
            frame.withColumn("bucket", md5_bucket(F.col("value"), buckets))
            .orderBy("value")
            .collect()
        )
        for row in computed:
            expected = stable_hash(row["value"], buckets)
            assert row["bucket"] == expected, (
                f"md5 fold disagreed with stable_hash on {row['value']!r} "
                f"at {buckets} buckets."
            )


def test_temporal_split_cuts_at_the_same_rows(spark, sample_frame, distributed):
    """The Spark split assigns the same rows to the same splits as pandas.

    A distributed split is the easiest place in this port to silently shuffle
    rows across a temporal boundary, which would leak the future into training
    without changing a single shape. This checks the boundaries by row index
    rather than by count, so a split that happened to have the right sizes but
    the wrong membership still fails.
    """
    _pipeline, featurized = distributed
    train_df, val_df, test_df = temporal_split(sample_frame)

    rows = featurized.select("row_id", "split").orderBy("row_id").collect()
    assigned = [row["split"] for row in rows]
    indices = [row["row_id"] for row in rows]

    assert indices == list(range(len(sample_frame)))

    expected = (
        ["train"] * len(train_df) + ["val"] * len(val_df) + ["test"] * len(test_df)
    )
    assert assigned == expected


def test_cross_pairs_match_the_reference(reference, distributed):
    """Both pipelines rank the sparse fields the same way and cross the same pairs.

    The ranking is by the variance of each field's training frequency
    distribution. Spark computes that variance with a streaming reduction and
    numpy computes it with a two pass one, so the values are equal only to
    floating point, not bitwise. The selection is still stable because the fields
    are separated by far more than that noise, and the reference tie break on
    column name is reproduced exactly. If a future dataset ever puts two fields
    within float noise of each other, this is the assertion that will catch it.
    """
    ref_pipeline, _ = reference
    spark_pipeline, _ = distributed

    assert spark_pipeline.pairs_ == ref_pipeline.crosses.pairs_
    assert len(spark_pipeline.pairs_) == 10


def test_numerical_fit_statistics_match(reference, distributed):
    """The train only mean and std of the log transformed dense columns agree."""
    ref_pipeline, _ = reference
    spark_pipeline, _ = distributed

    for index, col in enumerate(NUM_COLS):
        assert spark_pipeline.numerical_.means[col] == pytest.approx(
            float(ref_pipeline.numerical.means_[index]), rel=1e-9, abs=1e-12
        )
        assert spark_pipeline.numerical_.stds[col] == pytest.approx(
            float(ref_pipeline.numerical.stds_[index]), rel=1e-9, abs=1e-12
        )


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_pipeline_parity_column_by_column(reference, distributed, split):
    """Every feature column matches the reference on every split.

    Integer columns are compared exactly. The hashed categorical buckets and the
    crossed buckets are indices into an embedding table, so a one bucket
    difference is a different feature and there is no meaningful tolerance for
    it. The dense and frequency columns are float32 and are compared within the
    float32 noise floor.

    Failures name the column, because a parity break in one field is a very
    different diagnosis from a parity break across all of them.
    """
    from src.spark.pipeline import spark_to_dataset

    _ref_pipeline, ref_datasets = reference
    spark_pipeline, featurized = distributed

    expected = ref_datasets[split]
    actual = spark_to_dataset(featurized, spark_pipeline, split=split)

    assert len(actual) == len(expected)
    assert np.array_equal(actual.label, expected.label)

    numerical_names = spark_pipeline.numerical_columns()
    assert actual.numerical.shape == expected.numerical.shape
    for index, name in enumerate(numerical_names):
        np.testing.assert_allclose(
            actual.numerical[:, index],
            expected.numerical[:, index],
            rtol=FLOAT_RTOL,
            atol=FLOAT_ATOL,
            err_msg=f"dense column {name} diverged on the {split} split.",
        )

    assert actual.categorical.shape == expected.categorical.shape
    for index, col in enumerate(CAT_COLS):
        assert np.array_equal(
            actual.categorical[:, index], expected.categorical[:, index]
        ), f"hashed column {col} diverged on the {split} split."

    assert actual.cat_freq.shape == expected.cat_freq.shape
    for index, col in enumerate(CAT_COLS):
        np.testing.assert_allclose(
            actual.cat_freq[:, index],
            expected.cat_freq[:, index],
            rtol=FLOAT_RTOL,
            atol=FLOAT_ATOL,
            err_msg=f"frequency column {col} diverged on the {split} split.",
        )

    assert actual.crosses.shape == expected.crosses.shape
    for index, pair in enumerate(spark_pipeline.pairs_):
        assert np.array_equal(
            actual.crosses[:, index], expected.crosses[:, index]
        ), f"cross column {pair} diverged on the {split} split."


def test_meta_matches_the_reference(reference, distributed):
    """The featurized space description matches, so models size the same tables."""
    ref_pipeline, _ = reference
    spark_pipeline, _ = distributed

    ref_meta = ref_pipeline.meta
    spark_meta = spark_pipeline.meta

    assert spark_meta.n_numerical == ref_meta.n_numerical
    assert spark_meta.cat_vocab_sizes == ref_meta.cat_vocab_sizes
    assert spark_meta.cross_vocab_sizes == ref_meta.cross_vocab_sizes


def test_parquet_output_is_partitioned_by_split(spark, distributed):
    """The written Parquet has one directory per split and reads back unchanged.

    Partitioning on the split column is what lets a downstream trainer read the
    training rows as a directory prune rather than a filter over everything. This
    checks the directory layout exists and that the round trip preserves the
    values, since a partition column is stored in the path rather than the file
    and is therefore the easiest thing to lose on write.
    """
    from src.spark.io import write_features

    spark_pipeline, featurized = distributed
    output_dir = tempfile.mkdtemp(prefix="clickrank_parity_")
    output_path = os.path.join(output_dir, "features.parquet")
    try:
        write_features(featurized, output_path)

        directories = sorted(
            name for name in os.listdir(output_path) if name.startswith("split=")
        )
        assert directories == ["split=test", "split=train", "split=val"]

        reloaded = spark.read.parquet(output_path)
        assert reloaded.count() == featurized.count()

        original = featurized.orderBy("row_id").select(
            "row_id", "label", *spark_pipeline.feature_columns()
        )
        restored = reloaded.orderBy("row_id").select(
            "row_id", "label", *spark_pipeline.feature_columns()
        )
        assert original.collect() == restored.collect()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
