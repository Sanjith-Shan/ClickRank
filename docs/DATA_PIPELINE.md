# Distributed Data Pipeline

This document covers the distributed lane of ClickRank. There are two halves. The first is a PySpark reimplementation of the feature pipeline that produces numerically identical output to the pandas reference, which lives in `src/spark/` and runs through `scripts/run_spark_pipeline.py`. The second is a SQL analytics layer on DuckDB that measures the data behind each feature engineering decision the project already makes, which lives in `src/sql/` and runs through `scripts/run_data_insights.py`.

The two lanes have very different dependency footprints and that is deliberate. Spark needs a JVM. DuckDB needs nothing. Every Spark entry point probes for a Java runtime and exits cleanly with a message when it cannot find one, and the Spark tests skip rather than fail, so a machine with no JDK still runs `pytest -q` green and still gets the full analytics report.

## Why a Distributed Pipeline for This Data

The pandas pipeline is correct and it is the reference. It is also bounded in two separate ways, and it is worth being precise about which bound bites first, because the usual answer is wrong for this workload.

The obvious bound is memory. The pandas path reads the whole file into one frame, splits it, and holds three featurized `Dataset` containers alongside it. Nothing streams. On the real Criteo file that ceiling arrives long before the 45 million labeled rows the dataset actually has, which is why `scripts/download_data.sh` takes a positional head rather than the whole file and why the README quotes a 2 million row run.

The bound that actually bites first is throughput, and it comes from the encoders rather than from memory. `CategoricalEncoder.transform_hash` loops over every value of every sparse column in interpreted Python and calls `hashlib.md5` once per cell. That is 26 md5 calls per row, plus one dictionary lookup per cell in `transform_freq`, plus one more md5 per cross. A million row sample is 26 million interpreted md5 calls before the model sees a single batch. The frame would still fit in memory well past the point where the wall time stops being tolerable.

The Spark port fixes both, but it fixes the second one first and by a wider margin, because the per value work moves from interpreted Python into a JVM expression that Spark's code generator compiles and runs per partition. That is the honest framing of what this lane buys, and `scripts/run_spark_pipeline.py --scale` measures it rather than asserting it.

The third thing it buys is a shape. Real feature pipelines write partitioned columnar output that a trainer reads back split by split. That is what this one does, and it is a structural difference from handing a numpy array to a constructor, not a performance one.

## Parity Is the Contract

A distributed rewrite of a feature pipeline is only useful if it computes the same features. If it does not, every model number the project has published becomes unreproducible through the new path, and a silent one bucket difference in a hash column is not a small error. It is a different feature.

So parity is the acceptance criterion for `src/spark/`, and `tests/test_spark_pipeline.py` enforces it column by column on the same synthetic sample. Integer columns are compared exactly, because a hashed bucket index is an index into an embedding table and there is no meaningful tolerance on it. Float columns are compared at the float32 noise floor and no wider.

Three parts of the port needed real care.

### The md5 fold

`src.features.categorical.stable_hash` takes the full md5 hex digest, reads it as a single 128 bit integer, and reduces it modulo the bucket count. Python does that without thinking about it. Spark cannot. It has no 128 bit integer type, and `conv` on a 32 character hex string saturates at the 64 bit boundary, so the obvious port silently reduces a truncated digest and produces a completely different bucket for most inputs. Nothing about the output shape would reveal it.

`src/spark/hashing.py` folds the digest instead. It reads the 32 hex characters as four base 2^32 limbs and applies Horner's method with a modulo at every step, which computes the same residue as the arbitrary precision path because modular arithmetic distributes over the multiply and the add. The widest intermediate is `buckets * 2^32 + (2^32 - 1)`, so any bucket count below about 2.1 billion stays inside a signed 64 bit long. The pipeline uses 10000 and 100000.

Four limbs rather than eight is not an arbitrary choice. The first version folded four hex characters at a time, which is also exact, and the eight step expression instantiated once per sparse field and once per cross produced a generated Java method past the 64 KB JVM limit. Spark fell back to interpreted execution for the whole stage, which is correct but slow, and it logged a compiler exception in the middle of a run that otherwise looked healthy. Halving the step count fixed it. This is the kind of thing that only shows up when the code is actually run, which is why it was.

`test_md5_bucket_matches_stable_hash` checks the fold against `stable_hash` directly on the empty string, the rare token, hex like Criteo values, unicode, and long strings, at three bucket counts including one that is not a power of two. It does not infer correctness from the pipeline output.

### NaN against NULL

The reference uses `np.nanmean` and `np.nanstd`, which skip missing values so that the zero fill applied at transform time cannot drag the training mean down. Spark aggregates skip NULL. They do not skip NaN, they propagate it, and a float column arriving from pandas carries NaN and not NULL. Left alone, every dense mean and standard deviation in the fit would have come back NaN, the guards would have replaced them with 0 and 1, and the pipeline would have silently stopped standardizing anything.

`src/spark/io.read_pandas` collapses NaN into NULL at ingestion so there is one missing value to reason about downstream instead of two. For the same reason `_clip_negative` is an explicit conditional rather than `greatest(col, 0)`. Spark's `greatest` ignores NULL and would return zero for a missing cell, erasing the missingness before the indicator that encodes it is computed.

### The cross ranking

`CrossGenerator.fit` ranks sparse fields by the population variance of their normalized training frequency distribution, sorts by negative variance then by column name, takes the top k, sorts those by name, and forms every unordered pair. The Spark side computes the same variance with `var_pop` over the same distribution and applies the identical Python sort and tie break.

The one place this is equal only to floating point rather than bitwise is the variance itself, because Spark reduces it with a streaming algorithm and numpy uses a two pass one. In practice the fields are separated by orders of magnitude more than that noise, and two genuinely constant fields both give exactly zero and fall through to the stable tie break on name. `test_cross_pairs_match_the_reference` asserts the selected pairs are identical, so if a future dataset ever puts two fields within float noise of each other, the test is what surfaces it rather than a quietly different feature set.

### Row order and the temporal split

The pandas split is positional and gets its ordering for free. Spark has no inherent row order, and a split that shuffles rows across a temporal boundary leaks the future into training without changing a single shape, which makes it the most dangerous class of bug in this port.

`add_row_index` attaches a dense zero based index that respects file read order. `monotonically_increasing_id` is monotonic in read order but packs the partition id into the high bits and therefore leaves gaps, so the index cannot be compared against an absolute row count. The function counts the rows in each partition, turns those counts into a prefix sum on the driver, and adds each partition's offset to a local row number within the partition. The partition count is small, so the offset map is a literal in the plan.

That index is then the only thing the split reads, it survives every join and shuffle downstream, and it becomes the sort key for any collection back to pandas. `test_temporal_split_cuts_at_the_same_rows` checks membership by row index rather than by split size, so a split with the right counts and the wrong rows still fails.

### Result

On a 2000 row synthetic sample with 1000 hash buckets, 5000 cross buckets, a min count of 5, and a top k of 5, every column matches. The hashed categorical columns, the crossed columns, and the labels are exactly equal. The dense and frequency columns came back bitwise equal as well, with a maximum absolute difference of 0.0, which is stronger than the test requires and is not something the test asserts, since the fit statistics are not guaranteed to reduce in the same order on both sides.

There is no known discrepancy to document. If one appears, the rule is that it gets written down here and marked xfail in the test with the reason, not tolerated away.

## Partitioning and File Layout

Output is Parquet partitioned by the split column.

Partitioning on split is the choice that matters. A trainer reading `split=train` gets a directory prune rather than a filter over the whole dataset, so loading one split never touches the bytes of the other two. Partitioning on anything else here would be worse. A sparse field has far too many distinct values and would produce a directory per value. The row index is unique and would produce a directory per row.

Within a split, `--output-files` controls how many files are written. This is the small files problem in its usual form. Spark writes one file per partition per split by default, so a run with 200 shuffle partitions writes 600 files, most of them a few kilobytes, and every subsequent reader pays a listing and open cost per file that dwarfs the bytes inside. A handful of files in the tens to hundreds of megabytes is the target. `write_features` repartitions before the write when asked, which costs one shuffle and saves it back on every read afterwards.

Snappy is the compression codec. Parquet's dictionary encoding already does most of the work on the hashed integer columns, which have low entropy relative to their width, and Snappy decompresses fast enough that a reader is not paying for the space saving. Gzip or zstd would be smaller and slower to read, which is the wrong trade for a file that is written once and read every epoch.

The flat column layout is deliberate too. The featurized frame has one named column per feature rather than array columns, which means a downstream reader can project only the columns it needs, the SQL analytics layer can query the output directly, and a parity failure names the column that broke rather than an index into an array.

## Spark Configuration That Matters Here

Nothing in `src/spark/` reads a hardcoded configuration value. `build_session` takes every knob as a parameter, so the same code path produces a laptop session and a cluster session.

**Shuffle partitions.** Spark defaults to 200. For a few million rows on one machine that turns every shuffle into 200 tasks that each do almost nothing, and scheduling overhead dominates. The default here is 16, a small multiple of a laptop's core count. On a cluster the number should be set so each post shuffle partition lands somewhere in the low hundreds of megabytes. This pipeline shuffles at the vocabulary aggregation and at every vocabulary join, so it is the knob with the most leverage.

**Broadcast join threshold.** The pipeline joins the fact table against a per field vocabulary table 26 times. When a vocabulary is small enough to broadcast, the fact table is never shuffled and the join is nearly free. When it is not, both sides shuffle and the cost is real. `spark.sql.autoBroadcastJoinThreshold` governs Spark's own decision, and `SparkPipelineConfig.broadcast_vocab_max_rows` governs the explicit `broadcast` hint the pipeline applies, based on the distinct kept value count it recorded at fit time. On synthetic data every field is far under the threshold. On real Criteo the widest fields are not.

**Adaptive query execution.** Enabled by default. It coalesces small post shuffle partitions and splits skewed ones at runtime, which matters because the skew here is not knowable in advance.

**Plan depth.** This is the knob that is not a configuration setting. Adding 26 columns with 26 chained `withColumn` calls builds a tower of 26 Project nodes that Catalyst walks on every optimizer pass and that the code generator then tries to fit into one JVM method. The pipeline adds each block of columns in a single `select` instead. The output is identical and the plan is a fraction of the size. This is the same pressure that forced the four limb md5 fold, seen from the other side.

## What Skew Looks Like on Criteo

The skew in this workload is not in the row distribution. It is in the categorical value distribution, and it hits the vocabulary aggregation.

The aggregation unpivots all 26 sparse columns into one narrow table and groups by field and value. Grouping by field alone would be catastrophic, since it produces exactly 26 groups and therefore 26 reducers regardless of how many partitions are configured. Grouping by the pair spreads the work, and it is why the fit does one shuffle rather than 26 separate group by jobs. That single aggregate then serves three purposes, the rare token cutoff, the frequency encoding, and the variance ranking that selects columns for crossing.

The residual skew is that a Criteo sparse field is not uniform. A handful of values cover a large share of the traffic and the rest is a very long tail, and a hash partitioner sends all rows of one value to one reducer. The heavy values therefore produce heavy partitions. The insights report measures this concentration directly per field, and the cardinality table there is the right place to look before tuning.

Three things address it. Adaptive query execution splits a skewed partition at runtime, which handles the common case. Raising the shuffle partition count spreads the tail more finely without helping a single heavy key. Where one key genuinely dominates, salting the key is the standard fix, which this pipeline does not currently implement because it has not needed to at the sample sizes measured.

The joins skew for the same reason. A left join on a sparse value sends every row carrying the heaviest value to one task. Broadcasting the vocabulary avoids this entirely by removing the shuffle from the fact table side, which is the main reason the broadcast threshold is a knob rather than a constant.

## Running It

Install the extra dependencies. These are kept out of `requirements.txt` on purpose, so the benchmark, the inference lane, and the analytics all stay runnable on a machine with no Java.

```bash
pip install -r requirements-spark.txt
```

Run the pipeline over the synthetic generator or a real Criteo file. It writes Parquet partitioned by split and reports rows, wall time, partition count, and output size.

```bash
python scripts/run_spark_pipeline.py --synthetic --sample-size 50000
python scripts/run_spark_pipeline.py --data-path data/criteo.csv --sample-size 2000000
```

Measure the two pipelines against each other at increasing row counts.

```bash
python scripts/run_spark_pipeline.py --synthetic --scale
```

Run the SQL analytics. This needs no JVM and writes the report and four charts into `results/insights/`.

```bash
python scripts/run_data_insights.py --synthetic --sample-size 200000
```

Run the parity tests on their own. They skip cleanly when pyspark or a JDK is absent.

```bash
pytest tests/test_spark_pipeline.py -q
```

## Reading the Numbers

Every measurement this lane produces is labeled with the hardware it ran on, the row count it came from, and the one minute load average at the time. The last one is there because a wall time on a laptop is only interpretable next to what else the machine was doing, and a run that competed for cpu with an unrelated build produces a wall time that is an upper bound and a throughput that is a lower bound. Where the report says the machine was contended, the numbers should be reproduced on a quiet machine before being quoted.

The parity result does not carry that caveat. It is exact arithmetic and it does not depend on how busy the machine was.
