# ClickRank

ClickRank (formerly AdRankBench) is an ad retrieval and ranking system measured on real data. A two tower model and a FAISS index pick a few hundred candidates out of 846,811 real Taobao ads, CTR rankers (DeepFM, DCN, DIN) score them, and the whole path is served over HTTP. The CTR model comparison, calibration and the CPU and GPU inference work run on real Criteo click logs. What it does not claim. The models are the standard published designs, credited in `DESIGN.md`. Retrieval recall is measured against exact search on the same embeddings, not against a leaderboard. Retrieval numbers come from Taobao and CTR and inference numbers come from Criteo, and no figure blends the two. Every figure is in `NUMBERS.md` or the README tables with the results file it came from.

## Retrieval, then ranking (Taobao)

Criteo has no user or ad identifiers, so the retrieval stage runs on the Taobao display ads log (26.5M impressions over 8 days, 1.14M users, 846,811 ads). Days 1 to 7 train, day 8 is the test day, and every model choice was made on a day 7 validation split without looking at day 8.

**Retrieval.** A two tower model (Covington et al. 2016) trained with in batch sampled softmax and the log q sampling bias correction of Yi et al. 2019. On the test day the clicked ad is in the user's top 500 of 846,811 ads for 19.6% of clicks, against 12.5% for a popularity list and 0.06% by chance. The sampling bias correction is the largest single effect in the design. Without it, validation hit rate falls from 10.8% to 4.3%.

**Index.** An IVF index returns 97.2% of the exact top 100 at 1.5 ms per query on one thread, and IVF-PQ holds the whole corpus in 35 MB instead of 217 MB while returning 83% of the exact top 100. Both were timed while other jobs ran, so the clean per request timings are the ones below. The full sweep of flat, IVF, IVF-PQ and HNSW settings is in `NUMBERS.md`.

**Why two stages.** Replaying 1,000 test day clicks, DeepFM scoring all 846,811 ads puts the clicked ad in its top 50 for 0.2% of requests (median rank 311,746). Retrieving 50 first and ranking only those puts it in the top 50 for 9.5%. DCN on a 300 request sample shows the same split, 0.3% against 10.0%. A ranker trained on logged impressions has only seen ads some earlier system chose to show, so it cannot order the whole catalog, and retrieval is what fixes that. The impression level metrics, which fall when retrieval is added, and why, are in `DESIGN.md`.

**Latency.** Per request on the M3 Pro CPU with one torch thread and one FAISS thread, the two stage path (user tower, HNSW search, candidate features, DeepFM on the candidates) takes 1.28 ms at the median with K = 50 and 2.12 ms with K = 500. Scoring all 846,811 ads with the same DeepFM takes 2,477 ms on one thread and 1,246 ms on all twelve cores, its best case. With exact search instead of HNSW the two stage path is 6.7 to 9.1 ms, almost all of it the brute force search. These ran at a load average near 11 while the machine was otherwise shared, and every row in `results/retrieval/stage_latency.jsonl` records that load.

**Freshness.** The same DeepFM trained on one day of data loses test day AUC steadily with age, from 0.587 one day stale to 0.566 seven days stale (He et al. 2014, Section 5). A full retrain on new days gains 0.012 to 0.017 AUC over not updating, on two seeds, while a cheap one pass warm start did not reliably recover that gap. See Model Freshness below and `DESIGN.md`.

```bash
python scripts/run_retrieval.py --source taobao --temperature 0.1 --epochs 2
python scripts/run_index_sweep.py --source taobao
python scripts/run_rankers.py --source taobao
python scripts/run_two_stage.py --source taobao
python scripts/make_numbers.py
```

## CTR models on Criteo

![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)

ClickRank trains and evaluates five click through rate prediction architectures (Logistic Regression, FM, DeepFM, DCN, DNN) on the Criteo Display Advertising Challenge dataset. The framework implements production grade feature engineering with log transforms, hash encoding, frequency encoding, and explicit feature crosses. All models are evaluated with AUC, logloss, normalized entropy, group AUC, and calibration analysis using temporal train and test splits to prevent data leakage.

The project is built around the core problems of ad ranking systems. It covers ranking and retrieval, query understanding and relevance through a two tower DSSM model, probability calibration for ad pricing, and budget pacing through a feedback controller simulator. Each piece maps to a concrete skill that ad ranking and search ads teams care about.

## Results

The table below comes from a run on 2 million rows of the real Criteo Display Advertising Challenge dataset, trained on the first 1.6 million rows with a temporal split and seed 42. The real Criteo click rate is about 25 percent and the categorical fields reach into the hundreds of thousands of unique values. After placing the Criteo file at `data/criteo.csv` you can reproduce this with `python scripts/run_benchmark.py --sample-size 2000000`. The same table is mirrored into `results/benchmark_report.md`. Models are sorted by test AUC.

| Model | AUC | LogLoss | NE | RelaImpr | GAUC | ECE | Train s | Params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepFM | 0.7872 | 0.4512 | 0.8130 | 0.1870 | 0.7873 | 0.0100 | 132 | 21.6M |
| DCN | 0.7871 | 0.4508 | 0.8124 | 0.1876 | 0.7871 | 0.0052 | 140 | 21.6M |
| DNN | 0.7863 | 0.4518 | 0.8141 | 0.1859 | 0.7864 | 0.0107 | 126 | 21.9M |
| FM | 0.7828 | 0.4553 | 0.8205 | 0.1795 | 0.7829 | 0.0091 | 146 | 21.4M |
| LogisticRegression | 0.7177 | 0.4983 | 0.8979 | 0.1021 | 0.7177 | 0.0057 | 11 | 53 |

All four interaction aware models beat the logistic regression baseline on real Criteo data. DeepFM and DCN lead at about 0.787 AUC against 0.718 for logistic regression, a lift of about 0.069 AUC and a drop in normalized entropy from 0.898 to 0.813. On real Criteo the categorical features have very high cardinality, so hash encoding and embeddings are essential, and the high order feature interactions are what separate the deep models from the linear baseline. Every model stays well calibrated with expected calibration error under 0.011. These numbers are in line with published Criteo benchmarks given the bounded 10000 bucket hash space this framework uses to keep memory in check. For fast experiments with no download the benchmark also ships a synthetic generator with built in interactions, selected with the `--synthetic` flag. See the methodology notes below and the dedicated document in `docs/METHODOLOGY.md` for the full reasoning.

## Inference Optimization

A trained ranker is only useful if it can score live traffic inside a tight latency budget, so the model has to leave the training framework and run on an inference optimized runtime. This stage exports the trained model once to ONNX and then sweeps every serving backend the project supports across a grid of model, runtime, precision, and batch size, on the exact same held out test rows.

The runtimes are PyTorch eager mode, ONNX Runtime on the cpu, cuda, and TensorRT execution providers, a natively built TensorRT engine at fp32, fp16, and int8, and OpenVINO on the cpu. The reasoning behind each of them, the precision and calibration story, and the measurement methodology all live in `docs/INFERENCE.md`.

### Two lanes, never blended

OpenVINO and the ONNX Runtime cpu provider are the **cpu lane**. Everything with cuda or TensorRT in its name is the **gpu lane**. They answer different deployment questions, so their numbers are reported separately and are never combined into a single figure. Every row carries the hardware it ran on.

### The cpu lane, measured

Measured on an **Apple M3 Pro cpu** on a quiet machine, DeepFM, 10000 held out rows in batches of 1024, seed 42. This is a laptop cpu number and not a datacenter number, and it is stated that way wherever it appears.

| Backend | Latency (ms/batch) | p99 (ms) | Throughput (samples/s) | AUC |
| --- | --- | --- | --- | --- |
| PyTorch eager | 2.154 | 3.031 | 464,293 | 0.7827 |
| ONNX Runtime | 4.168 | 18.477 | 239,909 | 0.7827 |
| OpenVINO | 1.467 | 2.067 | 681,526 | 0.7827 |

![Inference latency by backend](results/inference_latency.png)

All three backends run the same weights on the same rows, so the matching AUC is a correctness check rather than a coincidence. It confirms the export and the optimized runtimes preserve the model output instead of trading accuracy for speed. OpenVINO compiles the network for the host cpu, which is why it holds both the lowest latency and the tightest tail here.

The committed artifact in `results/inference_benchmark.md` covers the full sweep across both DeepFM and DCN at batch sizes 1, 256, 1024, and 4096. Note that it was captured while this laptop was running other work, and the report records the machine load next to the numbers for exactly that reason. Its absolute latencies are therefore higher than the quiet machine table above. The relative ordering and every accuracy column are unaffected, because those are deterministic on fixed rows.

### The gpu lane, measured

Measured on a rented **NVIDIA A100 SXM4 80GB** on RunPod Secure Cloud, driver 580.159.04, CUDA driver 13.0, TensorRT 11.2.1.2, ONNX Runtime 1.29.0, torch 2.8.0+cu128, host cpu an AMD EPYC 7742. The gpu was idle apart from this benchmark. These are rented cloud gpu numbers and they are never combined with the Apple Silicon cpu numbers above.

TensorRT across all three precisions, median latency per batch and steady state throughput.

| Precision | batch 1 | batch 256 | batch 1024 | batch 4096 | Peak throughput | AUC | LogLoss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp32 | 0.215 | 0.263 | 0.376 | 0.920 | 3,440,149 /s | 0.7731 | 0.4796 |
| fp16 | 0.210 | 0.261 | 0.361 | 0.856 | 4,769,481 /s | 0.7731 | 0.4796 |
| int8 | 0.207 | 0.246 | 0.347 | 0.750 | 5,386,846 /s | 0.7734 | 0.4782 |

Against the other runtimes at batch 1024, on the same rows and the same gpu.

| Backend | p50 (ms) | Throughput (samples/s) | AUC |
| --- | --- | --- | --- |
| TensorRT int8 | 0.347 | 2,936,026 | 0.7734 |
| TensorRT fp16 | 0.361 | 2,825,700 | 0.7731 |
| TensorRT fp32 | 0.376 | 2,728,896 | 0.7731 |
| PyTorch eager CUDA fp32 | 0.823 | 222,178 | 0.7731 |
| PyTorch eager CUDA fp16 autocast | 0.989 | 390,076 | 0.7731 |

Engine build time is a one time cost and is reported on its own. fp32 took 4.1 s and produced an 86.6 MB engine, fp16 took 3.8 s and produced a 43.4 MB engine, and int8 took 25.4 s and produced a 106.2 MB engine.

The raw artifacts for both gpu runs are committed. `results/gpu_a100_trt11/` is the run every number in this section comes from, on TensorRT 11.2.1.2 and driver 580.159.04, on an otherwise idle card. `results/gpu_a100_trt10/` is an earlier run of the same code on TensorRT 10.8.0.43 and driver 570.195.03, kept because it is a second TensorRT generation on the same model and it reaches the same conclusion about precision. Its absolute latencies should not be quoted, because that card was shared with another job at the time and the tails show it. Its medians agree with the newer run to within a few percent, which is the reason to trust the medians in both.

### Lowering the precision barely helps, and that is the finding

The prediction was that a DLRM style ranker is memory bandwidth bound rather than compute bound, so cutting the precision would return far less than the usual 2x for fp16 and 4x for int8. The measurement says the prediction was right, and more starkly than expected.

Going from fp32 to fp16 buys between 1.01x and 1.07x. Going all the way to int8 buys between 1.04x and 1.23x. For comparison, a compute bound convolutional network on the same card would be expected to roughly double on fp16.

Three independent pieces of evidence point at the same cause.

The first is the engine sizes. The fp16 engine is genuinely half the size of the fp32 one, 43.4 MB against 86.6 MB, so the weights really were converted and this is not a case of a precision flag being quietly ignored. It still did not get faster. The int8 engine is **larger** than the fp32 engine at 106.2 MB, because the embedding table is not quantized and the quantize and dequantize node pairs add to the graph without removing the bulk of it.

The second is PyTorch. Its fp16 autocast path is **slower** than its fp32 path, 0.989 ms against 0.823 ms at batch 1024. The casts cost real time and there is no arithmetic bottleneck for them to relieve.

The third is the cost model in `src/inference/analysis.py`, which puts DeepFM's arithmetic intensity at 0.50 FLOPs per byte at batch 1 rising to 59 at batch 4096. The batch 4096 column is where int8 does best at 1.23x, which is exactly where the model is closest to being compute bound, and the batch 1 column is where it does worst at 1.04x, which is where it is most memory bound. The speedup tracks the arithmetic intensity.

The practical conclusion is that quantizing this model is not where the wins are. The embedding gather is, and that is a memory layout and caching problem rather than a precision problem.

### Int8 cost no accuracy at all

| Precision | AUC | AUC delta | LogLoss | LogLoss delta |
| --- | --- | --- | --- | --- |
| fp32 | 0.7731 | reference | 0.4796 | reference |
| fp16 | 0.7731 | 0.0000 | 0.4796 | 0.0000 |
| int8 | 0.7734 | +0.0003 | 0.4782 | -0.0014 |

Int8 came out marginally ahead on both metrics. That difference is small enough to be noise rather than a real improvement, and the honest reading is that int8 cost nothing measurable. The reason is visible in how the quantization was set up. Only the MatMul and Gemm operations were quantized, which is the multilayer perceptron. The embedding gather was deliberately left in fp32, because an embedding lookup moves bytes rather than doing arithmetic, so quantizing it would add a rounding step to every lookup and accelerate nothing. Most of this model's parameters therefore never left fp32, which is why the accuracy held and also why the speedup was small. The two results are the same fact seen from two sides.

### Getting int8 to build at all

The spec called for an `IInt8EntropyCalibrator2`. That path does not exist on TensorRT 11, which removed implicit quantization entirely along with `BuilderFlag.FP16`, `BuilderFlag.INT8`, and `set_calibration_profile`. Networks are strongly typed now, so precision is a property of the graph rather than a request to the builder. Int8 moved to explicit quantization, where the graph carries QuantizeLinear and DequantizeLinear pairs inserted by a calibration pass that runs before the builder. The calibration rows still come from the validation split, because calibration is a fitting procedure and fitting it on test rows would make the reported int8 loss look smaller than it is.

Four separate incompatibilities had to be resolved between the ONNX Runtime quantizer and the TensorRT parser, each of which failed the build outright.

| Symptom | Cause | Fix |
| --- | --- | --- |
| Calibration failure with no scaling factors | The calibrator does not attach under dynamic shapes | Moved to explicit qdq quantization |
| `only activation types allowed as input to this layer` | ONNX Runtime quantizes bias to int32, TensorRT accepts int8, fp8, and fp4 only | `QuantizeBias: False` |
| `TensorRT only supports symmetric quantization` | Asymmetric activation zero points | `ActivationSymmetric: True` |
| `axis must be in the range [0, nbDims (0)]` | Scalar constants quantized with an axis attribute | Restricted the quantized ops to MatMul and Gemm |

The last one is worth keeping in mind, because restricting the op list is what a person would have wanted anyway. Add and Mul on scalar constants were never going to be where the time went.

### Two things that are not gpu numbers

**ONNX Runtime never reached the gpu.** Its CUDA and TensorRT execution providers both failed to load with `libcublasLt.so.13: cannot open shared object file`, because ONNX Runtime 1.29 wants CUDA 13 libraries that this image does not carry. It fell back to the cpu provider without failing the run, which is exactly the dangerous case, because the rows still appear and still carry the correct AUC. The giveaway is the latency. ONNX Runtime on the so called cuda provider measured 105 ms per batch against the cpu provider's 94 ms, which is the same number, and against native TensorRT's 0.376 ms. A backend that silently answers from the wrong device is the reason this benchmark prints the provider list and the reason the accuracy column alone is not a sufficient check.

**The cpu rows from this host are not comparable to anything.** PyTorch eager on the pod's EPYC measured 1900 ms per batch at batch 1024 against 5.4 ms on the Apple M3 Pro. That is thread thrash on a 128 core machine rather than a property of the processor, and no conclusion should be drawn from it. The Apple Silicon table earlier in this section is the cpu lane.

### Reproducing it

```bash
python scripts/run_inference_benchmark.py

# On a cuda host. Builds engines first, then sweeps.
python scripts/build_trt_engines.py --models deepfm dcn --precisions fp32 fp16 int8
python scripts/run_inference_benchmark.py --models deepfm dcn --batch-sizes 1 32 256 1024 4096
```

A pinned container is in `docker/Dockerfile.tensorrt` and the end to end rented GPU flow, including a bootstrap that installs TensorRT and a sync and run driver, is in `deploy/runpod/`. Backends that are not installed are skipped with a stated reason rather than failing, so the script always runs end to end with whatever is available.

## Serving

A benchmark table is not a serving system, so the inference work terminates in one. `src/serving/` is a FastAPI service that loads the ranker once, selects the fastest available backend through the same registry the benchmark uses, and scores a whole candidate set for one auction rather than an arbitrary batch.

The part that matters is that **the fitted feature pipeline is persisted at training time and loaded at serving time**. A request arrives as raw Criteo shaped fields, not as a featurized tensor, so the train only standardization statistics, the frequency encoding maps, and the hash and cross configuration all have to travel with the model. Applying different transforms online than were applied offline is training and serving skew, and it is the classic silent failure in production ranking. There is a test that asserts the served features match the offline pipeline exactly, and another that asserts the online and batch lanes produce identical scores for identical rows, because both lanes share one model artifact and one feature pipeline.

`scripts/run_load_test.py` drives the service at controlled concurrency and reports throughput, p50 through p999, the error rate, and the concurrency knee past which p99 climbs faster than throughput improves. It measures against an explicit p99 budget of 25 ms, derived in `docs/SERVING.md` as a slice of a roughly one hundred millisecond ad request. The current committed run **did not meet that budget** and says so plainly. It was also captured on a heavily loaded laptop, which the report records, so it should be rerun on a quiet machine before the number means anything.

## Data Pipeline

The pandas feature pipeline is bounded by memory. `src/spark/` reimplements it in PySpark so it is bounded by the cluster instead, writing partitioned Parquet the way a real pipeline hands data downstream. Correctness is the acceptance criterion rather than an afterthought, so there is a column by column parity test asserting the Spark and pandas paths produce matching output on the same rows across all three splits, including exact agreement on the md5 hash buckets.

An honest negative result is worth recording. Across the row counts measured so far pandas stayed faster than Spark and no crossover was observed, which is the expected shape at small scale where Spark pays session startup and shuffle costs that it cannot yet amortize. Those runs were also taken on a contended machine, so `results/spark_pipeline_report.md` marks its wall times as upper bounds. No crossover point is extrapolated, because an extrapolated crossover is not a measurement.

`scripts/run_data_insights.py` is the SQL half, running DuckDB over the data to produce `results/insights/insights_report.md`. Its through line is that each feature engineering decision this project already made is backed by a measurement rather than by convention. Categorical cardinality and tail coverage justify hash encoding, missingness correlated against the label justifies the is_missing indicators, feature pair lift over marginals justifies the crosses, and CTR drift across the temporal split justifies not using a random split. Every finding carries its sample size, so a high CTR slice with twelve impressions is not presented as a finding.

## Benchmark Automation

A benchmark nobody watches rots, so the benchmark is a gate. `scripts/check_regression.py` compares a fresh run against a committed baseline per cell, on model and backend and batch size, and it distinguishes accuracy regressions from latency regressions because a faster model that ranks worse is not an improvement.

The noise handling is the part worth reading. It compares medians rather than means, since a real run here had a mean of 44.5 ms against a median of 25.6 ms. A regression has to clear a relative floor, an absolute floor, and a per cell noise floor estimated from the recorded spread. Accuracy gets no noise term at all, correctly, because the benchmark is deterministic on fixed rows at seed 42, so a moved AUC is a changed model. A baseline is scoped to its hardware and to its configuration, and the gate refuses to compare across either.

This is not theoretical. The gate fired on a run where nothing in the code had changed, because the laptop was under concurrent load and two cells moved several hundred percent. Two other cells moved just as much and were correctly held back by their own noise floors. That episode is written up in `docs/BENCHMARK_AUTOMATION.md` as observed evidence rather than quietly dropped.

`scripts/sweep.sh` detects whether it is on a cpu or gpu box, picks the matching config from `benchmarks/`, runs the build and the sweep and the gate, and archives everything into a timestamped directory with a manifest recording the host, the git sha, and whether the tree was dirty.

## Architecture

ClickRank is a two stage pipeline. The first stage is feature engineering. Raw rows flow through a temporal split and then through a fit and transform feature pipeline that produces standardized numerical features, hash encoded categorical features, frequency encodings, and second order feature crosses. The second stage is model training and evaluation. The featurized splits are handed to each model, the trainer runs an early stopping loop, and the evaluation harness computes ranking and calibration metrics on the held out test split.

This mirrors how production ad ranking works. A retrieval stage built on something like BM25 or approximate nearest neighbor search narrows millions of candidates down to a shortlist. A ranking model in the FM or DeepFM or DCN family then scores that shortlist with rich feature interactions. ClickRank focuses on the ranking stage and adds a separate two tower module that demonstrates the retrieval and relevance side.

## Models

- Logistic Regression. The linear baseline. It uses standardized numerical features plus frequency encoded categoricals. Every other model is expected to beat it.
- FM (Factorization Machine). Adds explicit second order feature interactions through low rank embeddings using the efficient O(kn) interaction formulation.
- DeepFM. Combines the FM component with a deep network over a shared embedding table so it captures both low order and high order interactions.
- DCN (Deep and Cross Network). Pairs an explicit cross network with a deep tower to compare explicit against implicit interaction modeling.
- DNN. A plain feedforward network over embeddings and numerical features. This is the throw a neural net at it baseline.

## Feature Engineering

Numerical features in ad data follow heavy tailed power law distributions, so the numerical pipeline clips negatives to zero, applies a log1p transform, and standardizes with statistics fit on train only. Missing values are filled with zero and a binary is_missing indicator is added per column, which turns missingness into its own signal rather than discarding it. The output width is 26, made of 13 transformed values and 13 indicators.

Categorical features have very high cardinality, so one hot encoding is not an option. The pipeline uses stable md5 based hash encoding into a fixed bucket space per field, which keeps memory bounded and stays deterministic across processes. It also produces a frequency encoding where each value is replaced by its normalized train frequency, which the logistic baseline consumes. Rare categories below a minimum count collapse to a shared rare token before hashing to reduce noise from one off values.

Feature crosses encode explicit second order interactions. The cross generator picks the top categorical columns by frequency variance, forms all pairwise combinations, and hashes each crossed value into its own bucket space. This is what production ad ranking systems do by hand, and it gives even the simpler models access to interaction signal.

## Evaluation

Every model is scored on the held out test split with several metrics, because no single number captures ad ranking quality.

- AUC. Area under the ROC curve. It measures pure ranking quality across the population.
- Logloss. Binary cross entropy. It is calibration aware and punishes confident wrong probabilities.
- Normalized Entropy (NE). Logloss divided by the entropy of the base click rate. A value below 1 means the model beats the constant base rate predictor. NE is robust to the overall click rate of the traffic, which is why it is the standard metric in the ad CTR literature.
- Relative Improvement (RelaImpr). The fractional reduction in cross entropy against a constant base rate predictor.
- GAUC (Group AUC). AUC computed within each impression group and averaged with impression weights. This is the production relevant view because a real auction ranks ads inside one user request or one query, not across the whole dataset. In real systems the group key is a user id or a query id. ClickRank synthesizes impression groups for the test set so GAUC is computable offline.

Calibration is treated as a first class concern. The harness builds reliability curves by binning predictions into equal width buckets and comparing mean predicted probability against the observed click fraction. It also reports the Expected Calibration Error and overlays every model against the perfect calibration diagonal in a saved PNG. Calibration matters for ad pricing because bids and pacing decisions multiply the predicted click probability by a value, so a systematic bias in the probability turns directly into mispriced auctions. A model can rank well and still be miscalibrated, which is why calibration is reported alongside AUC.

## Search Ads Relevance

The relevance module is a two tower DSSM style semantic matching model for query to ad relevance. It lives in `src/relevance/` and runs through `scripts/run_relevance.py`. This is the retrieval and query understanding side of ad ranking, where the job is to match a short noisy query to relevant ad creatives.

Text is encoded with letter trigram word hashing, the original DSSM technique. Each token is wrapped with a boundary marker and broken into character n grams, and each n gram is hashed into a fixed bucket space to form a sparse bag of n grams vector. This sidesteps the open vocabulary problem and needs no external tokenizer or pretrained weights, so the whole module stays tiny and fast on cpu and brings no heavy dependencies.

The model has a separate query tower and ad tower. Each tower is a small MLP that maps the word hash vector into a shared embedding space. The relevance score is the cosine similarity of the two embeddings scaled by a temperature, which is the standard DSSM scoring choice and keeps the score bounded and easy to calibrate. Training is pointwise binary cross entropy on labeled query and ad pairs. The synthetic data hides a topic match signal, where a query and an ad are relevant when they share a hidden topic, so the model has to recover the topic from shared surface terms. Evaluation ranks the positive ad against sampled negatives per query and reports Recall@1, Recall@5, MRR, and NDCG@5. This demonstrates retrieval, relevance, and NLP query understanding in one compact module.

## Budget Pacing

The pacing module is a budget pacing simulator with a feedback controller. It lives in `src/pacing/` and runs through `scripts/run_pacing.py`. This maps to ads pacing and traffic control, where the job is to spend a daily campaign budget smoothly across the day rather than dumping it all in the morning.

The simulator synthesizes a realistic diurnal traffic curve where demand is low overnight, rises through the morning, and peaks in the early evening. It then replays a pacer against that curve slot by slot. Three pacers are compared. The PIDPacer uses a proportional integral derivative feedback loop that tracks an ideal spend trajectory and corrects when it drifts behind or ahead of plan. The AsapPacer spends as fast as possible and exhausts budget early, which is the failure mode smooth pacing exists to prevent. The ThrottlePacer is a simple proportional baseline that targets a flat per slot spend without feedback.

Each run reports budget utilization and a smoothness score, defined as the root mean squared error between the realized cumulative spend curve and the ideal traffic proportional curve. The script plots cumulative spend over the day against the ideal pacing line and saves it as a PNG. Smooth pacing matters because it avoids early budget exhaustion, keeps cost per mille stable, and gives the campaign broad time of day coverage instead of a narrow morning burst.

## Model Freshness

`scripts/run_freshness.py` reruns the data freshness experiment of He et al. (ADKDD 2014, Section 5) with the DeepFM ranker on the Taobao log. It trains one model per training day and scores each on day 8 to show how normalised entropy degrades with staleness, then compares no update, a one pass daily warm start and a full retrain by accuracy and by training compute. The protocol and its limits are in the Freshness section of `DESIGN.md`, and results go to `results/retrieval/freshness.jsonl`. Run `python scripts/run_freshness.py --source synthetic --quick` to exercise it in about half a minute without the real data.

## Quick Start

```bash
pip install -r requirements.txt

# Option A. Real Criteo data. Streams a 2 million row sample, about 1 GB.
bash scripts/download_data.sh 2000000
python scripts/run_benchmark.py --sample-size 2000000

# Option B. No download. Synthetic data with built in interactions.
python scripts/run_benchmark.py --synthetic --sample-size 100000
```

The download helper streams a public figshare mirror of the real Criteo dataset and extracts only the rows requested, so a multi million row sample costs about 1 GB instead of the full 11 GB. If no real Criteo file is present at `data/criteo.csv` the benchmark falls back to the synthetic generator automatically, so it always runs end to end. You can also point at your own file with `--data-path`.

Run the role aligned extensions on their own.

```bash
python scripts/run_relevance.py            # two tower DSSM retrieval and relevance
python scripts/run_pacing.py               # budget pacing with a PID controller
python scripts/run_inference_benchmark.py  # runtime, precision, and batch size sweep
python scripts/run_data_insights.py        # DuckDB analysis over the raw data
python scripts/serve.py                    # the scoring service
python scripts/run_load_test.py            # concurrency sweep against the service
python scripts/run_freshness.py --source taobao --device mps  # model staleness and update strategy
bash   scripts/sweep.sh                    # the whole benchmark procedure plus the gate
```

Optional extras install separately so the base install stays small. `requirements-gpu.txt` carries the CUDA and TensorRT packages, `requirements-serving.txt` the service, and `requirements-spark.txt` the distributed pipeline.

Every entry point seeds everything with seed 42 for reproducibility.

## Documentation

| Document | What it covers |
| --- | --- |
| `docs/METHODOLOGY.md` | The training half. The temporal split, why NE and GAUC are the headline metrics, the synthetic data design |
| `docs/INFERENCE.md` | The inference half. What each runtime is, precision and calibration, and the measurement methodology |
| `docs/SERVING.md` | The service, training and serving skew, the latency budget, and closed loop load generation |
| `docs/DATA_PIPELINE.md` | The Spark pipeline, how parity with pandas is guaranteed, and the partitioning choices |
| `docs/BENCHMARK_AUTOMATION.md` | The regression gate statistics, the sweep driver, and reading a profile |
| `docker/README.md` | Reproducing the GPU numbers in a pinned container, and the rented GPU flow |

## Methodology Notes

ClickRank uses a temporal positional split, not a random split. The data is time ordered, so train is the first 80 percent of rows, validation is the next 10 percent, and test is the last 10 percent. A random split would leak future impressions into training and inflate metrics in a way that never holds in production. Splitting by time matches how the model would actually be deployed, where it always predicts forward.

Normalized entropy and GAUC matter more than raw AUC for ad systems. Raw AUC measures population level ranking and is insensitive to the absolute click probability and to within request ordering. Ad pricing needs calibrated probabilities, which is what NE and logloss track, and ad auctions rank ads inside a single user request, which is what GAUC tracks. A model can win on raw AUC and still be the wrong choice for an auction. For the full reasoning on the split, on why NE and GAUC are the right headline metrics, and on the synthetic data design, see `docs/METHODOLOGY.md`.
