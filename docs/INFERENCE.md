# Inference Optimization

This document explains the reasoning behind the inference half of ClickRank, the way `docs/METHODOLOGY.md` explains the reasoning behind the training half. The first section argues why a runtime comparison is the right artifact for an ad ranking project at all. The second says what each runtime actually is, because the names get used loosely. The third covers precision and quantization, including why the calibration set has to come from validation. The fourth is the measurement methodology, which is where most inference benchmarks quietly go wrong. The fifth states the honesty rules as rules. The sixth writes down a prediction about this specific model before the numbers come in, so the numbers can confirm it or kill it. The last documents the sweep configuration schema.

Everything here is seeded with seed 42 and runs on the same temporal split and the same trained weights as the main benchmark, so the only thing that varies across a row of the table is the thing being compared.

## Why an Inference Runtime Comparison Is the Right Artifact

A ranking model that cannot be served is a research result, not a product. In an ad system the ranker sits on the request path. A user loads a page, an auction opens, a few hundred to a few thousand candidate ads have to be scored, and the winner has to come back inside a budget that is usually measured in tens of milliseconds for the whole auction, of which the model gets a slice. Miss the budget and the request times out and the impression is lost, which costs more revenue than a slightly worse model would.

That makes serving latency a first class model property rather than an implementation detail. The training benchmark in this repository answers which architecture ranks best. It does not answer whether the winner can be served, or what it costs to serve, or what the cheapest hardware is that meets the budget. Those are separate questions with separate answers, and a model comparison that ignores them can recommend a model that no one can deploy.

The comparison is also the honest way to size an optimization. Claims about inference speedups circulate as folklore, usually as a single multiplier detached from the model, the batch size, and the hardware. The only way to know what a runtime is worth for a particular model is to hold the weights fixed, hold the input rows fixed, hold the hardware fixed, and vary one thing. That is what this benchmark does. Every backend runs the same exported graph over the same held out test rows, and the AUC column exists so that a runtime which changed the numbers cannot hide.

There is a second reason specific to this model family. DLRM style recommenders are unusual. They are not the dense convolutional or transformer workloads that inference runtimes are tuned and marketed against. Most of the parameters sit in embedding tables that are read sparsely, and the arithmetic that follows is a small multilayer perceptron. Whether the standard optimization playbook transfers to that shape is a real open question rather than a foregone conclusion, which makes it worth measuring rather than assuming.

## What Each Runtime Actually Is

**PyTorch eager mode** is the reference. It executes the model one operator at a time from Python, dispatching each call into ATen kernels as it goes. Nothing is fused, nothing is compiled ahead of time, and the Python interpreter is in the loop for every operation. This is what the training code runs, so it is the honest starting point, and it is also the number people accidentally publish as their inference latency when they have not done the export step. Every other row is measured against it.

**ONNX Runtime** is the open standard graph runtime. The model is exported once into an ONNX graph, which is a framework independent description of the computation and the weights. ONNX Runtime loads that graph, applies graph level optimizations such as constant folding, operator fusion, and dead node elimination, and then dispatches the optimized graph to an execution provider. The execution provider is the part that actually runs kernels, and the same ONNX graph can be pointed at several of them.

The **CPU execution provider** runs ONNX Runtime's own MLAS kernels. It is the default and it is what the current CPU numbers in the README use. The **CUDA execution provider** runs cuDNN and cuBLAS kernels on an NVIDIA GPU, with ONNX Runtime still owning the graph, the memory, and the scheduling. The **TensorRT execution provider** is different in kind. It hands whole subgraphs to TensorRT, which compiles them into an engine, and then ONNX Runtime calls that engine for those subgraphs and runs anything TensorRT could not take on its own kernels. That fallback is the thing to watch. A graph that partitions badly ends up with dozens of small TensorRT subgraphs separated by ONNX Runtime nodes, and the cost of crossing back and forth eats the speedup. The number of partitions is worth logging next to the latency.

**Native TensorRT** skips ONNX Runtime entirely. TensorRT reads the ONNX graph, runs its own builder over it, and emits a serialized engine. The builder does the heavy work. It fuses operators, it picks kernels by benchmarking several candidate implementations of each layer on the actual target GPU, it selects tensor layouts and formats, and it lays out the memory the engine will need. The result is a plan that is specific to the TensorRT version, the GPU architecture, and often the exact GPU. That specificity is where the speed comes from and it is also why engines are not portable and why the version pin in `docker/Dockerfile.tensorrt` is not decoration. The native path is the ceiling for what TensorRT can do on this model, and comparing it against the TensorRT execution provider is what reveals how much the ONNX Runtime partitioning is costing.

**OpenVINO** is Intel's inference toolkit. It reads the same ONNX graph, compiles it for the host CPU, and runs the compiled network. It is the CPU lane. It is in the comparison because a large fraction of real ad serving runs on CPU fleets, where the economics are about queries per second per dollar rather than about the lowest achievable latency, and because it is the current winner on the machine this project was developed on.

## Precision

Precision is the numeric format the weights and activations are stored and computed in. FP32 is the 32 bit float the model was trained in and it is the correctness reference. FP16 is a 16 bit float with a much narrower exponent range. INT8 is an 8 bit integer with a scale factor per tensor or per channel that maps the integer range onto the real range the values actually occupy.

Lowering precision buys two different things and it is worth keeping them apart. The first is arithmetic throughput. Tensor cores execute FP16 and INT8 matrix operations at a much higher rate than FP32, so compute bound layers get faster. The second is memory traffic. A tensor at half the width takes half the bandwidth to move and half the cache to hold, so bandwidth bound layers get faster even when no faster arithmetic unit is involved. Which of the two dominates depends entirely on the model, which is the subject of the prediction section below.

FP16 conversion is essentially free to apply. The weights are cast, the runtime keeps sensitive reductions in FP32 where it needs to, and no extra data is required. The risk is overflow and underflow in the narrower exponent range rather than a loss of precision as such.

INT8 is a different operation. **Post training quantization** takes a model that was trained in FP32 and converts it to INT8 after the fact, with no retraining and no gradient step. To do that the runtime needs to know, for every tensor it intends to quantize, what range of real values that tensor actually takes, because that range is what the 256 available integer levels get mapped onto. Weights are static, so their range can be read straight off the tensor. Activations are not. Their range depends on the input data and cannot be known without running data through the network.

That is why INT8 needs a **calibration set**. The builder runs a few hundred batches of representative data through the network in FP32, records the distribution of every activation tensor, and picks a clipping range for each one. Entropy calibration, which is the TensorRT default, chooses the range that minimizes the information lost between the full precision distribution and the quantized one, which usually means clipping a thin tail on purpose rather than stretching the scale to cover an outlier and wasting most of the levels on values that almost never occur. Min max calibration takes the observed extremes instead, which is safer against clipping and worse against outliers.

**The calibration set must come from validation data and never from the test split.** This is the same leakage rule that governs the temporal split in `docs/METHODOLOGY.md`, applied one stage later. Calibration is a fitting procedure. It reads data and it produces parameters, the per tensor scales, which change the model's outputs. A scale fitted on the test split is a parameter fitted on the evaluation set, and the test AUC that comes out is then reporting how well the model does on data it was partly tuned on. The effect is subtle, which is what makes it dangerous. It does not produce an obviously broken number, it produces an INT8 accuracy loss that looks smaller than it really is, which is exactly the number the benchmark exists to measure honestly. The validation split sits between train and test in time and is already the split used for early stopping, so it is the natural source and it keeps the test rows untouched by anything except the final scoring pass.

The calibration cache is worth keeping. It is a small file of per tensor ranges, it is deterministic given the same calibration rows and the same algorithm, and caching it means a rebuild does not have to redo the calibration pass. It is also the artifact to inspect when an INT8 result is bad, because a tensor whose range is wildly wider than its neighbours is usually the layer that lost the accuracy.

## Benchmarking Methodology

**Warmup is not optional.** The first few calls into any runtime pay for things that a steady state serving process pays once and then never again. Lazy kernel selection, CUDA context creation, JIT compilation of the first kernel launch, allocator growth, page faults on first touch of a buffer, and on a GPU the clocks ramping up from idle. Including those in the timing measures startup, not serving. The benchmark runs a fixed number of warmup batches through each backend and discards them, then times the real passes. The GPU lane uses more warmup batches than the CPU lane because clock ramp is slower than allocator warmup.

**GPU timings need an explicit synchronize.** CUDA work is asynchronous. Launching a kernel puts it on a stream and returns to the host almost immediately, long before the GPU has finished. A host side timer that stops right after the launch measures how long it took to enqueue the work, which on a small model is a fraction of the real time and produces latency numbers that look spectacular and are fiction. The fix is to synchronize the device or the stream before stopping the clock, so the timer covers the work rather than the launch. The alternative is CUDA events recorded on the stream, which measure device time directly and avoid host side scheduling noise. Either is correct. Neither can be skipped. A GPU inference benchmark with no synchronization in it is measuring the wrong quantity, and it is the single most common way these tables end up wrong.

**p99 matters more than mean for serving.** A serving system is not judged by its average request. It is judged by its slow requests, because those are the ones that miss the deadline, and because a request in a real ad system fans out to many components where the slowest one sets the response time. Mean latency hides the tail by construction. A backend with a low mean and a fat tail is worse in production than one with a slightly higher mean and a tight distribution, and only the p99 column shows that. The current CPU table already contains an example, where ONNX Runtime's p99 is several times its mean while OpenVINO's tail sits close to its mean. Both mean and p99 are reported, and the p99 is the one to argue about.

**Batch size 1 and batch size 4096 are different questions.** Batch size 1 is the single request question. One user, one auction, score it now. At that size the model barely occupies the hardware, and what dominates is fixed per call overhead such as the Python dispatch, the kernel launch, the host to device copy, and the synchronize. This is where graph compilation and kernel fusion pay off most, because the win comes from removing overhead rather than from doing arithmetic faster. Batch size 4096 is the throughput question. Offline scoring, batch pipelines, or a serving path that groups the candidate ads of one auction into a single call. At that size the fixed overhead amortizes to nothing and what dominates is memory bandwidth and arithmetic throughput. A runtime can win decisively at one end and lose at the other, so a single batch size cannot answer both and the sweep covers both ends plus points in between.

**Engine build time is reported separately from inference time.** Building a TensorRT engine is expensive, because the builder benchmarks candidate kernels on the real device to pick the fastest one, and it can take minutes. It is also a one time cost that a real deployment pays at build time or at container start, not per request. Folding it into a latency number would be wrong in the obvious direction, and hiding it entirely would be wrong in the other, because it is a real operational cost that shapes how often a model can be shipped and how a rollback works. So it gets its own column and its own sentence. The same applies to the INT8 calibration pass, which is part of the build and not part of serving.

## The Honesty Rules

These are rules rather than guidelines, and they exist because inference benchmarks are unusually easy to make flattering without technically lying.

**Every number is labeled with the hardware that produced it.** No latency, throughput, or speedup figure appears in a table, a chart, or a sentence without the device, and where it matters the driver and the runtime version, attached to it. The sweep configs carry a `hardware` field for exactly this reason and it is stamped onto every row rather than written once at the top of a document and then forgotten.

**The 680K predictions per second figure is an Apple Silicon CPU number and it is never blended with a GPU number.** That figure comes from OpenVINO on an M3 Pro. It belongs in the CPU table and it stays there. It is never averaged with a GPU result, never used as the baseline that a GPU speedup is computed against, and never quoted in a sentence that also contains a GPU number without both hardware labels present. Comparing a laptop CPU against a data center GPU is not a comparison of anything.

**OpenVINO is the CPU lane and it is not a like for like comparison against the CUDA runtimes.** It stays in the sweep because CPU serving is real and because it is the current winner on the development machine, but it is reported as its own lane. A table that sorts OpenVINO and TensorRT into one ranked list is comparing hardware, not runtimes, whatever the column headers say.

**Accuracy loss from INT8 is reported even when it is bad, and especially when it is bad.** Every INT8 row carries its AUC and its delta against the FP32 row of the same runtime. If INT8 costs real AUC on this model, that is the finding and it gets written down in the same font as the speedup. A quantization result that reports the latency win and omits the accuracy column is not a result. If the loss is large enough that the configuration would never ship, the row still appears and the sentence says so.

**A backend that did not run says so rather than disappearing.** The existing report already does this, marking missing backends as not installed rather than silently dropping them from the table, so a reader can tell the difference between a backend that lost and a backend that was never tried.

## The Prediction Under Test

Writing the prediction down before the numbers arrive is the point. A prediction that can only be stated after the fact explains nothing.

**The prediction is that this model is memory bandwidth bound rather than compute bound, and that the usual precision speedups will therefore underdeliver.**

The mechanism is the shape of a DLRM style recommender. Nearly all of the parameters live in the embedding tables. The DeepFM checkpoint in this repository is about 21.6 million parameters and roughly 86 MB on disk, and the overwhelming majority of that is embedding weight. What happens at inference is that a batch of rows arrives, each row names one index per categorical field, and the model gathers one embedding row per field. Those gathers are scattered reads into a large table with almost no locality, because consecutive rows in a batch touch unrelated categories. Then the gathered vectors are concatenated with the dense features and pushed through a small multilayer perceptron whose hidden layers are a few hundred units wide.

The arithmetic in that MLP is tiny. A few matrix multiplies of a few hundred by a few hundred, per batch. On a modern GPU that is a rounding error of the available floating point throughput. The gather is not tiny. It moves a large number of cache unfriendly bytes and it cannot be fused away, because it is a data movement operation and not an arithmetic one. So the wall clock is expected to be dominated by getting embedding rows out of memory, and the arithmetic units are expected to be mostly idle waiting for them.

Two consequences follow. First, FP16 should give far less than the usual 2x. The tensor core speedup applies to the MLP, which is not where the time goes. FP16 does halve the bytes moved for the embedding gather, so it should give something rather than nothing, but the something should look more like a modest fraction than a doubling. Second, INT8 may cost real AUC for less benefit than usual. Embedding values are the model's learned representation of a category and they are what the interaction terms are computed from. Squeezing them into 256 levels per tensor is a blunt operation on the exact quantity the model's accuracy depends on, and the FM interaction term in DeepFM is a difference of squared sums over those embeddings, which is a numerically sensitive form where small representation errors do not stay small. Meanwhile the compute win that INT8 would normally buy lands on the part of the network that was never the bottleneck.

The evidence that would confirm this is a set of specific, falsifiable observations. FP16 delivering well under 2x at large batch sizes while a compute bound reference model on the same GPU delivers close to 2x. GPU utilization staying low while memory throughput stays high during a profiled run, which is directly readable from Nsight Compute or from NVML. Latency scaling close to linearly with batch size once the fixed overhead has amortized, which is the signature of a bandwidth bound kernel, rather than staying nearly flat as batch size grows, which is the signature of a compute bound kernel with idle capacity. Shrinking the embedding dimension moving the latency roughly proportionally while widening the MLP hidden layers barely moving it at all. And the INT8 AUC delta being noticeably worse than what the same quantization does to a dense model of similar parameter count.

The evidence that would refute it is the opposite. FP16 delivering close to the full arithmetic speedup, latency staying nearly flat as batch size grows, high compute utilization in a profile, or INT8 turning out to be nearly free in AUC. Any of those would mean the MLP is a bigger share of the work than this argument assumes, or that the gather is being served out of cache more effectively than expected because the hash space is capped at 10000 buckets per field, which is small enough that a large fraction of the table may be resident in cache. That last possibility is worth naming explicitly, because it is the most likely way this prediction is wrong on this particular configuration while still being right on a production sized table with millions of rows per field.

## Sweep Configuration Schema

The sweep lives in YAML files under `benchmarks/` rather than in a growing pile of command line flags. The reasons are that a config is a record of exactly what was run and can be committed next to the results it produced, that a matrix of four dimensions is unreadable as flags, and that the exclusions which keep meaningless cells out of the product need somewhere to live with a comment explaining each one.

The schema is deliberately thin. It is a description of a run, not an execution engine, and every key maps onto a flag that already exists.

### What is wired and what is not

The benchmark script is being written in parallel with these configs. As of this document, `scripts/run_inference_benchmark.py` is driven by command line flags and does not yet read a YAML file. The configs are therefore the source of truth for what a run should be, and they are translated to flags either by the `Makefile` targets or by hand. When the script grows a config reader the mapping below is the contract it should implement. The runtime identifier strings in `matrix.runtimes` are the part most likely to need reconciling, since they have to match whatever `--runtimes` accepts.

### Top level keys

| Key | Type | Meaning |
| --- | --- | --- |
| `name` | string | Short identifier for the sweep. Also the natural name for its output directory. |
| `description` | string | What this sweep is for, in a sentence or two. |
| `hardware` | string | Free text label for the machine. Stamped onto every row this config produces. Required by the honesty rules. |
| `data` | map | Where the rows come from. |
| `run` | map | Timing and output settings that do not vary across the matrix. |
| `matrix` | map | The four lists whose cartesian product is the sweep. |
| `exclude` | list of maps | Cells to drop from that product. |
| `calibration` | map | INT8 settings. Read only when `int8` appears in `matrix.precisions`. |
| `notes` | string | Anything a reader of the results needs to know. |

### Mapping to flags

Every key under `data` and `run` maps to the flag of the same name with underscores turned into dashes. Every key under `matrix` maps to the comma separated list flag of the same name.

| Config key | Flag |
| --- | --- |
| `data.data_path` | `--data-path` |
| `data.sample_size` | `--sample-size` |
| `data.synthetic` | `--synthetic`, passed only when true since it is a store true flag |
| `run.checkpoint` | `--checkpoint` |
| `run.output` | `--output` |
| `run.repeats` | `--repeats` |
| `run.warmup` | `--warmup` |
| `run.batch_size` | `--batch-size` |
| `matrix.models` | `--models` |
| `matrix.runtimes` | `--runtimes` |
| `matrix.precisions` | `--precisions` |
| `matrix.batch_sizes` | `--batch-sizes` |

`run.batch_size` and `matrix.batch_sizes` both exist because the underlying script has both flags. The singular one is the default for a single run and the plural one is the sweep. When `matrix.batch_sizes` is present it is what the sweep uses and the singular value is only the fallback.

### Exclusion rules

An entry in `exclude` is a map of dimension names to values. A cell of the product is dropped when every key in the rule matches that cell, so a rule with one key drops a whole plane and a rule with three keys drops one cell. The dimension names are the singular forms of the matrix keys, which are `model`, `runtime`, `precision`, and `batch_size`. This is the whole rule language and it is meant to stay that way. Anything more expressive belongs in a second config file rather than in a query syntax.

For example, `{runtime: openvino, precision: fp16}` drops every OpenVINO FP16 cell at every model and batch size, which is correct because OpenVINO is the CPU lane and a GPU half precision row there would be meaningless.

### The two shipped configs

`benchmarks/cpu_only.yaml` reproduces what runs today on Apple Silicon. One model, three CPU runtimes, FP32 only, swept across four batch sizes. It needs nothing beyond `requirements.txt`.

`benchmarks/gpu_full.yaml` is the full CUDA sweep. One model, five runtimes, three precisions, five batch sizes, with the twenty five meaningless cells excluded and the reasons written next to each exclusion. It expects the pinned container in `docker/Dockerfile.tensorrt` and it expects the engines to have been built first, so that the build cost is measured on its own rather than inside a timing run.
