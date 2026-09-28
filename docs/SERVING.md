# Serving

This document explains the reasoning behind the serving half of ClickRank, the way `docs/INFERENCE.md` explains the reasoning behind the runtime comparison and `docs/METHODOLOGY.md` explains the reasoning behind the training half. The first section argues why a service is a different artifact from a benchmark table and why the project needed one. The second describes the architecture and the one bundle both lanes load. The third is backend selection and why the preference order is what it is. The fourth is training and serving skew, which is the failure this whole layer exists to prevent. The fifth states the service level objective and derives it rather than asserting it. The sixth is the load testing methodology, which is where most serving numbers quietly go wrong. The seventh is how to read a knee. The last covers the batch lane and what sharing an artifact with the online lane actually buys.

Everything here runs on CPU, was measured on an Apple M3 Pro with no accelerator of any kind, and says so. The measured numbers live in `results/serving/load_test.md` and `results/serving/batch_scoring.json`, which are generated rather than written, so this document can stay about the reasoning.

## Why a Service Rather Than a Benchmark Table

`docs/INFERENCE.md` answers how fast the model runs. It exports one graph, points several runtimes at it, and times batches in a single process with nothing else happening. That is the right way to compare runtimes and it is a real result. It is also not a serving latency, and the gap between the two is not a detail.

Three things are missing from a single process batch timing loop and every one of them is load bearing.

**There is no feature pipeline in it.** The benchmark starts from an already featurized tensor. A request does not arrive as a tensor. It arrives as raw fields, and something has to turn thirteen dense values and twenty six sparse strings into the exact numbers the model was trained on. On this model that step is not free and it is not small, and a latency number that excludes it is a latency number for half the system.

**There is no concurrency in it.** One process, one batch, one call at a time. A real service has several requests in flight, they contend for the same cores and the same interpreter, and they queue. Queueing is not a small correction on top of service time, it is the thing that produces a tail. A p99 measured with one request in the system is a p99 of the model. A p99 measured with sixty four requests in the system is a p99 of the service, and only the second one predicts whether an auction gets its answer in time.

**There is no arrival process in it.** Batches in a loop arrive exactly when the previous one finished. Requests do not. The distinction between a generator that waits for a response before sending the next request and one that sends on a schedule regardless is the difference between two tails that can be several times apart on the same service, which is why the load testing section below spends as long on it as it does.

So the claim this layer makes is narrower and more honest than the benchmark's. The benchmark says this backend runs this model at this rate. The service says this system holds this tail at this request rate under this concurrency on this machine, and here is the point past which it stops doing that.

## Architecture

Five pieces, in the order a request meets them.

**The bundle** is the deployable artifact. It is not a checkpoint. `src/serving/artifact.py` builds a manifest that binds five things together. The trained weights, the exported ONNX graph, the fitted feature pipeline, the probability calibration, and the provenance of the data all of those were fitted on. Every large member is referenced by path with its sha256 recorded rather than copied, so the graph the service runs is byte for byte the graph `scripts/run_inference_benchmark.py` timed, and a fingerprint mismatch at load time is a refusal rather than a surprise.

**The featurizer** is the persisted offline pipeline, replayed. `src/serving/features.py` writes the fitted state of the three transformers to a versioned gzipped json file and reads it back into a real `FeaturePipeline` object. The served transform is therefore not a serving side reimplementation of the offline transform. It is the offline transform, running the same code in a different process. This is the subject of its own section below.

**The backend** comes from the registry in `src/inference/backends.py`, probed in a documented preference order at startup. The serving layer does not load models or select runtimes itself. It asks the same registry the benchmark sweeps, which means a backend added there is available here for free and a backend that regressed there regressed here too.

**The engine** in `src/serving/runtime.py` is featurize, run, calibrate, in that order, with the time in each half measured separately. It is one object and both lanes construct it.

**The app** in `src/serving/app.py` is three endpoints. `/score` takes one auction and returns it ranked. `/health` reports the selected backend, every backend that was skipped and why, the hardware, and the bundle fingerprints. `/metrics` exposes counters and three latency histograms in the Prometheus text format.

One request is one auction. That is a deliberate shape rather than an arbitrary one. An ad request opens an auction over a candidate set that retrieval has already narrowed, and the ranker's job is to score that whole set and order it. Making the request the auction means the batch dimension of the model call is the candidate set size, which is a quantity a capacity plan can reason about, rather than an arbitrary number chosen by whatever batching layer happened to sit in front. It also means the shared context fields go over the wire once rather than being duplicated onto every candidate, which is how a real bidder sends them.

The scoring handler is a synchronous function rather than an async one, which is deliberate. Scoring is CPU bound. Running it directly on the event loop would block every other request behind it and turn the service into a serial queue wearing an async interface. Declaring it synchronous hands it to the Starlette thread pool, so requests genuinely queue and genuinely overlap wherever the work releases the interpreter lock. The thread pool size is set explicitly and reported on `/health`, because it is the queueing parameter of the service and an operator has to be able to see it.

The model call itself is taken under a lock. The registry's run functions make no thread safety promise and at least one of them cannot, since the OpenVINO compiled model shares a single default inference request across calls and two threads entering it concurrently is a data race rather than parallelism. Serializing the model call is the correct conservative choice. It costs less than it appears to, because the feature transform runs outside the lock and is where most of the wall clock goes. Real horizontal scaling comes from worker processes, which is what `--workers` and the compose file provide.

## Backend Selection and Why That Order

The order is fastest first, on the measured evidence in `docs/INFERENCE.md` rather than on reputation.

1. **A natively built TensorRT engine.** TensorRT picks each layer's kernel by benchmarking candidates on the actual device and emits a plan specific to that GPU. Where an engine exists it is the ceiling for this model on that hardware.
2. **The ONNX Runtime CUDA execution provider.** A GPU path that needs no prebuilt engine. It is the fallback when the engine for this machine was never built, or was built by a different TensorRT version, since engines are not portable across either.
3. **OpenVINO on the CPU.** It sits above the ONNX Runtime CPU provider because on the Apple M3 Pro this project is developed on it measured roughly 1.47 times faster per batch than eager PyTorch with a far tighter tail, while the ONNX Runtime CPU provider did not beat eager PyTorch at all. That ordering is a measurement on one machine and may well invert on an Intel x86 serving fleet where the MLAS kernels are at home, which is exactly why the selection is a startup probe rather than a constant in a config file.
4. **The ONNX Runtime CPU provider.** The portable CPU path.
5. **Eager PyTorch.** The accuracy reference and the last resort. It is the one backend that cannot be unavailable, because it needs no export and no extra package.

Two rules govern the selection and both exist so that an operator is never wrong about what is running.

**Every skip is reported.** The probe records a reason for every backend it could not construct and serves the whole table on `/health`. A service that quietly fell back to eager PyTorch because an export was missing looks identical from the outside to one running a compiled engine, right up until somebody reads the latency graph and cannot explain it. This is the online form of the honesty rule in `docs/INFERENCE.md` that a backend which did not run says so rather than disappearing.

**Reduced precision is opt in.** FP16 and INT8 change the model's output. `docs/INFERENCE.md` argues at length that INT8 in particular may cost real AUC on a DLRM shaped model, because the accuracy lives in embedding tables and the FM interaction term is a difference of squared sums over exactly those embeddings, which is a numerically unforgiving form. Silently selecting a faster backend that returns different probabilities than the offline evaluation measured would be the accuracy equivalent of training and serving skew. So the default order is FP32 only, the reduced precision lane is behind a flag, and selecting it prints the tradeoff at startup.

## Training and Serving Skew

This is the part most ranking projects get wrong, and it is the reason the serving layer is structured the way it is rather than as a thin wrapper around a checkpoint.

A trained model is not the only thing that was fitted. The feature pipeline was fitted too, on the train split only, and it learned three sets of parameters. The numerical transformer learned a mean and a standard deviation per dense column, computed on the log transformed clipped values with missing cells excluded so the fill value does not drag the mean. The categorical encoder learned a value count table per sparse field, which decides both which values are rare enough to collapse into the shared rare token before hashing and what frequency encoding each value receives. The cross generator learned which categorical fields to cross, by ranking every field on the variance of its training value frequency distribution and taking the top few.

None of that is in the checkpoint. And here is what makes it dangerous rather than merely inconvenient. A checkpoint loaded against a freshly constructed pipeline still loads, because the tensor shapes in this model depend on the hash bucket sizes and the field counts rather than on the data. It still runs. It still returns numbers between zero and one that look exactly like click probabilities. What it cannot do is rank, because every embedding lookup is now reading a row that belongs to a different category than the one the request actually carried. Nothing in the response says so. The failure surfaces days later as an unexplained gap between an offline metric and an online one, and by then nobody remembers which of a dozen changes caused it.

Two other versions of the same mistake are worth naming because they are subtler.

**Refitting the pipeline on serving data.** Tempting, because the code to fit is right there. It is wrong because the frequency table and the standardization statistics would then be computed on a different population than the weights were trained against, and the drift would grow silently as traffic changed.

**Reimplementing the transform in the service.** Also tempting, because the offline transform loops in Python and a service could vectorize it. It is wrong because two implementations of one transform diverge. Not immediately, but at the first bug fix that lands on one side, or the first edge case in the string coercion, or the first change to how a missing value is filled. The gap is invisible in code review and shows up only as a metric that moved for no reason.

The fix implemented here is to treat the fitted pipeline as a shipped artifact with the same status as the weights, and to replay it rather than reimplement it. `src/serving/features.py` extracts the fitted state, writes it to a versioned file, and reads it back into a real `FeaturePipeline`. The service then calls that object's own `transform` method. Nothing in `src/data` is modified. The persistence layer sits around the pipeline and reaches into the three transformers by attribute, which is the right dependency direction, since the serving layer is the thing that has a reason to change.

One reduction happens on write and it is exactly behaviour preserving rather than approximately. Categorical values whose train count falls below `min_count` are dropped from the saved table. The encoder folds any value with a count below `min_count` into the shared rare token before hashing and assigns it a frequency of zero, and a value absent from the table reads back as count zero, which is also below `min_count`, so it takes the identical branch. The saved table is smaller and no output value changes. That is an argument, not a proof, so `tests/test_serving.py` transforms the same rows with the unpruned encoder and the pruned one and requires exact equality of both the hashed indices and the frequency encodings.

The parity test is the one that matters. It takes rows out of the offline test split, converts them into the dict per candidate shape a caller actually sends including a json null wherever a dense value is missing, pushes them through the served featurizer, and requires the resulting arrays to equal the offline pipeline's arrays exactly. Exact, not close. A hash bucket index has no tolerance, because a different bucket is a different embedding row rather than a slightly different number, and the dense block is compared exactly too, because both paths run the same arithmetic on the same inputs and any difference at all would mean they did not.

There is a matching failure from the model side and the bundle catches it as well. If the checkpoint changes under a bundle whose feature statistics were fitted against the old weights, the two halves no longer belong to each other. That is why the manifest records a sha256 for every member and why loading verifies them. The bundle also records its own validation AUC at build time, computed in eager PyTorch, which is a tripwire rather than a metric. A bundle whose weights and pipeline were fitted on different data will still load and still return probabilities, but its validation AUC will sit near chance, and it is much better to see that in a manifest than in a revenue graph.

## The Service Level Objective

Throughput with no latency bound is not a serving result. The report grades against a stated budget, and the budget is derived rather than asserted.

A page render budget of a few hundred milliseconds leaves the ad request roughly one hundred milliseconds end to end. Out of that, the exchange call and the network hops take their share, retrieval narrows millions of candidates to a shortlist and takes its share, and the auction, pricing, and creative selection take theirs. What is left for the ranking model is a slice measured in tens of milliseconds rather than in hundreds. `docs/INFERENCE.md` reaches the same place from the other direction when it says an auction budget is usually measured in tens of milliseconds for the whole thing, of which the model gets a slice.

**The objective is a p99 of 25 ms end to end for one auction.** End to end at the client, so it includes json handling, the network hop, the queue inside the service, the feature pipeline, and the model call. Not the model call alone, because the caller does not experience the model call alone.

Three choices inside that are worth defending. It is a **p99 rather than a mean**, for the reason `docs/INFERENCE.md` gives for the offline case and which is sharper online. A request in a real ad system fans out across components and the slowest one sets the response time, so a service with a low mean and a fat tail is worse in production than one with a slightly higher mean and a tight distribution. A mean hides that by construction. It is **25 rather than 10 or 50** because it should be tight enough that missing it is a real engineering problem and loose enough that meeting it is not automatic. And it is **stated before the numbers are read**, which is the only way a budget means anything. The load test constant lives in `scripts/run_load_test.py` and the report grades against it whether or not the service passes.

The number the report leads with is therefore not peak throughput. It is the highest sustainable request rate that held the p99 under the budget with no errors, and the concurrency it was achieved at. That single figure is what a capacity plan multiplies by a machine count.

## Load Testing Methodology

**Closed loop, and it says so.** A fixed number of virtual clients each send one request, wait for the response, and immediately send the next. The offered load is therefore a consequence of the service's own speed rather than an independent variable. That is the right model for a caller with a bounded connection pool, which is what an upstream ad server is. An open loop generator fires at a fixed rate whatever the service is doing and queues locally when it falls behind, which measures a different and usually much worse tail. The two are not interchangeable, they produce different numbers from the same service, and reporting one as the other is a common and completely invisible error. This one is closed loop.

**Coordinated omission is stated, not corrected.** A closed loop generator cannot issue the next request until the previous one returns. So when a response is slow, the requests that would have been sent during it are never sent, and the latencies that never got sampled are precisely the ones that would have been worst. The measured tail is therefore a floor on the real tail rather than an estimate of it. Correcting for this properly requires an open loop generator with an intended send schedule and a latency measured from that schedule rather than from the send, which is a different tool than this one. What is done instead is to name the bias, hold the client count fixed and known so the bias is at least consistent across cells, and report the achieved concurrency next to the requested one, computed from throughput and mean latency, so that a cell where the clients rather than the service were the bottleneck is visible in the json.

**Warmup is run and discarded.** The first calls into any runtime pay for lazy kernel selection, allocator growth, first touch page faults, and on this service the first pandas and numpy code paths as well. `docs/INFERENCE.md` is explicit that including those measures startup rather than serving. The backend is warmed at several batch sizes when the service starts, before it reports healthy, and each cell of the sweep runs its own warmup phase whose results are thrown away.

**Cells are timed rather than counted.** Every cell runs for a fixed wall time rather than a fixed number of requests. A fast cell therefore rests its percentiles on more samples rather than on a longer window, and a slow cell does not silently take ten times as long to finish.

**The sweep has two axes.** Concurrency, from one to two hundred and fifty six, and candidate set size per request. Both matter and they do not answer the same question. Concurrency is the queueing axis and it is what produces the tail. Candidate set size is the work per request axis and it is what an auction shortlist actually looks like after retrieval. A service can hold a budget comfortably at sixteen candidates and blow through it at two hundred and fifty six.

**The host load average is recorded.** A latency measurement taken while the machine is running something else is not a measurement of the service, and the only way a reader can tell is if the number is on the page. It is in the report header.

**Server side timing rides along.** Every response carries the feature time and the model time the service measured internally, and the report averages them per cell. That is what makes the end to end number diagnosable rather than merely true. The two do not sum to the end to end latency, and the difference between them is the queueing, the json, and the loopback hop, which is exactly the part a single process benchmark cannot see.

## Reading the Knee

Peak throughput is a stress test answer. It tells you what the service does when it is already past the point where anyone would want to run it.

The number a capacity plan is built on is the knee, which is the concurrency past which the tail degrades faster than the throughput improves. Below it, adding load buys work. Above it, adding load buys queue depth, and the extra requests in the system are waiting rather than being served. The distinction between finding that point and finding the maximum is the difference between a load test and a stress test.

The rule used here is a ratio of ratios and it is deliberately local. Stepping from one concurrency level to the next multiplies throughput by some gain and multiplies p99 by some cost. While the gain exceeds the cost, the extra load is productive. The first step where the cost exceeds the gain marks the knee at the level before it. The report prints the gain, the cost, and their ratio for every step, so the judgement is visible rather than buried in a single number.

It is a local rule rather than a fitted scalability curve on purpose. Fitting a model to five points would produce a smooth answer that looks more authoritative than five points deserve, and the interesting thing about a knee is where the discrete steps stop paying rather than what a curve says between them.

**When the sweep does not reach a knee, the report says so.** A sweep that runs out of concurrency levels before the efficiency drops below one has found headroom rather than an operating point, and the honest conclusion is that the sweep needs extending. Inventing a knee from a curve that never bent would be exactly the kind of number this project's honesty rules exist to prevent.

## The Batch Lane

The other half of a decisioning pipeline. `scripts/run_batch_scoring.py` reads a Parquet or CSV shard, scores it, and writes a joinable file with an identifier, a probability, and the label when the shard carried one. It reports rows per second and total wall time, split into the feature share and the model share, over the whole job rather than over the fastest chunk. Chunking is there to bound memory, since a shard can be larger than memory once it has been featurized into dense blocks, and not to make anything faster.

The point of the batch lane is not that it exists, it is what it does not contain. There is no featurizer in it, no model loader, and no calibration. It loads the same bundle the service loads, probes backends through the same registry in the same preference order, constructs the same `ScoringEngine`, and calls the same method the http handler calls. The only differences between the two lanes are the batch size and where the rows come from.

That matters because the classic way an offline number and an online number come to disagree is that they were produced by two pieces of code that were supposed to be the same and drifted. A feature clipped in one and not the other. A default fill that changed on one side. A categorical map rebuilt from a different day of data. Every one of those is invisible in code review and obvious only in a metric that moved for no reason. Sharing one artifact and one code path removes the opportunity rather than adding a check for it.

The check is still there. `tests/test_serving.py` sends the same rows down both lanes and requires the probabilities to be identical rather than close, because both lanes run the same weights through the same runtime on the same featurized arrays, and any difference at all would mean one of those three was not actually shared.

## What the Measurement Showed

The numbers live in `results/serving/load_test.md` and `results/serving/batch_scoring.json`, which are generated by the two scripts and carry the hardware, the backend, the configuration, and the host load average that produced them. Three findings are worth stating here because they are about the system rather than about one run.

**The feature pipeline is the serving bottleneck on this model, not the model.** At one client and a single candidate, the service spends several milliseconds in the feature transform and roughly a millisecond in the model call. The ratio is the finding rather than the absolute values, and it is stable because both halves are measured inside the same process on the same request. That is the opposite of what the inference benchmark alone would lead a reader to expect, and it is exactly the thing a single process batch timing loop cannot see, because it starts from an already featurized tensor.

The mechanism is legible. The transform hashes twenty six categorical fields and ten crosses per row through md5, looks every value up in a train frequency table, and does it in Python. Most of the cost is not even proportional to the candidate set. It is fixed per request, because the transform builds a DataFrame and makes a few dozen pandas calls whose overhead does not care how many rows are in it. That is why the per request latency rises so much more slowly than the candidate count does, and why a candidate set of sixteen costs barely more than a candidate set of one.

The consequence for a real deployment is a clear ordering of what to optimize. Moving the ranker to a faster runtime is worth a fraction of a millisecond here. Vectorizing the categorical hashing, or precomputing the frequency lookups into arrays, is worth several. Anyone who read only the runtime benchmark would have optimized the wrong half of the system, which is the entire argument for building the service rather than stopping at the table.

**Throughput does not scale with concurrency on a single worker, and it should not be expected to.** Scoring is CPU bound and holds the interpreter lock through the feature transform, and the model call is serialized behind a lock because the registry's run functions make no thread safety promise. Adding clients past the thread pool size therefore adds queue depth rather than work. The knee is early and the shape of the curve says so. The lever that actually scales this service is worker processes, which is what `--workers` and the compose file provide, and the per worker numbers here are what a capacity plan multiplies.

**The batch lane amortizes what the online lane cannot.** The same engine scoring a shard in chunks of four thousand rows reaches a rows per second figure orders of magnitude above the online lane's ad scores per second, because the fixed per request cost that dominates a single auction is paid once per chunk instead of once per request. The batch report splits its wall clock into the feature share, the model share, and the shard io share for the same reason the online report splits its latency, which is that a throughput figure with no breakdown cannot tell anyone what to fix.

**A caveat that applies to every absolute number in the current report.** The machine these measurements were taken on was running other heavy workloads at the time, at a one minute load average several times its core count. The load test records that load average at the start of the sweep and again for every individual cell, and the report prints a banner saying so. The ratios inside a single request, such as feature time against model time, are unaffected because both are measured in the same process on the same request. The absolute latencies and the throughput figures are inflated, and the shape of the concurrency curve is distorted by contention that has nothing to do with the service. Those numbers are upper bounds and the report says to rerun on an idle machine before quoting them. Reporting them with the banner is the honest option. Deleting the banner and quoting them would not be, and neither would quietly reporting the one cell that happened to look best.

## Running It

```bash
pip install -r requirements.txt
pip install -r requirements-serving.txt

# Build the bundle once. It fits the pipeline, loads or trains the checkpoint,
# exports the graph, fits the calibration, and writes the manifest.
python scripts/serve.py --build-only

# Serve.
python scripts/serve.py

# Load test. It uses a service already running at the url, and starts one
# itself when nothing is answering there.
python scripts/run_load_test.py

# Batch score a shard through the same bundle.
python scripts/run_batch_scoring.py
```

In a container, `docker/Dockerfile.serving` builds the CPU serving image and `docker/serving-compose.yml` wires the bundle build, the service, the load test, and the batch job as four commands against one image. The Dockerfile carries a comment describing the one argument change and the one dependency change that turn it into a GPU serving image on the TensorRT base the benchmark lane already pins.
