# Benchmark Automation and Profiling

`docs/METHODOLOGY.md` explains the training half of ClickRank and
`docs/INFERENCE.md` explains the inference half. This document explains the
layer that sits around both of them, which is the automation that runs the
benchmark and the tooling that gates and profiles the result.

The first section argues why a benchmark needs a gate at all, since a benchmark
is usually treated as a report rather than as a test. The second says what a
cell is and why comparison happens at that granularity. The third is the
statistics, including the ones that were rejected and why, and it is honest
about how little power a gate has over a handful of samples. The fourth says why
an accuracy regression outranks a latency regression. The fifth says why a
baseline belongs to one machine. The sixth covers the sweep driver and how to
add a configuration. The seventh and eighth cover the two Nsight tools, what
each one shows, and what a timeline of this specific model looks like when the
prediction in `docs/INFERENCE.md` is right and when it is wrong. The last states
plainly what has been executed and what has not.

## Why a Benchmark Needs a Gate

A benchmark produces a table. A table is a claim about the past. It says that on
some day, on some machine, with some set of installed versions, these were the
numbers. Nothing about publishing that table causes it to stay true.

What makes it stop being true is ordinary work. A dependency bump changes a
kernel. A refactor of the feature pipeline changes the shape of a tensor and the
runtime silently picks a different implementation. An export change moves an
operation from the fused path to the unfused one. A container rebuild pulls a
newer ONNX Runtime whose execution provider list no longer includes the one that
was doing the work. None of these announce themselves. They are all invisible
until somebody reruns the benchmark and compares against a number they remember,
which is not a process, and which is the reason performance work is famous for
being undone quietly by people who had no idea it existed.

The fix is to turn the benchmark into a test. A test has a recorded expectation,
it runs on every change, and it fails when the expectation is violated. That is
all `scripts/check_regression.py` is. It reads a fresh
`results/inference_benchmark.json`, reads a committed baseline from
`results/baselines/`, compares them, and exits non zero when something got
worse. The benchmark stops being a report that somebody reruns when they
remember and becomes a thing that fails a build.

There is a second reason, which is that a gate makes an improvement legible. The
same comparison that catches a regression reports a speedup, with the size of
the change and the cell it landed in. That is the difference between a claim
that an optimization worked and evidence that it worked on this machine, on
these cells, by this much.

## What a Cell Is

The gate compares per cell, where a cell is one model on one backend at one
batch size. The identifier is built from the model name, the benchmark's own
`backend_key`, and the batch size, so a cell key reads
`DeepFM/openvino-cpu-fp32/bs1024`.

The `backend_key` already encodes the runtime, the device, and the precision as
`runtime-device-precision`. The device is part of the identity on purpose. The
same runtime on a CPU and on a CUDA device are different deployments answering
different questions, and a comparison that folded them together would be
comparing hardware while claiming to compare runtimes. This is the same rule the
two lanes in `docs/INFERENCE.md` are built on, applied to the gate.

Comparing headline numbers instead would hide the interesting failures. A change
that makes batch size 1 slower and batch size 4096 faster has a flat average and
a broken online serving path, and online serving is the case the latency budget
exists for. A change that only affects one runtime is invisible in a mean across
runtimes. The whole point of running a matrix is that the cells disagree, so the
comparison has to happen where the disagreement is.

Both latency and accuracy are compared in every cell, and the two are treated
differently at every stage. That is the subject of the next two sections.

## The Statistics, and What They Cannot Do

**A naive threshold on a single sample produces constant false alarms.** That is
the whole problem. A laptop running a benchmark is also running a browser, a
file indexer, and whatever else the operating system decided to do during the
timed loop. The measured latency of an unchanged model moves run to run by more
than the size of the regressions worth catching. A gate with a fixed ten percent
threshold on such a machine fires most of the time, everyone learns to ignore
it, and it is then worse than no gate at all, because it looks like it is still
working.

### What the artifact gives us to work with

The benchmark records, per cell, the mean, the median, the ninety ninth
percentile, and the minimum of the per batch latencies, plus the repeat count
and the number of timed batches whose product is the sample count. It does not
record the individual latencies. Everything below is built from that summary,
and the largest single improvement available to this layer would be for the
benchmark to also write the raw latency array, which is discussed at the end of
this section.

### Compare on the median, not the mean

Latency is compared on `p50_ms`. The mean of a latency sample is dragged around
by its own tail, and a latency tail is exactly what an unrelated background
process produces. On one of the runs recorded while this tooling was being
written, a cell had a mean of 44.5 ms and a median of 25.6 ms, which is a mean
that says almost nothing about a typical batch. The median moves when the whole
distribution moves, which is the event worth reporting, and it ignores a single
stalled batch, which is not.

The tail is compared separately, on `p99_ms`, because a shift that appears only
in the tail is a different fact about the system than a shift in the body, and
because the tail is what misses a serving deadline. Tail findings are reported
and do not fail the gate by default. Over a handful of timed batches the ninety
ninth percentile is arithmetically close to the single largest observation, and
failing a build on a single observation is not a gate, it is a coin toss. The
`--fail-on-tail` flag promotes tail findings to failures, and it becomes
reasonable once the repeat count is high enough for p99 to be a percentile
rather than a maximum.

### Three floors, all of which have to be cleared

A latency move is called a regression only when it clears all three of these.

| Floor | Default | What it stops |
| --- | --- | --- |
| Relative | 10 percent of the baseline median | A large absolute move on a slow cell that is proportionally trivial |
| Absolute | 0.10 ms | A large relative move on a fast cell, so a 0.01 ms move on a 0.05 ms measurement is not announced as a twenty percent regression |
| Noise | 2 standard errors of the difference of medians | A move that is smaller than the run to run variation of the cell it was measured in |

Each floor on its own misbehaves in a specific way. A relative floor alone
screams about sub microsecond moves on the fastest cells, which is most of the
batch size 1 row. An absolute floor alone goes silent on the slowest cells,
where a fifty percent regression is worth catching and would sail past a fixed
millisecond threshold in the wrong direction. A noise floor alone lets a small,
real, and perfectly repeatable regression through on a very quiet backend, where
the noise is genuinely near zero and a two percent regression is measurable and
still not worth a build failure. Requiring all three is what makes the gate
quiet on a laptop and still useful.

Improvements are the same three tests with the sign flipped. They are reported
and never fail the gate. Reporting them is what tells you the baseline has
drifted far enough from reality to be worth refreshing.

### The noise floor, in detail

The noise floor is the only one of the three that uses the recorded distribution
rather than a constant, and it is computed per cell, which is the point of
computing it at all. The same one millisecond move is a regression on a backend
whose measurements are tight and is inside the noise on a backend whose
measurements are not. A single global threshold cannot express that and has to
be set wrong for one of the two.

It is built in three steps.

**Step one, estimate the standard deviation from the recorded spread.** The raw
samples are not available, so the estimator is the range based one used in
statistical process control. For a sample of `n` observations from a normal
distribution, the expected range is `d2(n)` times the population standard
deviation, where `d2` is a tabulated constant. So the estimate is the observed
spread divided by `d2(n)`. The spread used is `p99 - min`, and `d2(3)` is 1.693,
`d2(10)` is 3.078.

**Step two, turn that into the uncertainty of the median.** The standard error
of a sample median is about 1.2533 times the standard error of the mean under a
normal approximation, so the standard error of the median is
`1.2533 * sigma / sqrt(n)`.

**Step three, combine the two runs.** The quantity being tested is a difference
of two medians measured independently, so the two standard errors add in
quadrature. The floor is that combined standard error multiplied by
`--noise-sigmas`, which defaults to 2.0. Two standard errors is roughly a ninety
five percent one sided bound under a normal approximation.

### What this estimator cannot do, stated plainly

**It has almost no power at small sample counts, and that is not a flaw in the
arithmetic, it is the truth about the measurement.** With three timed samples on
a cell whose latencies span a couple of milliseconds, the uncertainty in the
median is itself a couple of milliseconds, and no statistic can conjure
confidence that the data does not contain. The gate is therefore blind to
regressions well over ten percent on a three sample cell. The fix is more
repeats, not a tighter threshold. A tighter threshold on the same data does not
detect more regressions, it reports more noise. `--repeats 50` on a quiet machine
is what makes this gate sharp, and `scripts/sweep.sh --repeats N` exists for
that reason.

**The normal approximation is wrong, and it is wrong in the safe direction.**
Latency distributions are not normal. They have a hard floor at the fastest the
work can be done and a long right tail, so their range is inflated relative to
their standard deviation compared to a normal sample. Dividing that inflated
range by the normal `d2` therefore overestimates sigma, which widens the noise
floor and makes the gate quieter than a correct model would. Erring toward
silence is the right direction for an estimator this crude, because a gate that
cries wolf gets switched off and a gate that occasionally misses a small
regression is still catching the large ones.

**Two biases fight each other once the sample count is large.** `p99` is not the
maximum once there are more than a hundred samples, so `p99 - min` understates
the true range, which understates sigma and makes the gate louder. In the other
direction, the `d2` table is clamped at ten rather than extrapolated, and the
real `d2` keeps growing with `n`, so dividing by the clamped value overstates
sigma and makes the gate quieter. These partially cancel and neither is
principled. This is the clearest signal that the estimator is a workaround for a
missing artifact rather than a considered choice.

**There is no multiple comparison correction, on purpose.** A twelve cell sweep
runs twelve independent tests, so the chance of at least one false alarm is
higher than the per cell rate. A Bonferroni correction would divide the
significance across the cells and would make each cell so insensitive that the
gate would detect nothing at all at these sample counts. Investigating one false
alarm out of twelve cells is cheap. Losing all detection to keep the family wise
error rate down is not a trade worth making here, and the honest thing is to say
so rather than to pretend the per cell rate is the family rate.

### What the gate actually did on this machine

Three sweeps were run on the Apple M3 Pro while this tooling was being written,
and the results are worth writing down because they show both the design working
and its limits.

The first two, run minutes apart with identical settings and identical code,
disagreed on the median latency of some cells by a factor of four. The machine
was under concurrent load from unrelated work, the ONNX Runtime cell at batch
size 1024 moved from 5.1 ms to 27.8 ms, and the noise floor the gate computed
for that cell from its own recorded spread was 27.1 ms. So the gate reported
that move as within noise, which is the right call for a change that was caused
by the operating system rather than by the code.

Two other cells did clear all three floors on that run and the gate exited 2.
Nothing in the code had changed. Those two are false alarms, and they are the
honest cost of running a latency gate on a shared laptop rather than on a quiet
dedicated machine. A regression gate on this workload wants a quiet host and a
high repeat count, and the numbers above are what it looks like when it does not
get either.

A third observation came for free. Comparing a hundred thousand row run against
a baseline recorded from a twenty thousand row run exits 3, an accuracy
regression, because the AUC of 0.7731 and the AUC of 0.7661 are the same weights
scored over different test rows. The gate has no way to know that difference is
benign, which is exactly why a baseline is scoped to its config as well as to its
machine.

**The right fix is to record the raw latencies.** Every limitation above comes
from having four summary statistics instead of the sample they were computed
from. With the raw per batch latencies in the artifact, the comparison becomes a
Mann Whitney U test or a bootstrap confidence interval on the difference of
medians, neither of which assumes normality, both of which have real power at
small `n`, and both of which report a p value rather than a hand rolled floor.
That is the single highest value change available to this layer and it belongs
in `scripts/run_inference_benchmark.py`, which this document does not own.

## Why Accuracy Outranks Latency

**Accuracy is compared with an absolute tolerance and no noise allowance at
all.** The benchmark scores every held out test row with a fixed seed on every
run, so the AUC and the logloss of an unchanged model over unchanged data are
deterministic. The only jitter available is float32 accumulation order, which is
orders of magnitude below the default tolerance of one thousandth of an AUC
point. A moved AUC is therefore not measurement noise. It is a changed model, or
a changed graph, or a changed set of rows, and the gate should say so in every
one of those cases.

**A faster model that ranks worse is not an improvement, and the exit code says
which one happened.** A latency regression exits 2 and an accuracy regression
exits 3, and when both fire the process exits 3, because that is the finding
that changes what gets shipped. The asymmetry is not stylistic. A slower model
can be served by adding machines, which costs money that is easy to quantify and
easy to reverse. A model that ranks worse loses revenue on every impression it
misorders, that loss is proportional to traffic, and no amount of hardware fixes
it. In an ad system the ranking quality is the product and the latency is the
constraint the product has to fit inside.

This is also the specific failure mode that inference optimization produces. The
whole INT8 section of `docs/INFERENCE.md` is about a transformation that buys
latency and may cost accuracy, and the honesty rule there is that the accuracy
column is published whatever it says. The gate is that rule made executable. A
quantization change that halves the latency and drops the AUC by two points
fails the build with exit code 3, and the report prints the accuracy table above
the latency table.

**A baseline is scoped to its data settings as well as to its hardware.** The
AUC depends on which rows are in the test split, so a run at a different
`--sample-size` scores a different set of rows and produces a different AUC for
the same weights. Two runs recorded while this tooling was being written scored
0.7661 and 0.7731 for identical weights purely because one loaded twenty
thousand rows and the other loaded a hundred thousand. That is not a regression
and the gate would call it one. The `arguments` block is copied into every
baseline so the mismatch is visible, and `scripts/sweep.sh` takes its settings
from the config for exactly this reason. Compare like with like or the accuracy
column means nothing.

## Why a Baseline Is Hardware Scoped

**A latency number is a property of a machine, so a latency baseline is an
artifact of one machine and nothing else.** Comparing this run on an M3 Pro
against a baseline recorded on an A100 does not measure whether the code got
slower. It measures the difference between an Apple laptop and a data center
GPU, which is a fact nobody needed a regression gate to learn.

So the comparison is refused by default when the hardware does not match. Every
baseline carries a fingerprint of the machine that produced it, taken from the
hardware record the benchmark already collects.

| Field | Why it is in the fingerprint |
| --- | --- |
| `cpu_model` | The CPU lane runs here, and it is the whole story for OpenVINO and the ONNX Runtime CPU provider |
| `system` and `machine` | An arm64 macOS result and an x86_64 Linux result share no kernels and no compiler output |
| `logical_cores` | Thread count changes throughput on every CPU runtime, and a container with a different CPU limit is a different machine |
| `gpu_name` | The obvious one. A different GPU is a different device with different bandwidth and different tensor cores |
| `gpu_driver_version` | Kernel selection and clock behaviour change across driver versions, so a driver bump is a hardware change for this purpose |

A mismatch exits 5 and prints which fields differ. `--allow-hardware-mismatch`
forces the comparison and prints a note on every run saying that every latency
finding below compares two machines rather than two builds. That escape hatch
exists because there is one legitimate use for it, which is looking at the
relative shape of a curve across machines, and it is loud because there is no
legitimate use for quoting the numbers it produces as a regression.

**Library versions are a warning and not a refusal, and the difference matters.**
A torch bump or an ONNX Runtime bump changing the latency is precisely the event
this gate exists to catch. Refusing to compare across a version change would
hide the most valuable finding the tool can produce. So a moved library version
prints a warning, the comparison goes ahead, and the versions sit in the output
next to the numbers so the reader can attribute the change.

The practical consequence is that there is one baseline per machine per config,
and they live side by side under `results/baselines/`. `results/baselines/cpu_only.json`
is the Apple Silicon one. A GPU box records `results/baselines/gpu_full.json`
the first time it runs the sweep there, and neither is ever compared against the
other.

## The Regression Gate in Use

```bash
# check the default results against the default baseline
python scripts/check_regression.py

# check a specific run against a specific baseline, and keep the findings
python scripts/check_regression.py \
  --results results/sweeps/latest/inference_benchmark.json \
  --baseline results/baselines/cpu_only.json \
  --json-out /tmp/findings.json

# record a new baseline, deliberately
python scripts/check_regression.py --update-baseline
```

| Exit code | Meaning |
| --- | --- |
| 0 | No regression. Improvements and warnings may still have been reported |
| 1 | Usage or input error, such as a missing results file or a missing baseline |
| 2 | Latency regression |
| 3 | Accuracy regression, which outranks a latency regression |
| 4 | A cell that is in the baseline did not run |
| 5 | The baseline came from different hardware |

**Exit code 4 is the one no latency comparison could produce.** A backend that
stops running leaves no slower row behind to compare against. The row is simply
gone, and a gate that only diffs numbers reports a clean pass on a build that
lost its accelerated execution provider. That is not a hypothetical. It is what
happens when an ONNX Runtime install loses its CUDA provider, when a TensorRT
engine on disk no longer matches the runtime version, or when an `onnxruntime`
and an `onnxruntime-gpu` package end up installed over each other, which the
`install-gpu` target in the `Makefile` already carries a warning about. In every
one of those cases the process does not crash, it falls back, and the fallback
is slower. The gate treats a disappeared cell as a failure and reports the
reason the benchmark itself gave for skipping it, taken from the
`unavailable_backends` block, so the message is the runtime's own words rather
than a guess.

Cells that appeared since the baseline are reported and do not fail. A new
backend has no baseline to be compared against until somebody updates the
baseline.

**`--update-baseline` prints a banner made of asterisks and it says to read the
diff before committing.** A baseline updated by accident silently accepts
whatever regression was in the tree at that moment, and every future run is then
measured against the regression. That is worse than having no gate, because the
gate keeps passing and keeps looking like it is working. Nothing in
`scripts/sweep.sh` updates a baseline unless `--update-baseline` was typed on
its command line.

## The Sweep Driver

`scripts/sweep.sh` runs the whole procedure. The procedure is four steps that
have to happen in a specific order and that are easy to get wrong by hand.

1. Pick the config that matches the machine. The script runs `nvidia-smi -L` and
   requires it both to exist and to list a device, because a container can carry
   the binary with no GPU visible to it. A machine that passes both gets
   `gpu_full` and everything else gets `cpu_only`.
2. Build the TensorRT engines, on a GPU box only. This happens before the sweep
   so the build cost is measured on its own and never lands inside an inference
   timing, which is the rule `docs/INFERENCE.md` states and which is easy to
   violate by running the two in the wrong order.
3. Run the sweep, with the flags translated from the config.
4. Run the regression gate, and exit with its exit code, so the script can be
   the whole of a CI step.

Everything lands in a timestamped directory under `results/sweeps/`, together
with the config that produced it, the full logs, the gate's report in text and
in json, and a `run.txt` manifest recording the host, the GPU, the interpreter,
the git commit, whether the tree was dirty, how long the sweep took, and what
the gate said. `results/sweeps/latest` is a symlink to the newest run.

Archived runs are not committed. Each one holds an exported ONNX graph of about
eighty megabytes, and the numbers in it are only meaningful on the machine that
produced them. `results/sweeps/.gitignore` keeps them out. The committed record
of a run is the baseline under `results/baselines/`, which is small, readable,
and carries its own hardware record.

```bash
bash scripts/sweep.sh --help
bash scripts/sweep.sh                          # auto detect, run, gate, archive
bash scripts/sweep.sh --dry-run                # print the resolved plan and stop
bash scripts/sweep.sh --repeats 50             # the knob that makes the gate sharp
bash scripts/sweep.sh --config gpu_full        # force a config
bash scripts/sweep.sh --update-baseline        # record instead of gating
```

The shebang is bash rather than sh. `set -o pipefail` is not in POSIX, and a
failure swallowed inside a pipeline is exactly the class of bug a script like
this must not have, so the portability of `sh` was traded for it deliberately.
The body otherwise stays close to POSIX shell. There is no `[[ ]]`, no process
substitution, and one array reference, which is the `PIPESTATUS` needed to read
the gate's exit code back out of the pipeline that tees its report into the
archive.

### Adding a configuration

A configuration is a YAML file under `benchmarks/` following the schema in the
last section of `docs/INFERENCE.md`. To add one.

1. Write `benchmarks/<name>.yaml`. Copy `cpu_only.yaml` and change the matrix,
   the data settings, and the `hardware` label. The `hardware` field is free
   text and it is stamped onto every row the config produces, so it has to
   describe the machine the config is meant to run on.
2. Check that it parses and says what you meant. `python tools/sweep_config.py <name> --print summary`
   prints the resolved matrix and the cell count after exclusions, and
   `--print flags` prints the exact command line it translates to. The CI
   `configs` job parses every file in `benchmarks/` on every commit, so a typo
   fails there rather than on a machine somebody is paying for by the second.
3. Run it. `bash scripts/sweep.sh --config <name>`.
4. Record its baseline once you have read the numbers.
   `bash scripts/sweep.sh --config <name> --update-baseline`.

`tools/sweep_config.py` is the translator between a config and the benchmark's
command line, and it exists because translating by hand is how a config and the
run it claims to describe drift apart. It prints, on stderr, every part of the
config it could not translate, and there are three of those today.

`matrix.runtimes` has no flag. The benchmark probes every backend it knows about
and records the ones it could not build under `unavailable_backends`, so the
runtime list in a config is a statement of what is expected rather than a
selection. `exclude` has no flag either, because exclusion rules drop individual
cells from a cartesian product and a flag list cannot express that, so the sweep
runs the full product. The `calibration` block is read where the engines are
built, in `scripts/build_trt_engines.py`, and not where the sweep is run.

All three warnings print on every run rather than being suppressed, because the
alternative is a reader who believes a config key had an effect it did not have.

One further discrepancy is worth naming. The `bench-cpu-sweep` and `bench-gpu`
targets in the `Makefile` pass comma separated lists to `--models`,
`--precisions`, and `--batch-sizes`, and the benchmark script defines those
flags with `nargs="+"`, which takes space separated values.
`tools/sweep_config.py` emits the space separated form, which is the one that
works.

## Nsight Systems and Nsight Compute

The benchmark answers how long a batch takes. A profiler answers why, and the
two Nsight tools answer different halves of that.

**Nsight Systems is the timeline.** It traces a whole process at low overhead
and draws CUDA kernels, memory copies, library calls, and NVTX ranges against
wall clock on every stream. It shows the gaps as well as the work, which on a
model this small is usually the finding, because the most likely answer to why a
DeepFM batch takes two milliseconds on an A100 is that the GPU spent most of
those two milliseconds idle waiting for the host. Nothing about a per kernel
counter report can tell you that. Systems can, at a glance, because the idle
time is drawn as empty space.

**Nsight Compute is the microscope.** It replays a single kernel many times,
collecting a different set of hardware counters on each pass, and reports what
that kernel was limited by. It is the tool for why one kernel is slow. It is
also enormously slower than the workload it is profiling, because of those
replays, so it is never pointed at a whole benchmark. It is pointed at a named
kernel.

**The order is Systems first and Compute second, always.** Nsight Compute on a
kernel that turns out to be two percent of the wall clock is a beautifully
measured answer to the wrong question. Systems names the kernel worth asking
about and Compute answers the question about it. Running Compute first is the
most common way a profiling session wastes an afternoon.

`scripts/profile_nsight.sh` runs both, with flags chosen for this workload rather
than left at their defaults, and turns the capture into markdown through
`nsys stats` and `tools/summarize_profile.py`.

### The Nsight Systems flags, and why

`--trace=cuda,nvtx,cublas,cudnn`. CUDA gives the kernels and the copies, which
is the substance. NVTX gives the named ranges, which is what makes the timeline
readable rather than a wall of mangled names. cuBLAS and cuDNN attribute the
library calls the multilayer perceptron makes, so a gemm is labelled as a gemm
rather than as whatever kernel the library happened to pick. `osrt` is
deliberately absent. It traces every operating system runtime call, and on a
Python process that means thousands of futex, poll, and read entries per second,
which multiplies the capture size and buries the CUDA rows under noise that has
nothing to do with the model. Default flags capture a lot of noise, and this is
most of it.

`--sample=none`, `--cpu-core-events=none`, and `--backtrace=none`. CPU sampling
is a periodic interrupt that collects host call stacks, and backtrace collection
on every API call is the single most expensive default in nsys. Both are worth
turning on when the question is which line of Python launched a kernel. Neither
is worth its overhead when the question is how long the kernels took.

`--cuda-memory-usage=false`. Per kernel allocation tracking is useful for a
memory bug and costs real overhead on a model that allocates on every forward.
The embedding table is allocated once at load and is not what this profile is
about.

`--gpu-metrics-devices=all`. This is the flag that matters most for this model.
It samples the GPU hardware counters continuously and draws SM activity and DRAM
throughput as timeline rows underneath the kernels. That is the direct evidence
for or against the memory bandwidth bound prediction in `docs/INFERENCE.md`, and
having the two rows side by side makes the case in a way a table of kernel
durations cannot.

`--stats=false`. nsys can print its own summary when a capture finishes. It is
off here because the summary is generated separately through `nsys stats` with a
chosen report list and CSV output, which is what `tools/summarize_profile.py`
consumes. Two summaries with different report sets is one summary too many.

### The Nsight Compute flags, and why

`--launch-skip 200 --launch-count 20`. Compute replays each profiled kernel
several times per section, so profiling every launch of a benchmark that runs
thousands would take hours and would say nothing the first twenty did not.
Skipping the first two hundred launches steps past warmup, where kernel
selection and clock ramp make the numbers unrepresentative, which is the same
argument the warmup section of `docs/INFERENCE.md` makes about timing.

The section list is chosen rather than left at `--set full`, which collects
everything and multiplies the replay count. Each section here answers a specific
question about this model.

| Section | What it answers |
| --- | --- |
| `SpeedOfLight` | Compute throughput and memory throughput as percentages of peak, side by side. This one section decides the bandwidth bound question |
| `MemoryWorkloadAnalysis` | DRAM traffic, L1 and L2 hit rates, and sectors per request. For an embedding gather this is the locality story |
| `LaunchStats` | Grid and block dimensions and registers per thread. On a small model the kernels are often too small to fill the device, and this is where that shows |
| `Occupancy` | Achieved against theoretical occupancy and the limiter. Low achieved occupancy on a memory bound kernel usually means too little work in flight to hide load latency |

`--kernel-name-base=demangled` because template heavy CUDA kernel names are
unreadable mangled and merely long demangled, and because a filter expression
has to be written against something a person can read. `--target-processes all`
because a runtime that forks a worker would otherwise be missed entirely.

Nsight Compute also needs permission to read the GPU performance counters, and
on the hardware this project actually targets that permission is not available.

**This was tested rather than assumed.** On a RunPod pod on 2026-08-28, running
`ncu --metrics dram__bytes.sum` against a trivial kernel produced
`ERR_NVGPUCTRPERM`, the error that means the user cannot access the GPU
performance counters on the target device. The `ncu` binary is present at
`/usr/local/cuda/bin/ncu`, it launches, and it attaches to the process. It is
refused at the counter read. RunPod containers are unprivileged and are not
granted `CAP_SYS_ADMIN`, and **this cannot be fixed from inside the pod.**

The usual remedies do not apply there. Running the container with
`--cap-add=SYS_ADMIN` requires control of the Docker invocation, which a managed
pod does not give you, and setting the driver module parameter
`NVreg_RestrictProfilingToAdminUsers` to 0 requires host root and a reboot.
Both are available on a machine you own and neither is available on a rented
container. If you are profiling on your own hardware, use them. If you are
profiling on RunPod, plan around their absence from the start.

So nothing in this repository depends on Nsight Compute counters. The question
those counters would have answered, which is how many bytes actually cross the
memory bus and therefore whether this model is bandwidth bound, is answered
analytically instead. `src/inference/analysis.py` derives the byte count from
the tensor shapes and the dtype widths, which is reproducible, needs no elevated
permission, and can be checked by hand. That is the better artifact regardless
of permissions, because a reader can verify the arithmetic without owning a GPU.

Nsight Systems is a different matter and is expected to work, because a timeline
trace does not read the restricted counters. The timeline is the piece that will
actually be produced on a rented box.

The script prints all of this before it runs and again if a run fails, because
discovering it after provisioning a GPU is an expensive way to learn it.

### NVTX ranges

A capture with no NVTX ranges in it is a list of kernel names against a clock.
It is readable, and working out which stripe is the embedding gather and which
is the multilayer perceptron means reading mangled template names one at a time.
An NVTX range is a named interval pushed on the host, and Nsight Systems draws
it as a labelled bar on its own row above the CUDA rows, so the timeline reads
as embedding gather, then concatenate, then linear one, rather than as a hundred
anonymous kernels.

`tools/nvtx.py` adds them from the outside. It installs forward pre hooks and
forward hooks on a torch module, so every submodule forward becomes a named
range, the model itself is unchanged, and the instrumentation is removable.
Nothing in `scripts/run_inference_benchmark.py` imports it and nothing has to.

```python
from tools.nvtx import instrument, range as nvtx_range

handle = instrument(model)        # one named range per submodule forward
with nvtx_range("timed pass"):    # a range around anything at all
    run_the_batches()
handle.remove()
```

Two rules go with it. Remove the instrumentation before publishing a latency
number, because the hooks are cheap and not free and a timing run and a
profiling run should not be the same run. And push a range around the timed
section so the capture can be limited to it with
`scripts/profile_nsight.sh --nvtx-capture <name>`, since otherwise the capture
includes model loading, the ONNX export, and every warmup batch, which together
are usually longer than the part worth looking at.

Everything in `tools/nvtx.py` is inert when NVTX is not available, which is
every machine without a CUDA build of PyTorch, so an annotated script still runs
unchanged on a laptop.

### Turning a capture into something readable

An `.nsys-rep` file opens in the Nsight Systems GUI and nowhere else. It cannot
be diffed, cannot be read in a pull request, and does not survive being sent to
somebody who has not installed the GUI. `nsys stats` runs canned reports over a
capture and writes them as CSV, and `tools/summarize_profile.py` turns those CSV
files into the same markdown table style the rest of this project reports in.

It also buckets the kernel time by what each kernel is, which is the table that
decides the question `docs/INFERENCE.md` asked. The buckets are substring
matches against kernel names and they are heuristics rather than a ground truth,
so a misclassified kernel is a mislabelled row and not a wrong measurement, and
the per kernel table underneath always carries the raw names so a reader can
check.

## Reading a Timeline for This Workload

`docs/INFERENCE.md` predicts that this model is memory bandwidth bound on the
embedding gather rather than compute bound on the multilayer perceptron. A
profile is how that prediction gets confirmed or killed. There are three
distinct shapes to look for and they call for three completely different
responses.

**Bandwidth bound looks like this.** One wide kernel dominates the GPU row, and
its demangled name contains `gather`, `indexSelect`, `embedding`, or
`EmbeddingBag`. The gemm kernels are visible and narrow next to it. In the GPU
metrics rows, DRAM throughput sits high while SM activity sits low, which is the
signature, because the arithmetic units are idle waiting for memory. In Nsight
Compute, `SpeedOfLight` on that kernel shows memory throughput far above compute
throughput, and `MemoryWorkloadAnalysis` shows a low L2 hit rate and a high
sectors per request ratio, because each of the thirty two lanes in a warp is
pulling a different cache line from an unrelated part of the table. Latency
scales close to linearly with batch size once the fixed overhead has amortized.
The response is to reduce bytes moved. Smaller embedding dimensions, a smaller
hash space, FP16 storage for the tables, or a table layout with better locality.
Faster arithmetic buys nothing here, which is the same reason the prediction says
FP16 will underdeliver.

**Compute bound looks like the opposite.** The gemm kernels dominate, SM
activity is high, DRAM throughput is moderate, `SpeedOfLight` shows compute
throughput as the larger of the two numbers, and occupancy is limited by
registers or shared memory rather than by memory latency. Latency stays close to
flat as batch size grows until the device saturates and then rises. The response
is precision. FP16 and INT8 do what the marketing says on a workload of this
shape, and this is the case where the tensor core speedup is real.

**Launch bound is the third shape and at batch size 1 it is the most likely
one.** The timeline has more gap than kernel. Individual kernels are a few
microseconds and the spaces between them are longer than the kernels are. The
CUDA API row is full of `cudaLaunchKernel` and the GPU rows are mostly empty.
This is neither compute nor bandwidth bound, it is host bound, and neither
precision nor a bigger GPU fixes it. The response is to stop launching so many
kernels. CUDA graph capture, operator fusion, or a runtime that compiles the
whole graph ahead of time, which is exactly what the TensorRT and OpenVINO rows
in the benchmark are testing. A profile that shows this shape is also the
explanation for why a graph compiling runtime wins at batch size 1 and stops
winning at batch size 4096.

There is a fourth thing to look for that is specific to one row of the sweep.
When the ONNX Runtime TensorRT execution provider is in the capture, the
timeline shows TensorRT engine kernels alternating with ONNX Runtime kernels.
Many short alternations mean the graph partitioned badly, that the engine is a
dozen small subgraphs rather than one, and that the cost of crossing back and
forth is eating the speedup. `docs/INFERENCE.md` names the partition count as
the thing to log next to that latency, and the timeline is where it becomes
visible.

One more note on cache. The hash space in this project is capped at ten thousand
buckets per field, which makes the embedding tables small enough that a large
fraction may be resident in L2 on a data center GPU. If the profile shows a high
L2 hit rate on the gather, the bandwidth bound prediction may be wrong on this
configuration while still being right on a production sized table with millions
of rows per field. That possibility is named in `docs/INFERENCE.md` as the most
likely way the prediction fails, and the L2 hit rate in
`MemoryWorkloadAnalysis` is the number that decides it.

## What Has Been Executed and What Has Not

Every claim in this document is either a measurement, a citation of a documented
interface, or a design argument, and this table says which is which.

| Piece | Status |
| --- | --- |
| `scripts/check_regression.py` | Executed on an Apple M3 Pro. Baselines recorded, comparisons run, every exit code exercised |
| `tests/test_regression_gate.py` | Executed. Passes on an Apple M3 Pro under `pytest -q` |
| `scripts/sweep.sh` | Executed on an Apple M3 Pro. Full end to end runs of `cpu_only`, including engine skip, sweep, gate, and archive. The GPU branch has never run |
| `tools/sweep_config.py` | Executed against both shipped configs |
| `tools/nvtx.py` | Only its no op path is executed. The machine has no NVTX, so every range compiles to nothing and that is what was verified. The NVTX path is unrun |
| `tools/summarize_profile.py` | Executed against CSV written by hand to the column layout `nsys stats` documents. It has never seen output from a real `nsys` binary |
| `scripts/profile_nsight.sh` | Only its unavailable path is executed. It detects that `nsys` and `ncu` are absent, prints what to run instead, and exits 0. The profiling path is unrun |

**The Nsight path is unexecuted.** The machine this project is developed on is
an Apple M3 Pro. It has no NVIDIA GPU, no CUDA, and no Nsight, so no capture has
ever been taken, no `nsys stats` output has ever been parsed, and no NVTX range
has ever appeared on a timeline. The flags in `scripts/profile_nsight.sh` were
chosen from the documented behaviour of the tools and from what this workload
needs, and they have not been validated by running them. The section above on
what a bandwidth bound timeline looks like describes what the prediction in
`docs/INFERENCE.md` implies should be seen. It is not a report of something
seen.

The way to change that is to run the container. `docker/README.md` has the
setup, and inside it the sequence is `bash scripts/sweep.sh` followed by
`bash scripts/profile_nsight.sh --mode both`. Until somebody does that, this
half of the document is a design and not a result, and it says so here rather
than being written in a voice that lets a reader assume otherwise.

**Every number this layer produces is labeled with the machine that produced
it.** The baselines carry a hardware record, the gate refuses to compare across
machines, the sweep manifest records the host and the GPU, and the profile
summary refuses to print an unknown hardware label without saying that it is
unknown. That is the same honesty rule `docs/INFERENCE.md` states, enforced by
the tooling rather than by remembering.
