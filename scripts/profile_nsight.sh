#!/usr/bin/env bash
# Profile the ClickRank inference benchmark under NVIDIA Nsight.
#
# The benchmark answers how long a batch takes. A profiler answers why it takes
# that long, and those are different questions with different tools. This script
# runs both Nsight tools over the same workload with flags chosen for this model
# rather than left at their defaults, and it turns the capture into a markdown
# summary so the result is readable without opening a GUI.
#
# Nsight Systems is the timeline. It traces the whole process at low overhead
# and shows CUDA kernels, memory copies, library calls, and NVTX ranges laid out
# against wall clock on every stream. It is the tool for where the time goes,
# including the time that is not spent in kernels at all, which on a model this
# small is usually most of it.
#
# Nsight Compute is the microscope. It replays one kernel many times, collecting
# hardware counters on each pass, and reports what that kernel was limited by.
# It is the tool for why one kernel is slow. It is far too slow to run over a
# whole benchmark and it is pointed at a named kernel that Nsight Systems has
# already identified as worth asking about.
#
# The order is Systems first and Compute second, always. Compute on a kernel
# that turns out to be two percent of the wall clock is a well measured answer
# to the wrong question.
#
#   bash scripts/profile_nsight.sh --help
#   bash scripts/profile_nsight.sh
#   bash scripts/profile_nsight.sh --mode compute --kernel "regex:.*[Gg]ather.*"
#
# THIS SCRIPT HAS NEVER BEEN EXECUTED AGAINST A REAL NSIGHT INSTALL.
#
# The machine this project is developed on is Apple Silicon. It has no NVIDIA
# GPU, no CUDA, and no Nsight, so the profiling path here is unrun. What has
# been exercised on that machine is the path this script takes when nsys is
# absent, which is to explain the situation and exit 0, and the markdown
# summary step, which was exercised against CSV written to the column layout
# nsys stats documents rather than against output from a real nsys binary. That
# distinction is stated again in docs/BENCHMARK_AUTOMATION.md and it is not
# softened anywhere.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python3}"

MODE="systems"
OUTPUT_DIR="${REPO_ROOT}/results/profiles"
BATCH_SIZE="1024"
SAMPLE_SIZE="100000"
REPEATS="10"
WARMUP="20"
KERNEL_FILTER="regex:.*"
LAUNCH_COUNT="20"
LAUNCH_SKIP="200"
NVTX_CAPTURE=""
GPU_METRICS=1
DRY_RUN=0
SKIP_SUMMARY=0

log() {
  printf '%s\n' "$*"
}

step() {
  printf '\n== %s ==\n' "$*"
}

die() {
  printf 'profile_nsight.sh error. %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'USAGE'
profile_nsight.sh. Profile the inference benchmark under NVIDIA Nsight.

  usage
    bash scripts/profile_nsight.sh [options]

  modes
    systems   Nsight Systems only. The timeline. This is the default and it is
              where every investigation starts.
    compute   Nsight Compute only. Per kernel hardware counters. Point it at a
              kernel that Systems has already shown to matter.
    both      Systems, then Compute, over two separate runs of the workload.
              They are never collected in one process, because Compute replays
              kernels and the replays would corrupt the Systems timeline.

  options
    -m, --mode MODE         systems, compute, or both. Default systems
    -o, --output DIR        Where captures and reports are written.
                            Default results/profiles
        --batch-size N      Batch size to profile. Default 1024
        --sample-size N     Rows to load before the split. Default 100000
        --repeats N         Timed passes. Default 10, which is fewer than a
                            benchmark run uses, because a profile needs enough
                            iterations to be representative and not enough to
                            make the capture enormous.
        --warmup N          Warmup batches. Default 20. Warmup matters more
                            under a profiler than without one, since the first
                            iterations carry lazy kernel selection and clock
                            ramp and would otherwise be the loudest thing on
                            the timeline.
        --kernel FILTER     Nsight Compute kernel filter. Default regex:.*
                            Example, regex:.*[Gg]ather.*
        --launch-count N    Kernel launches Nsight Compute profiles. Default 20
        --launch-skip N     Launches to skip first, so the profiled launches
                            are steady state rather than warmup. Default 200
        --nvtx-capture NAME Limit the Systems capture to an NVTX range of this
                            name. Needs the process to push that range. See
                            tools/nvtx.py.
        --no-gpu-metrics    Turn off the GPU metrics sampler.
        --skip-summary      Do not run nsys stats or write the markdown.
        --dry-run           Print the exact commands and exit.
    -h, --help              This message.

  environment
    PYTHON                  Interpreter to use. Default python3

  This script has never been run against a real Nsight install. The development
  machine for this project has no NVIDIA GPU. On a machine without nsys it
  prints what to do and exits 0.
USAGE
}

require_value() {
  if [ "$2" -lt 2 ]; then
    die "$1 needs a value. Run bash scripts/profile_nsight.sh --help"
  fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    -m|--mode)
      require_value "$1" $#
      MODE="$2"
      shift 2
      ;;
    -o|--output)
      require_value "$1" $#
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --batch-size)
      require_value "$1" $#
      BATCH_SIZE="$2"
      shift 2
      ;;
    --sample-size)
      require_value "$1" $#
      SAMPLE_SIZE="$2"
      shift 2
      ;;
    --repeats)
      require_value "$1" $#
      REPEATS="$2"
      shift 2
      ;;
    --warmup)
      require_value "$1" $#
      WARMUP="$2"
      shift 2
      ;;
    --kernel)
      require_value "$1" $#
      KERNEL_FILTER="$2"
      shift 2
      ;;
    --launch-count)
      require_value "$1" $#
      LAUNCH_COUNT="$2"
      shift 2
      ;;
    --launch-skip)
      require_value "$1" $#
      LAUNCH_SKIP="$2"
      shift 2
      ;;
    --nvtx-capture)
      require_value "$1" $#
      NVTX_CAPTURE="$2"
      shift 2
      ;;
    --no-gpu-metrics)
      GPU_METRICS=0
      shift
      ;;
    --skip-summary)
      SKIP_SUMMARY=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      die "unknown option $1. Run bash scripts/profile_nsight.sh --help"
      ;;
    *)
      die "unexpected argument $1. Every input is a flag. Run bash scripts/profile_nsight.sh --help"
      ;;
  esac
done

case "${MODE}" in
  systems|compute|both) ;;
  *) die "unknown mode ${MODE}. It has to be systems, compute, or both." ;;
esac

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------
#
# The same shape as scripts/build_trt_engines.py takes when TensorRT is absent.
# Nothing is faked, nothing is estimated, and the exit is 0 so that a machine
# without the tool does not turn a whole pipeline red for a step that could
# never have run there.

# Put the CUDA toolkit on PATH before probing for anything. On the RunPod
# PyTorch image, and on plenty of other CUDA installs, nsys and ncu and nvcc all
# exist under /usr/local/cuda/bin while nothing points at that directory. A bare
# command -v probe therefore reports the tools missing on a machine where they
# are present, which sends people chasing an installation problem that does not
# exist. Appending is safe when the directory is absent or already on the path.
if [ -d /usr/local/cuda/bin ]; then
  case ":${PATH}:" in
    *":/usr/local/cuda/bin:"*) ;;
    *) PATH="/usr/local/cuda/bin:${PATH}"; export PATH ;;
  esac
fi

HAVE_NSYS=0
HAVE_NCU=0
command -v nsys >/dev/null 2>&1 && HAVE_NSYS=1
command -v ncu >/dev/null 2>&1 && HAVE_NCU=1

NEED_NSYS=0
NEED_NCU=0
case "${MODE}" in
  systems) NEED_NSYS=1 ;;
  compute) NEED_NCU=1 ;;
  both) NEED_NSYS=1; NEED_NCU=1 ;;
esac

MISSING=""
if [ "${NEED_NSYS}" -eq 1 ] && [ "${HAVE_NSYS}" -eq 0 ]; then
  MISSING="nsys"
fi
if [ "${NEED_NCU}" -eq 1 ] && [ "${HAVE_NCU}" -eq 0 ]; then
  if [ -n "${MISSING}" ]; then
    MISSING="${MISSING} and ncu"
  else
    MISSING="ncu"
  fi
fi

if [ -n "${MISSING}" ]; then
  step "nothing to profile with on this machine"
  log "  not on PATH. ${MISSING}"
  log ""
  log "  Nsight Systems and Nsight Compute ship with the CUDA toolkit and need"
  log "  an NVIDIA GPU to capture anything. This machine is $(uname -srm)."
  log ""
  log "  Run this inside the pinned container instead. It carries the toolkit,"
  log "  both Nsight command line tools, and the TensorRT version the engines"
  log "  were built against."
  log ""
  log "    docker compose -f docker/docker-compose.yml run --rm shell"
  log "    bash scripts/profile_nsight.sh --mode both"
  log ""
  log "  or, in one command,"
  log ""
  log "    docker compose -f docker/docker-compose.yml run --rm shell \\"
  log "      bash scripts/profile_nsight.sh --mode both"
  log ""
  log "  The container is docker/Dockerfile.tensorrt and docker/README.md has"
  log "  the setup, including the NVIDIA container toolkit that has to be"
  log "  installed on the host before the GPU is visible inside it."
  log ""
  log "  Nsight Compute additionally needs permission to read the GPU"
  log "  performance counters, and on RunPod that permission is NOT available."
  log "  Verified by running it on a real pod on 2026-08-28. ncu is present at"
  log "  /usr/local/cuda/bin/ncu, it launches and attaches, then fails at the"
  log "  counter read with ERR_NVGPUCTRPERM. RunPod containers are unprivileged"
  log "  and are not granted CAP_SYS_ADMIN. This is not fixable from inside the"
  log "  pod. --cap-add=SYS_ADMIN needs control of the docker invocation and"
  log "  NVreg_RestrictProfilingToAdminUsers needs host root and a reboot, so"
  log "  both work on hardware you own and neither works on a rented pod."
  log ""
  log "  Nothing in this repository depends on those counters. Memory traffic"
  log "  is derived analytically from tensor shapes in src/inference/analysis.py"
  log "  instead. Nsight Systems does not need the counter permission and is"
  log "  the mode to use on a rented box."
  log ""
  log "  Note also that CUDA is not on PATH on the RunPod image. nsys, ncu and"
  log "  nvcc all live in /usr/local/cuda/bin but nothing points at them, so a"
  log "  which based probe will wrongly report them missing. Run"
  log "  export PATH=/usr/local/cuda/bin:\$PATH first."
  log ""
  log "  Nothing was profiled and nothing was estimated. Exiting 0, because a"
  log "  machine with no GPU is not a failure of this script."
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${OUTPUT_DIR}/${STAMP}"
BASE="deepfm_bs${BATCH_SIZE}"

# The workload. Deliberately the real benchmark script and not a separate
# harness, because a profile of a reimplementation of the thing under test is a
# profile of the reimplementation. --no-power is passed because the NVML power
# sampler runs its own polling thread, and a background thread sampling the
# driver during a capture shows up on the timeline as work that the serving path
# does not actually do.
WORKLOAD="${PYTHON} scripts/run_inference_benchmark.py \
  --synthetic \
  --sample-size ${SAMPLE_SIZE} \
  --batch-size ${BATCH_SIZE} \
  --batch-sizes ${BATCH_SIZE} \
  --models deepfm \
  --precisions fp32 \
  --repeats ${REPEATS} \
  --warmup ${WARMUP} \
  --no-power \
  --output ${RUN_DIR}/workload"

# ---------------------------------------------------------------------------
# Nsight Systems flags, and why each one is here
# ---------------------------------------------------------------------------
#
# --trace=cuda,nvtx,cublas,cudnn
#   cuda gives the kernels and the memory copies, which is the substance.
#   nvtx gives the named ranges, which is what makes the timeline readable
#   instead of a wall of mangled kernel names. cublas and cudnn attribute the
#   library calls the multilayer perceptron makes, so a gemm is labelled as a
#   gemm rather than as whatever kernel the library picked today. osrt is
#   deliberately not in the list. It traces every operating system runtime call
#   and on a Python process that means thousands of futex, poll, and read
#   entries per second, which triples the capture size and buries the CUDA rows
#   under noise that has nothing to do with the model.
#
# --sample=none and --cpu-core-events=none
#   CPU sampling is a periodic interrupt that collects host call stacks. The
#   question here is where GPU time goes, the host side story is already visible
#   through the CUDA API rows, and the sampler is pure overhead and pure noise
#   for this workload.
#
# --backtrace=none
#   Backtrace collection on every API call is the single most expensive default
#   in nsys. It is worth turning on when the question is which line of Python
#   launched a kernel. It is not worth it when the question is how long the
#   kernels took.
#
# --cuda-memory-usage=false
#   Tracking allocation size per kernel is useful for a memory bug and costs
#   real overhead on a model that allocates on every forward. The embedding
#   table is allocated once at load and is not what this profile is about.
#
# --gpu-metrics-devices
#   This is the flag that matters most for this specific model. It samples the
#   GPU hardware counters continuously and draws SM occupancy and DRAM
#   throughput as timeline rows underneath the kernels. That is the direct
#   evidence for or against the memory bandwidth bound prediction in
#   docs/INFERENCE.md. High DRAM throughput next to low SM activity is the
#   signature of a bandwidth bound workload, and the two rows sitting side by
#   side make the case in a way a table of kernel durations cannot.
#
# --force-overwrite=true
#   So a rerun into the same directory does not fail on an existing file.
#
# --stats=false
#   nsys can print summary statistics when the capture finishes. It is turned
#   off here because the summary is generated below through nsys stats instead,
#   with a chosen list of reports and CSV output that tools/summarize_profile.py
#   consumes. Two summaries with different report sets is one summary too many.

NSYS_FLAGS="profile \
  --trace=cuda,nvtx,cublas,cudnn \
  --sample=none \
  --cpu-core-events=none \
  --backtrace=none \
  --cuda-memory-usage=false \
  --force-overwrite=true \
  --stats=false \
  --output=${RUN_DIR}/${BASE}"

if [ "${GPU_METRICS}" -eq 1 ]; then
  NSYS_FLAGS="${NSYS_FLAGS} --gpu-metrics-devices=all"
fi

# Capturing only a named NVTX range is the right way to keep model loading, the
# ONNX export, and the warmup batches out of the capture. It needs the process
# to actually push that range, which the benchmark does not do on its own. See
# tools/nvtx.py for how to add it from the outside without editing the benchmark.
if [ -n "${NVTX_CAPTURE}" ]; then
  NSYS_FLAGS="${NSYS_FLAGS} --capture-range=nvtx --nvtx-capture=${NVTX_CAPTURE} --capture-range-end=stop"
fi

# ---------------------------------------------------------------------------
# Nsight Compute flags, and why each one is here
# ---------------------------------------------------------------------------
#
# --launch-skip and --launch-count
#   Nsight Compute replays each profiled kernel several times to collect every
#   counter, so profiling every launch of a benchmark that runs thousands of
#   them would take hours and would tell you nothing the first twenty launches
#   did not. Skipping the first launches steps past warmup, where kernel
#   selection and clock ramp make the numbers unrepresentative.
#
# --section
#   The section list is chosen rather than left at --set full, which collects
#   every section and multiplies the replay count. Each one here answers a
#   specific question about this model.
#
#     SpeedOfLight            the compute throughput and the memory throughput
#                             as percentages of peak, side by side. This single
#                             section decides the bandwidth bound question. A
#                             kernel at eighty percent of memory throughput and
#                             five percent of compute throughput is bandwidth
#                             bound and no further evidence is needed.
#     MemoryWorkloadAnalysis  DRAM traffic, L1 and L2 hit rates, and sector
#                             counts per request. For an embedding gather this
#                             is the locality story. A scattered gather shows a
#                             low L2 hit rate and a high sectors per request
#                             ratio, because each of the thirty two lanes in a
#                             warp pulls a different cache line.
#     LaunchStats             grid and block dimensions and registers per
#                             thread. On a small model the kernels are often too
#                             small to fill the device and this is where that
#                             shows up.
#     Occupancy               achieved against theoretical occupancy, and the
#                             limiter. Low achieved occupancy on a memory bound
#                             kernel usually means not enough work in flight to
#                             hide the latency of the loads.
#
# --kernel-name-base=demangled
#   Template heavy CUDA kernel names are unreadable mangled and merely long
#   demangled. Demangled is the lesser evil and it is what makes a filter
#   expression possible to write.
#
# --target-processes all
#   The benchmark launches work from the Python process itself, but a runtime
#   that forks a worker would otherwise be missed entirely.
#
# --csv and --log-file
#   A text log next to the .ncu-rep, so the numbers can be read and diffed
#   without the Nsight Compute GUI.

NCU_FLAGS="--target-processes all \
  --kernel-name-base=demangled \
  --kernel-name ${KERNEL_FILTER} \
  --launch-skip ${LAUNCH_SKIP} \
  --launch-count ${LAUNCH_COUNT} \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section LaunchStats \
  --section Occupancy \
  --force-overwrite \
  --export ${RUN_DIR}/${BASE}_ncu"

# The reports nsys stats runs over the capture. Each one becomes a CSV that
# tools/summarize_profile.py turns into a markdown table.
NSYS_REPORTS="--report cuda_gpu_kern_sum \
  --report cuda_gpu_mem_time_sum \
  --report cuda_gpu_mem_size_sum \
  --report cuda_api_sum \
  --report nvtx_pushpop_sum"

# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

step "plan"
log "  repository    ${REPO_ROOT}"
log "  mode          ${MODE}"
log "  run directory ${RUN_DIR}"
log "  nsys          $(command -v nsys 2>/dev/null || echo "not present")"
log "  ncu           $(command -v ncu 2>/dev/null || echo "not present")"
log "  gpu           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || echo "unknown")"
log "  driver        $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || echo "unknown")"

step "NVTX ranges"
log "  A capture with no NVTX ranges in it is a list of kernel names against a"
log "  clock. It is readable, and working out which stripe is the embedding"
log "  gather and which is the multilayer perceptron means reading mangled"
log "  names one at a time."
log ""
log "  tools/nvtx.py adds the ranges from the outside, with no change to the"
log "  benchmark. One call puts a named range around every submodule forward."
log ""
log "    from tools.nvtx import instrument, range as nvtx_range"
log "    handle = instrument(model)"
log "    with nvtx_range(\"timed pass\"):"
log "        run_the_batches()"
log "    handle.remove()"
log ""
log "  Remove the instrumentation before publishing a latency number. The hooks"
log "  are cheap and they are not free, and a timing run and a profiling run"
log "  should not be the same run."

if [ -z "${NVTX_CAPTURE}" ]; then
  log ""
  log "  --nvtx-capture was not given, so the whole process is captured. That"
  log "  includes model loading, the ONNX export, and every warmup batch, which"
  log "  together are usually longer than the part worth looking at. Push a"
  log "  range around the timed section and pass its name to limit the capture."
fi

step "commands"
log "  ${WORKLOAD}"
log ""
if [ "${MODE}" != "compute" ]; then
  log "  nsys ${NSYS_FLAGS} ${WORKLOAD}"
  log "  nsys stats ${NSYS_REPORTS} --format csv --output ${RUN_DIR}/${BASE} ${RUN_DIR}/${BASE}.nsys-rep"
fi
if [ "${MODE}" != "systems" ]; then
  log "  ncu ${NCU_FLAGS} ${WORKLOAD}"
fi

if [ "${DRY_RUN}" -eq 1 ]; then
  log ""
  log "dry run. Nothing was executed."
  exit 0
fi

mkdir -p "${RUN_DIR}"

# ---------------------------------------------------------------------------
# Nsight Systems
# ---------------------------------------------------------------------------

if [ "${MODE}" != "compute" ]; then
  step "Nsight Systems capture"
  # Word splitting on the flag strings is intended. Every value in them was set
  # by this script or came from a flag, and none contains a space.
  # shellcheck disable=SC2086
  if ! nsys ${NSYS_FLAGS} ${WORKLOAD} 2>&1 | tee "${RUN_DIR}/nsys.log"; then
    die "the Nsight Systems capture failed. See ${RUN_DIR}/nsys.log"
  fi

  if [ ! -f "${RUN_DIR}/${BASE}.nsys-rep" ]; then
    die "nsys reported success and wrote no ${RUN_DIR}/${BASE}.nsys-rep"
  fi

  if [ "${SKIP_SUMMARY}" -eq 0 ]; then
    step "nsys stats"
    # shellcheck disable=SC2086
    if ! nsys stats ${NSYS_REPORTS} --format csv --output "${RUN_DIR}/${BASE}" \
        "${RUN_DIR}/${BASE}.nsys-rep" 2>&1 | tee "${RUN_DIR}/nsys_stats.log"; then
      log "  nsys stats failed. The capture is still at ${RUN_DIR}/${BASE}.nsys-rep"
      log "  and opens in the Nsight Systems GUI. Only the markdown summary is missing."
    else
      step "markdown summary"
      GPU_LABEL="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 1 || echo "unknown")"
      "${PYTHON}" tools/summarize_profile.py "${RUN_DIR}" \
        --output "${RUN_DIR}/profile_summary.md" \
        --hardware "${GPU_LABEL}" \
        --capture "${BASE}.nsys-rep" \
        --title "DeepFM inference profile at batch ${BATCH_SIZE}"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Nsight Compute
# ---------------------------------------------------------------------------

if [ "${MODE}" != "systems" ]; then
  step "Nsight Compute"
  log "  Nsight Compute replays each kernel to collect counters, so this is slow"
  log "  by construction and the numbers it reports are per kernel rather than"
  log "  per request. It is not a latency measurement and it must never be"
  log "  quoted as one."
  log ""
  # shellcheck disable=SC2086
  if ! ncu ${NCU_FLAGS} --csv --log-file "${RUN_DIR}/${BASE}_ncu.csv" ${WORKLOAD} 2>&1 \
      | tee "${RUN_DIR}/ncu.log"; then
    log ""
    log "  the Nsight Compute run failed. See ${RUN_DIR}/ncu.log"
    log ""
    log "  The usual cause is counter permissions. ERR_NVGPUCTRPERM in that log"
    log "  means the driver is refusing to expose the performance counters to a"
    log "  non admin user. Run the container with --cap-add=SYS_ADMIN, or set"
    log "  the driver module parameter NVreg_RestrictProfilingToAdminUsers to 0"
    log "  and reboot the host."
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# What came out
# ---------------------------------------------------------------------------

step "artifacts"
log "  ${RUN_DIR}"
# find rather than ls, because parsing the output of ls breaks on file names
# that shell would rather split on.
find "${RUN_DIR}" -maxdepth 1 -mindepth 1 -exec basename {} \; | sort | sed 's/^/    /'
log ""
log "  Open the timeline with"
log "    nsys-ui ${RUN_DIR}/${BASE}.nsys-rep"
if [ "${MODE}" != "systems" ]; then
  log "    ncu-ui ${RUN_DIR}/${BASE}_ncu.ncu-rep"
fi
log ""
log "  Every number in this directory belongs to the GPU named in the plan"
log "  above and to no other. See docs/BENCHMARK_AUTOMATION.md for how to read"
log "  a timeline for this workload."
