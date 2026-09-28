#!/usr/bin/env bash
# Run the whole ClickRank inference benchmark procedure end to end.
#
# The procedure is four steps that have to happen in this order and that are
# easy to get wrong by hand. Pick the config that matches the machine. Build the
# TensorRT engines first, on a GPU box, so the build cost is measured on its own
# and never lands inside an inference timing. Run the sweep. Compare the result
# against the committed baseline and fail if something got worse. Then archive
# everything so the run can be looked at again later.
#
# Doing that by hand is how a benchmark stops being run. Every step here already
# works on its own, and this script exists so that all four happen together, in
# the right order, with the right config, every time.
#
# The shebang is bash rather than sh because set -o pipefail is not in POSIX and
# a silent failure inside a pipeline is exactly the class of bug this script
# must not have. Everything below the shebang stays close to POSIX shell. There
# is no [[ ]], no process substitution, and no array, with one exception. The
# regression gate is run through tee so its report lands in the archive, and
# reading the gate's own exit code out of that pipeline needs PIPESTATUS, which
# is the second and last bashism in the file.
#
#   bash scripts/sweep.sh --help
#   bash scripts/sweep.sh
#   bash scripts/sweep.sh --config cpu_only --repeats 30
#   bash scripts/sweep.sh --config gpu_full
#
# The exit code is the regression gate's exit code, so this script can be the
# whole of a CI step. See docs/BENCHMARK_AUTOMATION.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python3}"

CONFIG="auto"
ARCHIVE_ROOT="${REPO_ROOT}/results/sweeps"
BASELINE=""
SAMPLE_SIZE=""
REPEATS=""
WARMUP=""
UPDATE_BASELINE=0
SKIP_ENGINES=0
SKIP_GATE=0
ALLOW_HW_MISMATCH=0
DRY_RUN=0

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

log() {
  printf '%s\n' "$*"
}

step() {
  printf '\n== %s ==\n' "$*"
}

die() {
  printf 'sweep.sh error. %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'USAGE'
sweep.sh. Run the ClickRank inference benchmark procedure end to end.

  usage
    bash scripts/sweep.sh [options]

  What it does, in order.
    1. Detects whether this machine has an NVIDIA GPU and picks the matching
       sweep config from benchmarks/ unless one was named.
    2. Builds the TensorRT engines, on a GPU box only, so the build cost is
       measured on its own rather than inside an inference timing.
    3. Runs the sweep with the flags translated from the config.
    4. Runs the regression gate against the committed baseline.
    5. Archives the run into a timestamped directory and points results/sweeps/latest at it.

  options
    -c, --config NAME         Config to run. A name such as cpu_only or gpu_full,
                              or a path to a yaml file. Default is auto, which
                              picks gpu_full on a machine with a working
                              nvidia-smi and cpu_only everywhere else.
    -a, --archive-dir DIR     Where the timestamped run directory is created.
                              Default results/sweeps
    -b, --baseline PATH       Baseline json for the regression gate.
                              Default results/baselines/<config name>.json
        --sample-size N       Override the row count from the config.
        --repeats N           Override the timed pass count from the config.
                              More repeats is the only real fix for a noisy
                              gate, so this is the knob worth raising.
        --warmup N            Override the warmup batch count from the config.
        --update-baseline     Record this run as the new baseline instead of
                              gating on it. Loud on purpose.
        --allow-hardware-mismatch
                              Let the gate compare against a baseline recorded
                              on another machine. Every latency finding then
                              compares two machines rather than two builds.
        --skip-engines        Do not build TensorRT engines even on a GPU box.
        --skip-gate           Run the sweep and stop before the regression gate.
        --dry-run             Print the resolved plan and the exact commands,
                              then exit without running anything.
    -h, --help                This message.

  environment
    PYTHON                    Interpreter to use. Default python3

  exit codes
    The regression gate's exit code is this script's exit code.
      0  no regression
      1  usage or input error, here or in the gate
      2  latency regression
      3  accuracy regression
      4  a cell in the baseline did not run
      5  the baseline came from different hardware
USAGE
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

require_value() {
  # $1 is the flag name and $2 is the count of remaining arguments.
  if [ "$2" -lt 2 ]; then
    die "$1 needs a value. Run bash scripts/sweep.sh --help"
  fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    -c|--config)
      require_value "$1" $#
      CONFIG="$2"
      shift 2
      ;;
    -a|--archive-dir)
      require_value "$1" $#
      ARCHIVE_ROOT="$2"
      shift 2
      ;;
    -b|--baseline)
      require_value "$1" $#
      BASELINE="$2"
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
    --update-baseline)
      UPDATE_BASELINE=1
      shift
      ;;
    --allow-hardware-mismatch)
      ALLOW_HW_MISMATCH=1
      shift
      ;;
    --skip-engines)
      SKIP_ENGINES=1
      shift
      ;;
    --skip-gate)
      SKIP_GATE=1
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
      die "unknown option $1. Run bash scripts/sweep.sh --help"
      ;;
    *)
      die "unexpected argument $1. Every input is a flag. Run bash scripts/sweep.sh --help"
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  die "no interpreter named ${PYTHON} on PATH. Set PYTHON to the one to use."
fi

cd "${REPO_ROOT}"

for required in \
  "scripts/run_inference_benchmark.py" \
  "scripts/check_regression.py" \
  "tools/sweep_config.py"
do
  if [ ! -f "${required}" ]; then
    die "${required} is missing. This script must run from a complete checkout."
  fi
done

# GPU detection. nvidia-smi being on PATH is not enough on its own, because a
# container can carry the binary with no device visible to it, so the device
# listing has to succeed too. Anything else is the cpu lane, and that includes
# this project's development machine, which is Apple Silicon and has no CUDA at
# all rather than a CUDA setup that is broken.
LANE="cpu"
GPU_NAME="none detected"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  if nvidia-smi -L 2>/dev/null | grep -q "GPU 0"; then
    LANE="gpu"
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)"
  fi
fi

if [ "${CONFIG}" = "auto" ]; then
  if [ "${LANE}" = "gpu" ]; then
    CONFIG="gpu_full"
  else
    CONFIG="cpu_only"
  fi
  AUTO_PICKED="yes"
else
  AUTO_PICKED="no"
fi

# The config is the source of truth for what a run is, so it is parsed once here
# and every downstream command is built from what comes back. A bad config fails
# now rather than after the benchmark has already spent twenty minutes.
if ! CONFIG_NAME="$("${PYTHON}" tools/sweep_config.py "${CONFIG}" --print name --quiet)"; then
  die "could not read the sweep config ${CONFIG}. The message above says why."
fi
CONFIG_HARDWARE="$("${PYTHON}" tools/sweep_config.py "${CONFIG}" --print hardware --quiet)"
# The config's own --output is dropped here. This script substitutes its own
# timestamped run directory below, and a command line carrying the same flag
# twice is a command line a reader stops trusting.
CONFIG_FLAGS="$("${PYTHON}" tools/sweep_config.py "${CONFIG}" --print flags --omit=--output)"

if [ -z "${BASELINE}" ]; then
  BASELINE="results/baselines/${CONFIG_NAME}.json"
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${ARCHIVE_ROOT}/${STAMP}-${CONFIG_NAME}"
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo "not a git checkout")"
GIT_DIRTY="clean"
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
  GIT_DIRTY="dirty"
fi

# The sweep writes into the run directory rather than into the config's output
# directory. The run directory is the artifact, it is timestamped, and nothing
# else in the repository writes there, so two sweeps never overwrite each other
# and a shared results file is never clobbered by a run that was only meant to
# be a check.
BENCH_FLAGS="${CONFIG_FLAGS} --output ${RUN_DIR}"
if [ -n "${SAMPLE_SIZE}" ]; then
  BENCH_FLAGS="${BENCH_FLAGS} --sample-size ${SAMPLE_SIZE}"
fi
if [ -n "${REPEATS}" ]; then
  BENCH_FLAGS="${BENCH_FLAGS} --repeats ${REPEATS}"
fi
if [ -n "${WARMUP}" ]; then
  BENCH_FLAGS="${BENCH_FLAGS} --warmup ${WARMUP}"
fi

RESULTS_JSON="${RUN_DIR}/inference_benchmark.json"

GATE_FLAGS="--results ${RESULTS_JSON} --baseline ${BASELINE} --json-out ${RUN_DIR}/regression.json"
if [ "${ALLOW_HW_MISMATCH}" -eq 1 ]; then
  GATE_FLAGS="${GATE_FLAGS} --allow-hardware-mismatch"
fi
if [ "${UPDATE_BASELINE}" -eq 1 ]; then
  GATE_FLAGS="${GATE_FLAGS} --update-baseline"
fi

# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

step "plan"
log "  repository     ${REPO_ROOT}"
log "  git            ${GIT_SHA} (${GIT_DIRTY})"
log "  interpreter    ${PYTHON}"
log "  detected lane  ${LANE}"
log "  gpu            ${GPU_NAME}"
log "  config         ${CONFIG_NAME} (auto picked ${AUTO_PICKED})"
log "  config says    ${CONFIG_HARDWARE}"
log "  run directory  ${RUN_DIR}"
log "  baseline       ${BASELINE}"
if [ "${UPDATE_BASELINE}" -eq 1 ]; then
  log "  gate           the baseline will be REWRITTEN from this run"
elif [ "${SKIP_GATE}" -eq 1 ]; then
  log "  gate           skipped"
else
  log "  gate           this run will be checked against the baseline"
fi

if [ "${LANE}" = "cpu" ] && [ "${CONFIG_NAME}" = "gpu_full" ]; then
  log ""
  log "  warning. the gpu_full config was asked for on a machine with no GPU."
  log "  The benchmark will run the cpu backends and report every gpu backend as"
  log "  unavailable with the reason the runtime gave. Nothing is faked, and the"
  log "  gpu rows will be missing rather than estimated."
fi

if [ "${LANE}" = "gpu" ] && [ "${SKIP_ENGINES}" -eq 0 ]; then
  log ""
  log "  engines will be built first, so the build cost is measured on its own."
fi

step "commands"
log "  ${PYTHON} scripts/build_trt_engines.py    (gpu lane only)"
log "  ${PYTHON} scripts/run_inference_benchmark.py ${BENCH_FLAGS}"
if [ "${SKIP_GATE}" -eq 0 ]; then
  log "  ${PYTHON} scripts/check_regression.py ${GATE_FLAGS}"
fi

if [ "${DRY_RUN}" -eq 1 ]; then
  log ""
  log "dry run. Nothing was executed."
  exit 0
fi

mkdir -p "${RUN_DIR}"

# ---------------------------------------------------------------------------
# 1. Engines
# ---------------------------------------------------------------------------

ENGINE_STATUS="skipped, cpu lane"
if [ "${LANE}" = "gpu" ] && [ "${SKIP_ENGINES}" -eq 0 ]; then
  step "building TensorRT engines"
  if "${PYTHON}" scripts/build_trt_engines.py 2>&1 | tee "${RUN_DIR}/engines.log"; then
    ENGINE_STATUS="built"
  else
    ENGINE_STATUS="failed"
    die "the engine build failed. See ${RUN_DIR}/engines.log. The sweep was not run, because a sweep against stale engines reports a number for a build that is not the one under test."
  fi
elif [ "${SKIP_ENGINES}" -eq 1 ]; then
  ENGINE_STATUS="skipped, --skip-engines"
  step "skipping the engine build"
  log "  --skip-engines was passed. Any TensorRT row will use whatever engine is already on disk."
else
  step "skipping the engine build"
  log "  no NVIDIA GPU was detected, so there is no engine to build."
fi

# ---------------------------------------------------------------------------
# 2. The sweep
# ---------------------------------------------------------------------------

step "running the sweep"
log "  ${PYTHON} scripts/run_inference_benchmark.py ${BENCH_FLAGS}"
log ""

SWEEP_START="$(date -u +%s)"
# Word splitting on BENCH_FLAGS is intended. The flag string is built above from
# a config this script parsed, not from anything a caller typed, and none of the
# values in the schema contain a space.
# shellcheck disable=SC2086
if ! "${PYTHON}" scripts/run_inference_benchmark.py ${BENCH_FLAGS} 2>&1 | tee "${RUN_DIR}/sweep.log"; then
  die "the sweep failed. See ${RUN_DIR}/sweep.log"
fi
SWEEP_END="$(date -u +%s)"
SWEEP_SECONDS="$((SWEEP_END - SWEEP_START))"

if [ ! -f "${RESULTS_JSON}" ]; then
  die "the sweep finished and wrote no ${RESULTS_JSON}. Nothing can be checked."
fi

log ""
log "  sweep finished in ${SWEEP_SECONDS} seconds"

# ---------------------------------------------------------------------------
# 3. The regression gate
# ---------------------------------------------------------------------------

GATE_CODE=0
GATE_STATUS="skipped"
if [ "${SKIP_GATE}" -eq 1 ]; then
  step "skipping the regression gate"
  log "  --skip-gate was passed, so nothing was compared against ${BASELINE}."
elif [ "${UPDATE_BASELINE}" -eq 0 ] && [ ! -f "${BASELINE}" ]; then
  step "no baseline yet"
  GATE_STATUS="no baseline"
  log "  there is no baseline at ${BASELINE}, so there is nothing to compare against."
  log "  record this run as the baseline, after reading it, with"
  log ""
  log "    ${PYTHON} scripts/check_regression.py --results ${RESULTS_JSON} --baseline ${BASELINE} --update-baseline"
  log ""
else
  step "regression gate"
  # The gate exits non zero by design, so the pipeline status has to be captured
  # rather than allowed to kill the script under set -e. The archive step below
  # has to run whatever the gate says, because a failing run is the run whose
  # artifacts are most worth keeping.
  set +e
  # shellcheck disable=SC2086
  "${PYTHON}" scripts/check_regression.py ${GATE_FLAGS} 2>&1 | tee "${RUN_DIR}/regression.txt"
  GATE_CODE="${PIPESTATUS[0]}"
  set -e
  case "${GATE_CODE}" in
    0) GATE_STATUS="pass" ;;
    1) GATE_STATUS="usage or input error" ;;
    2) GATE_STATUS="latency regression" ;;
    3) GATE_STATUS="accuracy regression" ;;
    4) GATE_STATUS="a baseline cell did not run" ;;
    5) GATE_STATUS="hardware mismatch" ;;
    *) GATE_STATUS="unexpected exit code ${GATE_CODE}" ;;
  esac
fi

# ---------------------------------------------------------------------------
# 4. Archive
# ---------------------------------------------------------------------------

step "archiving"

CONFIG_FILE="$("${PYTHON}" -c "import sys; sys.path.insert(0, '.'); import tools.sweep_config as c; print(c.config_path(sys.argv[1]))" "${CONFIG}")"
cp "${CONFIG_FILE}" "${RUN_DIR}/config.yaml"

# The manifest is what makes an archived run readable a year later. It records
# what was run, on what, from which commit, and what the gate said, so nobody
# has to reconstruct any of that from a directory name.
{
  printf 'ClickRank sweep run\n'
  printf '\n'
  printf 'started utc        %s\n' "${STAMP}"
  printf 'config             %s\n' "${CONFIG_NAME}"
  printf 'config file        %s\n' "${CONFIG_FILE}"
  printf 'config hardware    %s\n' "${CONFIG_HARDWARE}"
  printf 'detected lane      %s\n' "${LANE}"
  printf 'gpu                %s\n' "${GPU_NAME}"
  printf 'host               %s\n' "$(uname -srm)"
  printf 'interpreter        %s\n' "$("${PYTHON}" --version 2>&1)"
  printf 'git commit         %s\n' "${GIT_SHA}"
  printf 'git tree           %s\n' "${GIT_DIRTY}"
  printf 'engines            %s\n' "${ENGINE_STATUS}"
  printf 'sweep seconds      %s\n' "${SWEEP_SECONDS}"
  printf 'baseline           %s\n' "${BASELINE}"
  printf 'gate               %s\n' "${GATE_STATUS}"
  printf 'gate exit code     %s\n' "${GATE_CODE}"
  printf '\n'
  printf 'benchmark command\n'
  printf '  %s scripts/run_inference_benchmark.py %s\n' "${PYTHON}" "${BENCH_FLAGS}"
  printf '\n'
  printf 'A latency number belongs to the machine that produced it. The host and\n'
  printf 'the gpu above are that machine, and every number in this directory is\n'
  printf 'scoped to it.\n'
} > "${RUN_DIR}/run.txt"

# A stable path to the newest run, so a follow up command does not have to know
# the timestamp. Removed and recreated rather than forced, because ln -n is not
# portable across every ln this script might meet.
rm -f "${ARCHIVE_ROOT}/latest"
ln -s "${STAMP}-${CONFIG_NAME}" "${ARCHIVE_ROOT}/latest" 2>/dev/null || true

log "  run directory  ${RUN_DIR}"
log "  manifest       ${RUN_DIR}/run.txt"
log "  newest run     ${ARCHIVE_ROOT}/latest"
log ""
# find rather than ls, because parsing the output of ls breaks on file names
# that shell would rather split on.
find "${RUN_DIR}" -maxdepth 1 -mindepth 1 -exec basename {} \; | sort | sed 's/^/    /'

step "result"
log "  config    ${CONFIG_NAME}"
log "  lane      ${LANE}"
log "  gate      ${GATE_STATUS}"
log "  artifacts ${RUN_DIR}"

exit "${GATE_CODE}"
