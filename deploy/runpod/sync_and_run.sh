#!/usr/bin/env bash
#
# Sync this repository to a RunPod pod, run the GPU measurement there, and pull
# the results back.
#
# Run this from your own machine, from the repository root. It never creates or
# destroys a pod. A separate session owns pod lifecycle, and a second session
# deploying its own pod is how a small budget disappears in an hour.
#
# The working rule is that you do not develop on the pod. GPU time bills by the
# second and most of a session is thinking rather than computing, so code is
# written locally, synced up, run, and the results come back down.
#
# Note that basic proxied RunPod SSH does not support scp or rsync. The pod
# needs a public IP with TCP 22 exposed, and the host and port below have to
# come from the SSH over exposed TCP line rather than the proxied one.
#
# Usage
#     bash deploy/runpod/sync_and_run.sh --host 1.2.3.4 --port 12345
#     bash deploy/runpod/sync_and_run.sh --host 1.2.3.4 --port 12345 --bootstrap
#     bash deploy/runpod/sync_and_run.sh --host 1.2.3.4 --port 12345 --dry-run

set -euo pipefail

HOST=""
PORT=""
KEY="${HOME}/.ssh/id_ed25519"
USER_NAME="root"
REMOTE_DIR="/workspace/ClickRank"
RUN_BOOTSTRAP=0
DRY_RUN=0
SAMPLE_SIZE=""

MODELS="deepfm dcn"
PRECISIONS="fp32 fp16 int8"
BATCH_SIZES="1 32 256 1024 4096"

usage() { sed -n "2,20p" "$0" | sed -e "s/^#\{0,1\} \{0,1\}//"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --host)        HOST="$2"; shift 2 ;;
        --port)        PORT="$2"; shift 2 ;;
        --key)         KEY="$2"; shift 2 ;;
        --user)        USER_NAME="$2"; shift 2 ;;
        --remote-dir)  REMOTE_DIR="$2"; shift 2 ;;
        --models)      MODELS="$2"; shift 2 ;;
        --precisions)  PRECISIONS="$2"; shift 2 ;;
        --batch-sizes) BATCH_SIZES="$2"; shift 2 ;;
        --sample-size) SAMPLE_SIZE="$2"; shift 2 ;;
        --bootstrap)   RUN_BOOTSTRAP=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "unknown argument $1. Try --help." >&2; exit 2 ;;
    esac
done

if [ -z "$HOST" ] || [ -z "$PORT" ]; then
    echo "ERROR --host and --port are both required." >&2
    echo "They come from the pod's SSH over exposed TCP line, not the proxied one." >&2
    echo "If no pod is listed in campaign/specs/RUNPOD_SHARED.md then there is nothing" >&2
    echo "to measure on. Say so and stop rather than deploying your own." >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SSH_OPTS="-p ${PORT} -i ${KEY} -o StrictHostKeyChecking=accept-new"
TARGET="${USER_NAME}@${HOST}"

say() { printf '\n=== %s ===\n' "$1"; }

run() {
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '  [dry run] %s\n' "$*"
    else
        "$@"
    fi
}

say "checking the pod answers on ssh"
# The pod's sshd lags its RUNNING status by two to three minutes, so the status
# field is not a reliable readiness signal. Poll the real thing instead.
if [ "$DRY_RUN" -eq 0 ]; then
    if ! ssh $SSH_OPTS -o ConnectTimeout=10 "$TARGET" "echo reachable" 2>/dev/null; then
        echo "ERROR cannot reach ${TARGET} on port ${PORT}." >&2
        echo "If the pod was only just created, its sshd usually needs another two to" >&2
        echo "three minutes even though the dashboard already says RUNNING." >&2
        exit 1
    fi
fi

say "syncing code up"
# data and results are excluded deliberately. The dataset is downloaded on the
# pod rather than pushed across the wire, and results only ever travel downward
# so that a stale local file cannot overwrite a fresh measurement.
run rsync -avz --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    --exclude '.DS_Store' \
    --exclude 'data/' \
    --exclude 'results/' \
    --exclude '*.pt' \
    --exclude '*.onnx' \
    --exclude '*.engine' \
    -e "ssh ${SSH_OPTS}" \
    "${REPO_ROOT}/" "${TARGET}:${REMOTE_DIR}/"

if [ "$RUN_BOOTSTRAP" -eq 1 ]; then
    say "bootstrapping the pod"
    run ssh $SSH_OPTS "$TARGET" \
        "cd ${REMOTE_DIR} && bash deploy/runpod/bootstrap.sh"
fi

SAMPLE_FLAG=""
if [ -n "$SAMPLE_SIZE" ]; then
    SAMPLE_FLAG="--sample-size ${SAMPLE_SIZE}"
fi

say "building engines"
run ssh $SSH_OPTS "$TARGET" "cd ${REMOTE_DIR} && \
    export PATH=/usr/local/cuda/bin:\$PATH && \
    python3 scripts/build_trt_engines.py --models ${MODELS} --precisions ${PRECISIONS}"

say "running the sweep"
run ssh $SSH_OPTS "$TARGET" "cd ${REMOTE_DIR} && \
    export PATH=/usr/local/cuda/bin:\$PATH && \
    python3 scripts/run_inference_benchmark.py --models ${MODELS} \
      --batch-sizes ${BATCH_SIZES} ${SAMPLE_FLAG}"

say "pulling results back"
run rsync -avz -e "ssh ${SSH_OPTS}" \
    "${TARGET}:${REMOTE_DIR}/results/" "${REPO_ROOT}/results/"

say "done"
cat <<EOF
Results are in ${REPO_ROOT}/results.

Before any of these numbers goes into a README, a ledger, or a resume, confirm
the hardware record at the top of results/inference_benchmark.md names the card,
the driver, and the TensorRT version. A number without that record is not usable.

These are rented cloud GPU numbers. Never blend them with the Apple Silicon CPU
figures from this same project.

Terminate the pod when you are finished. Do not merely stop it, because a
stopped pod can still bill for storage. Pod lifecycle belongs to whichever
session owns it.
EOF
