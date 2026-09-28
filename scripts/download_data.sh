#!/usr/bin/env bash
# Download a sample of the real Criteo Display Advertising Challenge dataset.
#
# This streams a public figshare mirror of dac.tar.gz and extracts the first N
# rows of the labeled train.txt into data/criteo.csv. Streaming through head
# means only the portion needed is downloaded, so a few million row sample costs
# roughly 1 GB of transfer and about 1 GB on disk rather than the full 11 GB
# extracted file. This keeps it usable on a normal laptop. Pass a row count as
# the first argument. The default is 2 million rows.
#
# Run from anywhere.
#   bash scripts/download_data.sh 2000000
#
# We do not use a global set -e. A failed download must not abort hard. We handle
# failure explicitly and always exit 0 so the benchmark can fall back to
# synthetic data on its own.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
OUT_FILE="${DATA_DIR}/criteo.csv"
ROWS="${1:-2000000}"

# figshare mirror of the Kaggle Display Advertising Challenge dac.tar.gz.
# The archive holds readme.txt, then test.txt, then the labeled train.txt.
URL="https://ndownloader.figshare.com/files/10082655"

mkdir -p "${DATA_DIR}"

if [ -f "${OUT_FILE}" ]; then
  echo "data file already present at ${OUT_FILE}. Nothing to do."
  echo "delete it first if you want to re download."
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required and was not found. Skipping download."
  echo "the benchmark falls back to synthetic data with the --synthetic flag."
  exit 0
fi

echo "streaming the first ${ROWS} labeled rows of real Criteo from figshare."
echo "this downloads roughly 1 GB for a few million rows and can take a few minutes."

# Stream the gzip, extract only train.txt to stdout, keep the first ROWS lines.
# head closes the pipe after ROWS lines which stops the download early.
# GNU tar only matches a glob with --wildcards, and bsdtar on macOS rejects that
# flag, so pick the form the installed tar understands.
if tar --version 2>/dev/null | grep -q "GNU tar"; then
  TAR_GLOB=(--wildcards '*train.txt')
else
  TAR_GLOB=('*train.txt')
fi
curl -sL "${URL}" 2>/dev/null | tar -xzO "${TAR_GLOB[@]}" 2>/dev/null | head -n "${ROWS}" > "${OUT_FILE}"

GOT=$(wc -l < "${OUT_FILE}" 2>/dev/null | tr -d ' ')
if [ "${GOT:-0}" -ge 1 ]; then
  echo "wrote ${GOT} real Criteo rows to ${OUT_FILE}."
  echo "run the benchmark with"
  echo "  python scripts/run_benchmark.py --sample-size ${GOT}"
  exit 0
fi

rm -f "${OUT_FILE}"
echo ""
echo "could not download a Criteo sample from the mirror."
echo "this is fine. the benchmark falls back to synthetic data automatically."
echo "  python scripts/run_benchmark.py --synthetic --sample-size 100000"
exit 0
