# Bug log

Each entry says what broke, how it showed up, the tool that found it, and the fix. Entries
from before the retrieval work are in the commit history and the docs they were written up
in, for example the silent ONNX Runtime CPU fallback in `docs/INFERENCE.md`.

## 2026-09-28. CI had been red since 2026-08-28

- **Symptom.** Every push since the inference documentation commit failed the `test` job with
  `ModuleNotFoundError: No module named 'fastapi'` while collecting `tests/test_serving.py`.
- **Found by.** GitHub Actions, read with `gh run view --log-failed` during the rename.
- **Cause.** The serving tests import FastAPI and httpx, which live in
  `requirements-serving.txt`, and the test job only installed `requirements.txt`.
- **Fix.** The test job installs both files. The first green run is the one after the rename.

## 2026-09-28. A cut short download extracted into files that looked whole

- **Symptom.** The first Zenodo download was reset mid transfer. `tar` still extracted what
  it had, and `raw_sample.csv` came out with 264,991 lines instead of 26,557,962. The file
  size on disk looked right because the tar header records the full size.
- **Found by.** `wc -l` against the dataset card.
- **Fix.** Downloads resume with `curl -C -` until complete, and the loader now reports the
  rows it read next to the published counts (`check_against_card` in `src/retrieval/data.py`),
  so a short file shows up in `data_summary.jsonl` rather than as a quietly smaller run.

## 2026-09-28. torch and FAISS in one process crashed on macOS

- **Symptom.** `OMP: Error #15` (two OpenMP runtimes initialised), and with
  `KMP_DUPLICATE_LIB_OK=TRUE` a segfault inside IVF and HNSW search.
- **Found by.** pytest, on the first test that imported both.
- **Cause.** The faiss-cpu wheel for macOS ships its own `libomp.dylib` and torch ships
  another. Two copies of the runtime in one process is undefined behaviour, and the
  environment variable only silences the check.
- **Fix, on the development machine.** FAISS's bundled `libomp.dylib` is a symlink to
  torch's, so one runtime is loaded. The original is kept beside it as
  `libomp.dylib.faiss-orig`. Reinstalling faiss-cpu undoes this. Linux CI uses a single
  system runtime and is not affected.

## 2026-09-28. GAUC by user would not finish

- **Symptom.** The existing `group_auc` builds a boolean mask per group, which is quadratic
  in practice. Grouping the Taobao test day by real user means a few hundred thousand groups
  over a few million impressions.
- **Found by.** Reading the code before the first real run, then timing it. On 200,000
  random rows in 20,000 groups the reference took 14.8 s and the replacement 0.5 s on the
  M3 Pro under load. The reference's cost grows with rows times groups, and the test day is
  about 16 times the rows and 20 times the groups of that sample.
- **Fix.** `fast_group_auc` in `src/retrieval/metrics.py` uses the rank sum form of AUC inside
  each group in one pandas pass. `tests/test_fast_gauc.py` pins it to the reference to 1e-12,
  including ties and single class groups.

## 2026-09-28. Old checkpoints no longer loaded into the serving path

- **Symptom.** `tests/test_two_stage_serving.py` failed with `tower checkpoint mismatch:
  missing ['user_feat', 'ad_id_map', 'user_id_map']`.
- **Found by.** pytest, after the id maps were added to the two tower model.
- **Cause.** The serving loader rebuilds the model against an extended user table and loads
  the checkpoint with a strict key check. The new id map buffers were not in older checkpoints,
  and the user map is one row shorter than the extended table.
- **Fix.** `_load_tower_with_cold` treats a missing map as the identity it was trained with,
  and copies the saved user map into the first rows of the extended one.

## 2026-09-28. The laptop shut down under the measurement runs

- **Symptom.** The Mac powered off while the two stage run, the freshness study and another
  project's benchmarks shared it. The two stage run holds the top 500 ids for 391,741 test
  users from two indexes at once, a few GB on top of the encoded log.
- **Found by.** The machine going down, and the partial `results/retrieval/*.jsonl` rows it
  left behind.
- **Fix.** Heavy runs now go one at a time through `logs/guard.sh`, which samples free memory,
  swap and load every 10 s, pauses the job with SIGSTOP when free memory drops below 15% or
  swap grows by more than 1 GB in a minute, and kills it below 7%. Swap level alone was a
  false signal, because macOS keeps swap allocated long after pressure passes, so the first
  version paused a healthy job and never resumed it. The remaining phases reran with
  `--skip quality`, since the quality rows had already been written.

## 2026-09-28. The Windows box could not run the Python stack

- **Symptom.** `ImportError: DLL load failed ... An Application Control policy has blocked
  this file` for numpy and then pandas, in a fresh venv on the always on Windows machine.
- **Found by.** Trying to move ranker training off the laptop.
- **Cause.** Smart App Control is on and blocks unsigned native extensions.
- **Fix.** None in this repo. Every run stayed on the Mac. Turning Smart App Control off is a
  machine wide security change and was left to the owner.

## 2026-09-28. The Criteo download only ever worked on macOS

- **Symptom.** On a Linux GPU box `scripts/download_data.sh` printed "could not download a
  Criteo sample from the mirror" and left no data, although the figshare mirror answered.
- **Found by.** Running it on the A100 pod for the TorchRec run, then streaming the tarball by
  hand, which listed its members fine.
- **Cause.** The script extracts `'*train.txt'` from the stream. bsdtar on macOS treats that as
  a glob, GNU tar does not unless given `--wildcards`, so on Linux it matched nothing and the
  errors went to /dev/null.
- **Fix.** The script passes `--wildcards` when the installed tar is GNU tar.
