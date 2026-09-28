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
