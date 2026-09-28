# Two stage serving

`src/serving/two_stage.py` serves retrieval and ranking behind one endpoint. A request names a
user, a placement and an hour. The service runs the user tower, searches the FAISS index for the
top K ads, scores those K with a ranker (DeepFM by default), and returns the best n with their
click probability and the time each stage took.

The request path is `TwoStagePipeline` from `src/retrieval/pipeline.py`, the same class
`scripts/run_two_stage.py` times offline. Served numbers and measured numbers come from one code
path.

## Starting it

It needs the artifacts that `scripts/run_retrieval.py` and `scripts/run_rankers.py` write.

```bash
# synthetic, to try it
python scripts/make_synthetic_ids.py
python scripts/run_retrieval.py --source synthetic --data data/synthetic_ids --batch-size 1024 --epochs 3 --eval-users 2000
python scripts/run_rankers.py --source synthetic --train-rows 200000 --val-rows 30000 --epochs 2 --device cpu
python scripts/serve_two_stage.py --source synthetic --port 0

# real Taobao artifacts, with the HNSW index the sweep saved, and a parity check at startup
python scripts/serve_two_stage.py --source taobao --index "hnsw M=32 efConstruction=200 efSearch=128" --parity 20
```

`--port 0` picks a free port and prints it. When the index sweep has already saved an index with
the same build parameters under `artifacts/`, the service loads it instead of rebuilding.

## The endpoint

`POST /v1/recommend`

```json
{"user_id": 146882, "pid": "430548_1007", "hour": 12, "k": 100, "n": 10}
```

- `user_id` is the raw id from the log.
- `pid` is the raw placement string or its integer code.
- `hour` is Beijing time. When it is left out, the server clock is used.
- `k` is the retrieval depth. It must be between 1 and 1000.
- `n` is the number of results. It must be between 1 and 100.
- Unknown fields and out-of-range values are rejected with 422.

The response carries:

- the ranked ads as raw adgroup ids, each with `p_click`;
- `k`, and how many ads were `retrieved`;
- `cold_start` and `pid_known`;
- `timings_us` for `user_embed`, `search`, `feature_build`, `rank` and `total`.

`GET /healthz` reports what was loaded. That covers the dataset label (`SYNTHETIC` when it is), the index and its serialised size, the ranker, the thread settings, and the timestamp the state is frozen at. `GET /readyz` returns 503 if a startup parity check ran and failed. `GET /metrics` serves Prometheus text by default, or JSON with `?format=json`. It carries request, failure, cold start and unknown placement counters, plus one latency histogram per stage.

## Cold start and state

The service works from a snapshot. Every request sees the user's click history as of the end of the training days, the same state the offline evaluation uses.

A user id the log has never seen is not an error. It is served from an extra user row whose fields are all the out-of-vocabulary code and whose history is empty. An unseen placement likewise gets the out-of-vocabulary placement code. Both are counted in `/metrics`.

## Parity

`parity_check` serves logged test day impressions and compares two scores for each one whose ad was retrieved:

- the served click probability of that ad;
- the ranker's score on the frozen `rows_dataset` row for the same impression.

On the synthetic artifacts the largest difference was 3e-8, which is float32 rounding. The tolerance is 1e-5.

## The load test

`scripts/load_test_two_stage.py` is closed loop. N clients each send a request, wait for the answer, and send again, for D seconds after a warmup that is discarded. It replays (user, placement) pairs sampled from the test day. It reports:

- achieved requests per second;
- the end to end p50, p95 and p99 the clients saw;
- the server stage breakdown taken from each response.

With `--spawn` it starts the service itself on a free port. It writes one row to `serving_load.jsonl` under the retrieval results directory, with the machine and the load average.

```bash
python scripts/load_test_two_stage.py --source synthetic --spawn --clients 4 --duration 10
python scripts/load_test_two_stage.py --source taobao --spawn --index "hnsw M=32 efConstruction=200 efSearch=128" --clients 4 --duration 30
```

## What it does not claim

- It is one process on one laptop. The simulated clients run on the same machine and compete with the server for cores.
- Being closed loop, the tail it reports is a floor on the real tail, the coordinated omission caveat in `scripts/run_load_test.py`.
- The history is a snapshot. No new clicks are streamed in and no ads are re-embedded.
- A synthetic run is labelled `SYNTHETIC` and is a check that the code works, not a result.
