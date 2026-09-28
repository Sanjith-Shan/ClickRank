# Numbers

<!-- generated:begin -->

## Data

From `results/retrieval/data_summary.jsonl`. Counts match the dataset card.

| Impressions | Clicks | CTR | Users | Ads in corpus | Train (days 1 to 7) | Test (day 8) |
| --- | --- | --- | --- | --- | --- | --- |
| 26,557,961 | 1,366,056 | 5.1% | 1,141,729 | 846,811 | 23,249,296 | 3,308,665 |

## Retrieval hit rate on the test day

From `results/retrieval/hit_rate.jsonl`. The share of the 166,280 test day (user, clicked ad) pairs whose ad is in that user's top K of 846,811 ads, by exact search on the tower embeddings. Popularity returns the K most clicked ads of the training days to every user.

| K | Two tower | Popularity | Chance | Candidates scored, fewer than all ads |
| --- | --- | --- | --- | --- |
| 50 | 8.9% | 3.8% | 0.006% | 16,936x |
| 100 | 11.7% | 5.9% | 0.012% | 8,468x |
| 500 | 19.6% | 12.5% | 0.059% | 1,694x |

Sanity check, `results/retrieval/two_tower_sanity.jsonl`. A user's clicked ad scores above a random ad for 87.2% of test pairs.

## FAISS index sweep

From `results/retrieval/index_sweep.jsonl`. Recall is against exact search (`IndexFlatIP`) on the same embeddings over 20,000 test day users. Latency is one query at a time on one thread. The sweep ran beside other training jobs, so the load column matters, and the clean latencies are the two stage ones below.

| Index | Size | Recall@50 | Recall@100 | Recall@500 | Hit rate@100 | p50 @100, 1 thread | Machine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flat | 216.8 MB | 1.000 | 1.000 | 1.000 | 11.5% | 6.46 ms | Apple M3 Pro, load 18.97 |
| ivf_flat nlist=2048 nprobe=1 | 224.1 MB | 0.464 | 0.400 | 0.218 | 6.9% | 0.03 ms | Apple M3 Pro, load 17.26 |
| ivf_flat nlist=2048 nprobe=4 | 224.1 MB | 0.685 | 0.635 | 0.498 | 9.3% | 0.05 ms | Apple M3 Pro, load 18.12 |
| ivf_flat nlist=2048 nprobe=16 | 224.1 MB | 0.817 | 0.785 | 0.711 | 10.6% | 0.10 ms | Apple M3 Pro, load 18.59 |
| ivf_flat nlist=2048 nprobe=64 | 224.1 MB | 0.916 | 0.899 | 0.857 | 11.2% | 0.35 ms | Apple M3 Pro, load 18.21 |
| ivf_flat nlist=2048 nprobe=256 | 224.1 MB | 0.979 | 0.972 | 0.955 | 11.5% | 1.48 ms | Apple M3 Pro, load 22.27 |
| ivf_flat nlist=4096 nprobe=1 | 224.6 MB | 0.400 | 0.322 | 0.145 | 5.9% | 0.03 ms | Apple M3 Pro, load 35.12 |
| ivf_flat nlist=4096 nprobe=4 | 224.6 MB | 0.660 | 0.596 | 0.403 | 8.7% | 0.05 ms | Apple M3 Pro, load 32.79 |
| ivf_flat nlist=4096 nprobe=16 | 224.6 MB | 0.798 | 0.756 | 0.658 | 10.2% | 0.12 ms | Apple M3 Pro, load 31.71 |
| ivf_flat nlist=4096 nprobe=64 | 224.6 MB | 0.893 | 0.869 | 0.817 | 11.1% | 0.33 ms | Apple M3 Pro, load 37.89 |
| ivf_flat nlist=4096 nprobe=256 | 224.6 MB | 0.958 | 0.952 | 0.928 | 11.4% | 0.98 ms | Apple M3 Pro, load 37.53 |
| ivf_flat nlist=8192 nprobe=1 | 225.7 MB | 0.296 | 0.226 | 0.085 | 5.0% | 0.06 ms | Apple M3 Pro, load 42.68 |
| ivf_flat nlist=8192 nprobe=4 | 225.7 MB | 0.610 | 0.527 | 0.293 | 8.6% | 0.06 ms | Apple M3 Pro, load 40.7 |
| ivf_flat nlist=8192 nprobe=16 | 225.7 MB | 0.778 | 0.728 | 0.593 | 10.1% | 0.11 ms | Apple M3 Pro, load 35.37 |
| ivf_flat nlist=8192 nprobe=64 | 225.7 MB | 0.880 | 0.849 | 0.778 | 10.9% | 0.21 ms | Apple M3 Pro, load 33.98 |
| ivf_flat nlist=8192 nprobe=256 | 225.7 MB | 0.951 | 0.939 | 0.901 | 11.3% | 0.76 ms | Apple M3 Pro, load 34.02 |
| hnsw M=16 efConstruction=200 efSearch=16 | 338.9 MB | 0.524 | 0.432 | 0.197 | 8.3% | 0.06 ms | Apple M3 Pro, load 61.88 |
| hnsw M=16 efConstruction=200 efSearch=32 | 338.9 MB | 0.650 | 0.574 | 0.292 | 9.2% | 0.09 ms | Apple M3 Pro, load 64.77 |
| hnsw M=16 efConstruction=200 efSearch=64 | 338.9 MB | 0.739 | 0.702 | 0.426 | 9.7% | 0.15 ms | Apple M3 Pro, load 71.74 |
| hnsw M=16 efConstruction=200 efSearch=128 | 338.9 MB | 0.809 | 0.796 | 0.594 | 10.2% | 0.26 ms | Apple M3 Pro, load 72.07 |
| hnsw M=16 efConstruction=200 efSearch=256 | 338.9 MB | 0.857 | 0.853 | 0.753 | 10.6% | 0.38 ms | Apple M3 Pro, load 70.26 |
| hnsw M=16 efConstruction=200 efSearch=512 | 338.9 MB | 0.915 | 0.913 | 0.878 | 11.0% | 0.87 ms | Apple M3 Pro, load 63.44 |
| hnsw M=32 efConstruction=200 efSearch=16 | 447.2 MB | 0.573 | 0.471 | 0.215 | 8.5% | 0.06 ms | Apple M3 Pro, load 48.71 |
| hnsw M=32 efConstruction=200 efSearch=32 | 447.2 MB | 0.691 | 0.611 | 0.314 | 9.4% | 0.09 ms | Apple M3 Pro, load 46.57 |
| hnsw M=32 efConstruction=200 efSearch=64 | 447.2 MB | 0.778 | 0.739 | 0.456 | 10.1% | 0.12 ms | Apple M3 Pro, load 44.04 |
| hnsw M=32 efConstruction=200 efSearch=128 | 447.2 MB | 0.828 | 0.813 | 0.618 | 10.5% | 0.22 ms | Apple M3 Pro, load 41.07 |
| hnsw M=32 efConstruction=200 efSearch=256 | 447.2 MB | 0.884 | 0.881 | 0.781 | 10.9% | 0.42 ms | Apple M3 Pro, load 37.06 |
| hnsw M=32 efConstruction=200 efSearch=512 | 447.2 MB | 0.937 | 0.935 | 0.895 | 11.2% | 0.86 ms | Apple M3 Pro, load 33.18 |
| ivf_pq nlist=4096 m=16 nbits=8 nprobe=4 | 21.5 MB | 0.586 | 0.549 | 0.396 | 8.6% | 0.05 ms | Apple M3 Pro, load 24.31 |
| ivf_pq nlist=4096 m=16 nbits=8 nprobe=16 | 21.5 MB | 0.668 | 0.658 | 0.610 | 10.0% | 0.08 ms | Apple M3 Pro, load 23.33 |
| ivf_pq nlist=4096 m=16 nbits=8 nprobe=64 | 21.5 MB | 0.710 | 0.720 | 0.721 | 10.6% | 0.18 ms | Apple M3 Pro, load 22.44 |
| ivf_pq nlist=4096 m=32 nbits=8 nprobe=4 | 35.0 MB | 0.646 | 0.587 | 0.402 | 8.7% | 0.06 ms | Apple M3 Pro, load 21.7 |
| ivf_pq nlist=4096 m=32 nbits=8 nprobe=16 | 35.0 MB | 0.768 | 0.734 | 0.650 | 10.3% | 0.10 ms | Apple M3 Pro, load 20.61 |
| ivf_pq nlist=4096 m=32 nbits=8 nprobe=64 | 35.0 MB | 0.845 | 0.834 | 0.797 | 11.1% | 0.25 ms | Apple M3 Pro, load 20.82 |

## Rankers on the test day

From `results/retrieval/ranker_eval.jsonl`. Every test day impression, scored exhaustively.

| Model | AUC | GAUC by user | NE |
| --- | --- | --- | --- |
| deepfm | 0.6110 | 0.5516 | 0.9843 |
| dcn | 0.6065 | 0.5533 | 0.9884 |

## Freshness

From `results/retrieval/freshness.jsonl`. DeepFM, test day 8.

| Trained on day | Days stale | AUC | NE | NE vs freshest |
| --- | --- | --- | --- | --- |
| 1 | 7 | 0.5657 | 1.0055 |  |
| 2 | 6 | 0.5682 | 1.0088 |  |
| 3 | 5 | 0.5748 | 1.0003 |  |
| 4 | 4 | 0.5751 | 1.0029 |  |
| 5 | 3 | 0.5815 | 0.9998 |  |
| 6 | 2 | 0.5847 | 0.9944 |  |
| 7 | 1 | 0.5871 | 0.9925 |  |

<!-- generated:end -->
