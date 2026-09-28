# Design

This file says where each idea in ClickRank came from and why the retrieval stage is built
the way it is. The measured figures live in `NUMBERS.md`, each with the results file it came
from. Everything here is the standard design, credited. What the project adds is the
measurement of the two stage trade on real data.

## Sources

**Ranking models and metrics**

- Rendle, *Factorization Machines*, ICDM 2010. The FM model and its O(kn) interaction term.
- Guo, Tang, Ye, Li, He, *DeepFM: A Factorization-Machine based Neural Network for CTR
  Prediction*, IJCAI 2017.
- Wang, Fu, Fu, Wang, *Deep & Cross Network for Ad Click Predictions*, ADKDD 2017.
- He et al., *Practical Lessons from Predicting Clicks on Ads at Facebook*, ADKDD 2014.
  Normalised entropy as the headline calibration aware metric.
- Zhou et al., *Deep Interest Network for Click-Through Rate Prediction*, KDD 2018. Target
  attention over the user's behaviour sequence, and GAUC grouped by user.
- Naumov et al., *Deep Learning Recommendation Model for Personalization and Recommendation
  Systems* (DLRM), 2019. The family the DeepFM and DCN rankers belong to, and the reason the
  inference work treats them as memory bound.

**Retrieval**

- Huang et al., *Learning Deep Structured Semantic Models for Web Search using Clickthrough
  Data* (DSSM), CIKM 2013. The two tower shape and the cosine score kept from
  `src/relevance/`.
- Covington, Adams, Sargin, *Deep Neural Networks for YouTube Recommendations*, RecSys 2016.
  The canonical split into candidate generation and ranking, the user tower built from the
  mean of watched item embeddings, and the argument that candidate generation is judged by
  recall and ranking by calibrated scores.
- Yi et al., *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations*,
  RecSys 2019. In batch sampled softmax, the log q correction, and the streaming frequency
  estimator (their Algorithm 2) implemented in `src/retrieval/two_tower.py`.
- Johnson, Douze, Jegou, *Billion-scale similarity search with GPUs*, 2017, and Douze et
  al., *The Faiss library*, 2024. `IndexFlatIP` as the exact baseline, `IndexIVFFlat`,
  `IndexIVFPQ` and `IndexHNSWFlat` as approximate indexes, and the FAISS wiki's method for
  measuring recall of an approximate index against exact search on the same vectors.
- Jegou, Douze, Schmid, *Product Quantization for Nearest Neighbor Search*, TPAMI 2011. The
  inverted file and PQ codes behind IVF and IVF-PQ.
- Malkov, Yashunin, *Efficient and robust approximate nearest neighbor search using
  Hierarchical Navigable Small World graphs*, TPAMI 2018.

**Industry write ups that set the shape and the vocabulary**

- Meta engineering blog, the Andromeda retrieval posts (December 2024 and October 2025) and
  *From User Sequences to Scaling Laws: A Multi-Stage Architecture for Meta's Ads Ranking*
  (August 2026). A retrieval layer narrows a very large candidate set to thousands, and a
  ranking layer scores those under a latency budget. ClickRank reproduces that shape at
  laptop scale and measures it. It does not claim to reproduce either system.

**Data**

- Alibaba, *Ad Display/Click Data on Taobao.com*, Tianchi dataset 56. The copy used here is
  the open access Zenodo record 10.5281/zenodo.8088629 (CC BY 4.0), which holds the same
  three files. The rows are never committed. Only code and aggregates are.
- Criteo, *Display Advertising Challenge* dataset, for the CTR and inference work.
- Avazu, *Click-Through Rate Prediction* (Kaggle), supported by the loader as a fallback.

## Two datasets, two stories

Criteo has no user id and no ad id, so it cannot support retrieval. The retrieval stage and
the two stage measurement run on Taobao, which has both. The CTR model table, calibration and
the inference runtime comparison are Criteo numbers. No figure blends the two.

## The retrieval stage

**Towers.** The ad tower embeds the ad group id, category, campaign, customer, brand and a
binned price, concatenates them and runs a small MLP to an L2 normalised 64 dimension vector.
The user tower does the same for the user id and the profile fields, plus the mean of the
embeddings of the user's last 20 clicked ads before the request time (Covington et al.). The
score is the dot product of the two, which is cosine similarity.

**Training.** Clicked impressions from days 1 to 7 are the positives. Every other ad in the
batch is a negative (in batch sampled softmax), and each ad's logit is corrected by the log of
its estimated sampling probability, from Yi et al.'s streaming frequency estimator. An id with
no training click keeps a zero embedding instead of its random initial row, so an unseen ad
is represented by its category, brand and price only.

**Selection, without touching the test day.** The first full run peaked after two epochs and
then fell as the id embeddings memorised the training clicks. Every choice below was made
on a separate protocol that trains on days 1 to 6 and validates on day 7, and day 8 was not
looked at. Each row is the best epoch of that run, from `results/retrieval/two_tower_selection.jsonl`,
hit rate at 100 over 20,000 day 7 users.

| Variant | Best epoch | Day 7 hit rate@100 |
| --- | --- | --- |
| Temperature 0.1 (chosen) | 2 | 0.110 |
| Base, temperature 0.05 | 2 | 0.108 |
| 16 dimension id embeddings | 2 | 0.106 |
| Weight decay 1e-6 | 3 | 0.100 |
| No click history in the user tower | 3 | 0.098 |
| No log q correction | 2 | 0.043 |

The sampling bias correction is the largest single effect. Without it the popular ads, which
appear as in batch negatives far more often than their share of clicks, are pushed down and
hit rate falls by more than half. The final model uses temperature 0.1 and trains on days 1
to 7 for exactly two epochs, the epoch count the validation protocol chose.

## What the two stage measurement can and cannot say

Two measurements run on the test day, and they point in opposite directions for a reason
worth stating.

**Impression level AUC and NE drop when retrieval is added.** Every logged impression is
scored, and an ad that retrieval did not return ranks below every ad it did return. The
logged impressions were chosen by the production system that served Taobao in 2017, not by
this retriever, so at K = 500 retrieval keeps only about 14% of them and 20% of their clicks.
The other 80% of clicks tie at the bottom with the dropped ads, and AUC over the log falls
toward 0.5. This metric measures agreement with the logging policy. It
is reported in `NUMBERS.md` because it is the metric the spec asked for, and it is not the
number the design is judged on.

**Per request, retrieval is what makes ranking work.** For a sample of test day clicks the
request is replayed, and the clicked ad's rank is recorded twice. Once when the ranker
scores all 846,811 ads, and once when it scores only the K the retriever returned. A ranker
trained on logged impressions has only ever seen ads that some earlier system chose to
show, so across the whole catalog it cannot tell a plausible ad from an irrelevant one, and
the clicked ad lands deep in the list. Retrieval removes that problem before ranking starts.
This is the argument for two stages in Covington et al., measured here, and it is why
candidate generation is judged by recall and ranking by calibrated scores.

The per request sample is 300 clicks, so its percentages carry roughly plus or minus two
points at the rates observed. `run_two_stage.py --final-rank-requests` raises it.

## Freshness

The question and the design are from He et al., *Practical Lessons from Predicting Clicks on
Ads at Facebook*, ADKDD 2014, Section 5. They trained a model on one day of data, scored it
on each of the following days, and found normalised entropy got steadily worse as the gap
between training and serving grew. That result is the argument for retraining often. The
same experiment is rerun here with the DeepFM ranker on the eight day Taobao log, in
`src/retrieval/freshness.py` and `scripts/run_freshness.py`. It reuses the ranking feature
spec, the shared trainer, `normalized_entropy` and `fast_group_auc` unchanged.

**Protocol**

- Every model is scored on every impression of day 8, with AUC, NE and GAUC by user.
- Staleness curve. For each training day d from 1 to 7, a DeepFM is trained from scratch on
  a seeded sample of `--train-rows` rows from day d alone (`--window-days 1`). Every model
  gets the same rows count, epochs, batch size and seed, so only recency differs. NE is
  reported relative to the day 7 model at each staleness 8 minus d.
- Update strategy. A base model is trained on days 1 to 4 (`--base-days`). Three strategies
  are then scored on day 8. No update keeps the base model. Warm start fine tunes the base
  model for one pass on each new day in turn (days 5, 6 and 7), on that day's rows only, with
  a new Adam optimiser at the ranker's learning rate. Full retrain trains a new model from
  scratch on days 1 to 7. Warm start's share of the freshness gap is
  (NE no update minus NE warm) over (NE no update minus NE full). Compute is reported as
  training rows processed and training wall clock.
- Rows. For each seed and day a sampler draws a disjoint validation part and a training part
  of the same size for every day. Every experiment in a seed uses the same rows for the same
  day, and a multi day set is the union of its days, so each day is sampled at the same rate.
  Validation rows come from the training days, never from day 8, so no model is selected on
  the day it is scored on.
- Features. All models share one encoding, fitted on days 1 to 7, and the test rows'
  features are frozen at the end of day 7 for every model. What ages is the model, not the
  serving features. This matches a system whose feature store is current and whose model push
  is late.
- Unseen ids. Because the vocabulary spans days 1 to 7, a model trained on day 2 has
  embedding rows for ids that first appear on day 6. Those rows never get a gradient. After
  every training step the embedding and first order rows of every code the model has not
  trained on are set to zero, so an unseen id adds nothing, the same rule the two tower
  retriever uses. A warm started model accumulates the codes it has seen, and codes new on a
  fine tune day start from zero and are trained.
- Seeds. The update comparison runs on seeds 0 and 1 by default, the staleness curve on seed
  0 (`--stale-seeds` adds more). Every row is written with its seed, and a summary row holds
  the mean and standard deviation across seeds.

**What this does not claim**

- It is not a reproduction of He et al.'s numbers. Their models were boosted trees feeding a
  logistic regression, trained on Facebook's traffic. This is DeepFM on eight public days of
  Taobao display ads, one test day, and a staleness range of one to seven days.
- One test day and two seeds give a noisy estimate. A staleness difference smaller than the
  spread across seeds is not evidence of anything.
- The feature scaling constants and the index each id gets come from days 1 to 7 for every
  model. No label reaches a model through them, but a stale model is not fully blind to the
  later days' id sets. The unseen id rule above is what removes their effect on scores.
- Warm start compute is the sum over all its updates, while full retrain is charged for its
  single final run, which is conservative for warm start. The daily schedule figure, where
  full retrain would run on every new day, is counted in rows and not measured in seconds.
  Neither is a FLOP count. Wall clock is from the machine and load stamped on each row.
- Fine tuning uses one learning rate and one pass. Another learning rate, replayed older
  rows or online learning at finer grain than a day could all change the answer and were not
  tried.
