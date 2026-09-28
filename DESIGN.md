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

(Filled in as each part lands. See the sections below.)
