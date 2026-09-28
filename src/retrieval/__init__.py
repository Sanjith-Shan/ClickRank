"""Candidate retrieval in front of the ClickRank rankers.

The pieces, in the order a request meets them:

data        loads an id bearing ads dataset (Taobao, Avazu, or the synthetic
            sample) into one canonical shape, with the time split.
features    turns raw ids into dense indices fitted on the training days only,
            and builds each user's click history as of a cutoff.
two_tower   the user tower and the ad tower, trained with in batch sampled
            softmax and the sampling bias correction.
index       FAISS indexes over the ad tower's embeddings, exact and approximate.
metrics     recall of an approximate index against exact search, and hit rate.
ranking     the feature spec that lets the existing DeepFM and DCN code train
            on the same dataset and score retrieved candidates.
pipeline    retrieve then rank for one request, timed stage by stage.
"""
