"""Retrieve then rank, for one request, timed stage by stage.

A request names a user, a placement and an hour. The pipeline

1. builds the user's features (profile row, click history and prior click
   count as of the request time) and runs the user tower,
2. searches the FAISS index for the top K ads by inner product,
3. builds one ranking row per retrieved ad and scores them with the ranker,
4. returns the top n by predicted click probability.

The exhaustive baseline skips steps 1 and 2 and scores every corpus ad. The
point of the architecture is the gap between the two, in quality and in cost,
and both are measured with this one class so they share every line of code
except the candidate set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from src.retrieval.features import ClickHistory, Encoded
from src.retrieval.ranking import RankingSpec, candidate_dataset, score_module
from src.retrieval.two_tower import TwoTowerRetriever


@dataclass
class Response:
    ads: np.ndarray            # corpus rows, best first
    scores: np.ndarray         # predicted click probability, aligned with ads
    retrieved: np.ndarray      # the K candidates retrieval handed to the ranker
    stage_ns: Dict[str, int]   # user_embed, search, feature_build, rank, total


class TwoStagePipeline:
    def __init__(
        self,
        tower: TwoTowerRetriever,
        index,                      # src.retrieval.index.Index, or None for exhaustive only
        ranker: torch.nn.Module,
        spec: RankingSpec,
        enc: Encoded,
        hist: ClickHistory,
        tower_history_len: int,
        ranker_uses_history: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.tower = tower.eval()
        self.index = index
        self.ranker = ranker.eval()
        self.spec = spec
        self.enc = enc
        self.hist = hist
        self.tower_history_len = tower_history_len
        self.ranker_uses_history = ranker_uses_history
        self.device = device or torch.device("cpu")

    def _user_state(self, user_row: int, t: int):
        h = self.hist.history(np.array([user_row]), np.array([t]), max(self.tower_history_len,
                                                                       self.spec.history_len))
        count = int(self.hist.counts(np.array([user_row]), np.array([t]))[0])
        return h[0], count

    def user_vector(self, user_row: int, history: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            u = torch.as_tensor([user_row], device=self.device)
            h = torch.as_tensor(history[None, : self.tower_history_len], device=self.device)
            return self.tower.encode_users(u, h).float().cpu().numpy()

    def score_candidates(self, user_row: int, history: np.ndarray, count: int, pid: int, hour: int,
                         cand: np.ndarray, chunk: int = 131072) -> tuple:
        """Build rows and score them. Returns (scores, build_ns, rank_ns)."""
        hrow = history[: self.spec.history_len] if self.ranker_uses_history else None
        out = np.empty(len(cand), dtype=np.float32)
        build_ns = rank_ns = 0
        for s in range(0, len(cand), chunk):
            t0 = time.perf_counter_ns()
            ds = candidate_dataset(self.spec, self.enc, user_row, hrow, count, pid, hour,
                                   cand[s:s + chunk])
            t1 = time.perf_counter_ns()
            out[s:s + chunk] = score_module(self.ranker, ds, self.device, batch_size=chunk)
            t2 = time.perf_counter_ns()
            build_ns += t1 - t0
            rank_ns += t2 - t1
        return out, build_ns, rank_ns

    def request(self, user_row: int, pid: int, hour: int, t: int, k: int, n: int = 10) -> Response:
        t_start = time.perf_counter_ns()
        history, count = self._user_state(user_row, t)
        uvec = self.user_vector(user_row, history)
        t_embed = time.perf_counter_ns()
        _, ids = self.index.search(uvec, k)
        cand = ids[0][ids[0] >= 0].astype(np.int64)
        t_search = time.perf_counter_ns()
        scores, build_ns, rank_ns = self.score_candidates(user_row, history, count, pid, hour, cand)
        order = np.argsort(-scores, kind="stable")[:n]
        t_end = time.perf_counter_ns()
        return Response(
            ads=cand[order], scores=scores[order], retrieved=cand,
            stage_ns={"user_embed": t_embed - t_start, "search": t_search - t_embed,
                      "feature_build": build_ns, "rank": rank_ns, "total": t_end - t_start},
        )

    def exhaustive(self, user_row: int, pid: int, hour: int, t: int, n: int = 10) -> Response:
        t_start = time.perf_counter_ns()
        history, count = self._user_state(user_row, t)
        cand = np.arange(self.enc.n_ads, dtype=np.int64)
        scores, build_ns, rank_ns = self.score_candidates(user_row, history, count, pid, hour, cand)
        order = np.argpartition(-scores, n)[:n] if n < len(scores) else np.arange(len(scores))
        order = order[np.argsort(-scores[order], kind="stable")]
        t_end = time.perf_counter_ns()
        return Response(
            ads=cand[order], scores=scores[order], retrieved=cand,
            stage_ns={"feature_build": build_ns, "rank": rank_ns, "total": t_end - t_start},
        )

    def exhaustive_scores(self, user_row: int, pid: int, hour: int, t: int) -> np.ndarray:
        history, count = self._user_state(user_row, t)
        cand = np.arange(self.enc.n_ads, dtype=np.int64)
        return self.score_candidates(user_row, history, count, pid, hour, cand)[0]
