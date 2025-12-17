from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import math

from .base import KTModel


class PFA(KTModel):
    def __init__(
        self,
        beta0: float = -1.0,      # global bias
        beta_win: float = 0.7,    # weight of past successes (>= 0 generally)
        beta_fail: float = -0.3,  # weight of past failures (<= 0 generally)
        decay_gamma: float = 1.0, # 1.0 = no forgetting; otherwise 0.97–0.995 typically
        clip: float = 1e-6,       # numeric clamp to avoid strict 0/1
    ):
        self.beta0 = float(beta0)
        self.beta_win = float(beta_win)
        self.beta_fail = float(beta_fail)
        self.decay_gamma = float(decay_gamma)
        self.clip = float(clip)

        # State: (user, concept) -> (wins, fails)
        self._state: Dict[Tuple[int, int], Tuple[float, float]] = {}

    def reset_state(self):
        self._state.clear()

    def predict_concept_proba(self, user: int, concept: int) -> float:
        wins, fails = self._state.get((user, concept), (0.0, 0.0))
        z = self.beta0 + self.beta_win * wins + self.beta_fail * fails
        p = 1.0 / (1.0 + math.exp(-z))
        return float(min(max(p, self.clip), 1.0 - self.clip))

    def update(self, user: int, concepts: List[int], correct: int, weights: Optional[List[float]] = None):
        if not concepts:
            return
        y = 1 if int(correct) == 1 else 0
        if weights is None:
            weights = [1.0 / len(concepts)] * len(concepts)

        for c, w in zip(concepts, weights):
            wins, fails = self._state.get((user, c), (0.0, 0.0))

            if 0.0 < self.decay_gamma < 1.0:
                wins *= self.decay_gamma
                fails *= self.decay_gamma

            wins += w * y
            fails += w * (1 - y)

            self._state[(user, c)] = (float(wins), float(fails))
