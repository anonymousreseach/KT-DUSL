from __future__ import annotations
from typing import Dict, Tuple, List, Optional

import numpy as np

from .base import KTModel


class BKT(KTModel):
    """
    BKT 2-state per concept:
      p: P(M=1) before item
      guess/slip: emission params
      learn: transition after item
    """
    def __init__(self, p_init: float = 0.2, p_learn: float = 0.1, p_guess: float = 0.2, p_slip: float = 0.1):
        self.p_init = float(p_init); self.p_learn=float(p_learn)
        self.p_guess=float(p_guess); self.p_slip=float(p_slip)
        self._state: Dict[Tuple[int,int], float] = {}

    def reset_state(self):
        self._state.clear()

    def _p(self, user: int, concept: int) -> float:
        return self._state.get((user, concept), self.p_init)

    def predict_concept_proba(self, user: int, concept: int) -> float:
        p = self._p(user, concept)
        return p*(1 - self.p_slip) + (1 - p)*self.p_guess

    def update(self, user: int, concepts: List[int], correct: int, weights: Optional[List[float]] = None):
        # standard BKT update per concept (weights unused)
        for c in concepts:
            p = self._p(user, c)
            p_correct = p*(1 - self.p_slip) + (1 - p)*self.p_guess
            if int(correct) == 1:
                numer = p*(1 - self.p_slip); denom = max(1e-12, p_correct)
                p_post = numer/denom
            else:
                numer = p*self.p_slip; denom = max(1e-12, 1 - p_correct)
                p_post = numer/denom
            p_next = p_post + (1 - p_post)*self.p_learn
            self._state[(user, c)] = float(np.clip(p_next, 1e-9, 1-1e-9))
