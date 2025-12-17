from __future__ import annotations
from typing import Dict, Tuple, List, Optional

import numpy as np

from .base import KTModel
from .opinions import prior_alpha_beta


class KTUSL(KTModel):
    """
    Subjective Logic KT (Beta-Bernoulli) per (user, concept).
      - prior: alpha0 = a*W, beta0=(1-a)*W
      - predict concept: p = alpha/(alpha+beta)
      - update: alpha += w if correct else beta += w
    """
    def __init__(self, a: float = 0.5, W: float = 2.0):
        self.a = float(a); self.W = float(W)
        self._state: Dict[Tuple[int,int], Tuple[float,float]] = {}

    def reset_state(self):
        self._state.clear()

    def _prior(self) -> Tuple[float, float]:
        return prior_alpha_beta(self.a, self.W)

    def predict_concept_proba(self, user: int, concept: int) -> float:
        alpha, beta = self._state.get((user, concept), self._prior())
        s = alpha + beta
        return self.a if s <= 0 else float(alpha / s)

    def update(self, user: int, concepts: List[int], correct: int, weights: Optional[List[float]] = None):
        if not concepts: 
            return
        if weights is None:
            weights = [1.0/len(concepts)] * len(concepts)
        for c, w in zip(concepts, weights):
            alpha, beta = self._state.get((user, c), self._prior())
            if int(correct) == 1:
                alpha += float(w)
            else:
                beta  += float(w)
            self._state[(user, c)] = (alpha, beta)
