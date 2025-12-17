from abc import ABC, abstractmethod
from typing import List, Optional

class KTModel(ABC):
    @abstractmethod
    def reset_state(self): ...

    @abstractmethod
    def predict_concept_proba(self, user: int, concept: int) -> float:
        """Return p(correct) related to one concept for this user (before seeing the current answer)."""
        ...

    @abstractmethod
    def update(self, user: int, concepts: List[int], correct: int, weights: Optional[List[float]] = None):
        """Update internal state with the observed label for each concept."""
        ...
