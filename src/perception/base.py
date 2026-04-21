"""Abstract perception backend: same `detect` / `visualize` contract for all modes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class PerceptionDetector(ABC):
    """
    Session 25 contract: loop + states only depend on detect() return shape
    and ArUco-compatible `corners` entries (see Week 6.0).
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Any]:
        """Return (detected, ids, corners)."""

    @abstractmethod
    def visualize(
        self,
        frame: np.ndarray,
        corners: Any,
        ids: Optional[np.ndarray],
    ) -> np.ndarray:
        """Draw targets on a copy of `frame` and return it."""
