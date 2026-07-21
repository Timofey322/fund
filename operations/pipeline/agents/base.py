"""Base agent protocol."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from operations.checkpoint import PipelineCheckpoint, PipelineContext


class Agent(ABC):
    name: str

    @abstractmethod
    def run(self, ctx: PipelineContext, ckpt: PipelineCheckpoint | None = None) -> PipelineContext:
        """Execute role; optionally read/write checkpoint."""
