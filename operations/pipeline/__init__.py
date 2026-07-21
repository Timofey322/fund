"""Crypto fusion pipeline agents."""

from operations.pipeline.agents.base import Agent
from operations.pipeline.agents.data import DataAgent
from operations.pipeline.agents.fusion import FusionAgent
from operations.pipeline.agents.hmm import HmmAgent
from operations.pipeline.agents.monte_carlo import MonteCarloAgent
from operations.pipeline.agents.plot import PlotAgent
from operations.checkpoint import PipelineContext

DEFAULT_PIPELINE: list[Agent] = [
    DataAgent(),
    HmmAgent(),
    FusionAgent(),
    PlotAgent(),
    MonteCarloAgent(),
]

__all__ = [
    "Agent",
    "PipelineContext",
    "DataAgent",
    "HmmAgent",
    "FusionAgent",
    "PlotAgent",
    "MonteCarloAgent",
    "DEFAULT_PIPELINE",
]
