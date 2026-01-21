"""
pilotJSP: Imitation learning for job-shop scheduling with pilot heuristics.

This package implements imitation learning techniques for job-shop scheduling
problems, including DAgger algorithm, ordinal regression, and pilot rollout
heuristics.
"""

__version__ = "0.1.0"
__author__ = "HI-IDN"

from .jsp_instance import JSPInstance
from .features import FeatureExtractor
from .expert import GurobiExpert
from .preferences import PreferenceBuilder
from .dagger import DAggerAlgorithm
from .model import OrdinalRegressionModel
from .pilot import PilotHeuristic

__all__ = [
    "JSPInstance",
    "FeatureExtractor",
    "GurobiExpert",
    "PreferenceBuilder",
    "DAggerAlgorithm",
    "OrdinalRegressionModel",
    "PilotHeuristic",
]
