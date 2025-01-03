"""Defines the base evaluator class.

The evaluator class takes a set of krecs and generates a set of metrics and
evaluations, formatted nicely for the user.
"""

from abc import ABC, abstractmethod
from typing import Generic

import krec

from keval.config import ConfigType, HasConfigMixin


class BaseEvaluator(ABC, HasConfigMixin[ConfigType], Generic[ConfigType]):
    """Base evaluator class."""

    @abstractmethod
    def evaluate(self, krecs: list[krec.KRec]) -> None:
        """Evaluate the krecs."""
