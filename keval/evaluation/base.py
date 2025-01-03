"""Defines the base evaluation class.

The evaluation class defines the type of evaluation to be performed. This
should take the provided model and environment and generate a set of krecs
by running the model in the environment.
"""

from abc import ABC, abstractmethod
from typing import Generic

from kinfer import proto as K
from kinfer.inference.python import ONNXModel

from keval.config import ConfigType, HasConfigMixin


class BaseEvaluation(ABC, HasConfigMixin[ConfigType], Generic[ConfigType]):
    """Base evaluation class."""

    @abstractmethod
    def evaluate(self, model: ONNXModel, environment: Environment) -> K.IO:
        """Evaluate the model in the environment."""
