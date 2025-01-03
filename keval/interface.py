"""Defines the main interfaces for interacting with keval."""

from typing import Any, Type, TypeVar

from keval.config import Config
from keval.environment.base import BaseEnvironment
from keval.evaluation.base import BaseEvaluation
from keval.evaluator.base import BaseEvaluator

ConfigType = TypeVar("ConfigType", bound=Config)


def run(
    environment: Type[BaseEnvironment[ConfigType]],
    evaluation: Type[BaseEvaluation[ConfigType]],
    evaluator: Type[BaseEvaluator[ConfigType]],
    **kwargs: Any,
) -> None:
    """Run the evaluation."""
    raise NotImplementedError("Not implemented yet")
