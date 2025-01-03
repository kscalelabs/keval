"""Defines the main interfaces for interacting with keval."""

from typing import Any, Type, TypeVar

from keval.config import Config
from keval.environment.base import BaseEnvironment
from keval.evaluation.base import BaseEvaluation
from keval.evaluator.base import BaseEvaluator

ConfigType = TypeVar("ConfigType", bound=Config)


def run(
    environment_cls: Type[BaseEnvironment[ConfigType]],
    evaluation_cls: Type[BaseEvaluation[ConfigType]],
    evaluator_cls: Type[BaseEvaluator[ConfigType]],
    **kwargs: Any,
) -> None:
    """Run the evaluation."""
    environment_cfg = environment_cls.get_config(kwargs)
    evaluation_cfg = evaluation_cls.get_config(environment_cfg, kwargs)
    evaluator_cfg = evaluator_cls.get_config(evaluation_cfg, kwargs)

    environment = environment_cls(environment_cfg)
    evaluation = evaluation_cls(evaluation_cfg)
    evaluator = evaluator_cls(evaluator_cfg)

    krecs = evaluation.evaluate(environment)
    evaluator.evaluate(krecs)
