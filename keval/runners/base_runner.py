"""Defines the base runner for the model.

TODO:
    1. Add offset mapping
    2. Add joint mapping
"""

from abc import ABC, abstractmethod

from kinfer.inference.python import ONNXModel
from omegaconf import DictConfig, ListConfig, OmegaConf

from keval import metrics


class Runner(ABC):
    @abstractmethod
    def __init__(
        self,
        eval_config: DictConfig | ListConfig | OmegaConf,
        model: ONNXModel,
        metrics: metrics.Metrics,
    ) -> None:
        """Initializes the runner.

        Args:
            eval_config: The evaluation configuration.
            model: The model to evaluate.
            metrics: The metrics to use.
        """
        pass

    @abstractmethod
    def run(self) -> list[dict[str, metrics.BaseMetric]]:
        """Runs the runner."""
        pass
