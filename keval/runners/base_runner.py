"""Defines the base runner for the model."""

from abc import ABC, abstractmethod

from kinfer.inference.python import ONNXModel
from omegaconf import DictConfig

from keval import metrics
from keval.schema import SchemaConverter


class Runner(ABC):
    def __init__(self, config: DictConfig, model: ONNXModel, metrics: metrics.Metrics) -> None:
        """Initialize the runner.

        Args:
            config: The configuration.
            model: The model to evaluate.
            metrics: The metrics to use.
        """
        self.config = config
        self.model = model
        self.metrics = metrics
        self.input_schema, self.output_schema = SchemaConverter.create_schema(self.config)

    @abstractmethod
    def run(self) -> list[dict[str, metrics.BaseMetric]]:
        """Runs the runner."""
        pass
