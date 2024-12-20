"""Defines the ManiSkill runner."""

from kinfer.inference.python import ONNXModel
from omegaconf import DictConfig, ListConfig, OmegaConf

from keval import metrics
from keval.runners.base_runner import Runner


class ManiSkillRunner(Runner):
    def __init__(
        self,
        eval_config: DictConfig | ListConfig | OmegaConf,
        model: ONNXModel,
        metrics: metrics.Metrics,
    ) -> None:
        """Initializes the ManiSkill runner.

        Args:
            eval_config: The evaluation configuration.
            model: The model to evaluate.
            metrics: The metrics to use.
        """
        self.model = model
        self.eval_config = eval_config
        self.metrics = metrics

    def run(self) -> list[dict[str, metrics.BaseMetric]]:
        """Runs the ManiSkill simulations.

        Returns:
            The global metrics for the rollouts.
        """
        return []
