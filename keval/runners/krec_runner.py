"""Defines the krec data runner."""

import torch
from omegaconf import OmegaConf

from keval.metrics import Metrics
from keval.runners.base_runner import Runner


class KrecRunner(Runner):
    def __init__(self, eval_config: OmegaConf, model: torch.nn.Module, metrics: Metrics):
        """Initializes the krec data runner.

        Args:
            eval_config: The evaluation configuration.
            model: The model to evaluate.
            metrics: The metrics to use.
        """
        super().__init__(eval_config, model, metrics)
        self.model = model
        self.eval_config = eval_config
        self.metrics = metrics

    def run(self):
        pass
