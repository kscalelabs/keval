"""Defines the base runner for the model.

TODO:
    1. Add offset mapping
    2. Add joint mapping
"""

from abc import ABC, abstractmethod

import torch
from omegaconf import OmegaConf

from keval.metrics import Metrics


class Runner(ABC):
    @abstractmethod
    def __init__(self, eval_config: OmegaConf, model: torch.nn.Module, metrics: Metrics):
        """Initializes the runner.

        Args:
            eval_config: The evaluation configuration.
            model: The model to evaluate.
            metrics: The metrics to use.
        """
        pass

    def joint_offsets(self) -> dict[tuple[float, float]]:
        return self.get_joint_offsets()
