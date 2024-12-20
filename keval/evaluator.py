"""Global evaluator."""

import logging
from enum import Enum

from kinfer.inference.python import ONNXModel
from omegaconf import DictConfig, ListConfig, OmegaConf

from keval.metrics import Metrics
from keval.runners.krec_runner import KrecRunner
from keval.runners.mani_skill_runner import ManiSkillRunner
from keval.runners.mujoco_runner import MujocoRunner


class RunnerType(Enum):
    KREC = "krec"
    MUJOCO = "mujoco"
    MANI_SKILL = "mani_skill"


class Evaluator:
    def __init__(self, config: OmegaConf | DictConfig | ListConfig, model: ONNXModel, logger: logging.Logger) -> None:
        """Initializes the evaluator.

        Args:
            config: The configuration for the evaluation.
            model: The model to evaluate.
            logger: The logger to use.
        """
        self.config = config
        self.logger = logger
        self.embodiment = config.embodiment
        self.global_metrics = Metrics(config, logger)
        self.runners = {
            RunnerType.MUJOCO: MujocoRunner(config, model, self.global_metrics),
            RunnerType.KREC: KrecRunner(config, model, self.global_metrics),
            RunnerType.MANI_SKILL: ManiSkillRunner(config, model, self.global_metrics),
        }
        self.model = model

    def run_eval(self) -> None:
        """Runs the evaluation."""
        self.logger.info("Running evaluation")
        if self.config.eval_suites.locomotion:
            self.logger.info("Running locomotion evaluation")
            self.global_metrics.compile(self.runners[RunnerType.MUJOCO].run())

        if self.config.eval_suites.krec:
            self.logger.info("Running krec evaluation")
            pass

        if self.config.eval_suites.manipulation_mani_skill:
            self.logger.info("Running manipulation skill evaluation")
            pass

        if self.config.eval_suites.whole_body_control:
            self.logger.info("Running whole body control evaluation")
            pass

        self.logger.info("Evaluation complete")
