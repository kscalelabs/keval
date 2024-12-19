"""Defines the evaluator."""

import logging
from enum import Enum

from kinfer.inference.python import ONNXModel
from omegaconf import OmegaConf

from keval.metrics import Metrics
from keval.runners.krec_runner import KrecRunner
from keval.runners.mani_skill_runner import ManiSkillRunner
from keval.runners.mujoco_runner import MujocoRunner


class RunnerType(Enum):
    KREC = "krec"
    MUJOCO = "mujoco"
    MANI_SKILL = "mani_skill"


class Evaluator:
    def __init__(self, config: OmegaConf, model: ONNXModel, logger: logging.Logger):
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

    def adapt_model(self) -> None:
        """Adapt the schema of the model to the laste version of:"""
        self.logger.warning(
            """Adapt the schema of the model to the laste version of:
            https://github.com/kscalelabs/kinfer/blob/master/kinfer/protos/kinfer.proto
            """
        )

        # TODO: Implement the model adaptation
        pass

    def valid_locomotion(self: dict) -> None:
        """Validates the model metadata for locomotion."""
        assert hasattr(self.model_metadata, "num_actions")
        assert hasattr(self.model_metadata, "num_observations")
        assert hasattr(self.model_metadata, "robot_effort")
        assert hasattr(self.model_metadata, "robot_stiffness")
        assert hasattr(self.model_metadata, "robot_damping")
        assert hasattr(self.model_metadata, "sim_dt")
        assert hasattr(self.model_metadata, "sim_decimation")
        assert hasattr(self.model_metadata, "tau_factor")
