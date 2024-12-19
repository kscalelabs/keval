"""Metrics for evaluating the performance of the model."""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from omegaconf import OmegaConf


class MetricsType(Enum):
    EPISODE_DURATION = "EPISODE_DURATION"
    EPISODE_REWARD = "EPISODE_REWARD"
    TRACKING_ERROR = "TRACKING_ERROR"
    POSITION_ERROR = "POSITION_ERROR"
    CONTACT_FORCES = "CONTACT_FORCES"


# Base Metric Interface
class BaseMetric(ABC):
    def __init__(self, name: str):
        self.name = name
        self.value = None

    @abstractmethod
    def add_step(self, data: Any):
        """Process data to compute the metric."""
        raise NotImplementedError("Each metric must implement the run method.")

    @abstractmethod
    def compile(self):
        """Compile the metric."""
        raise NotImplementedError("Each metric must implement the compile method.")


class TrackingError(BaseMetric):
    def __init__(self):
        super().__init__(name=MetricsType.TRACKING_ERROR)
        self.commanded_velocity = []
        self.observed_velocity = []

    def add_step(self, observed_velocity: list[float], commanded_velocity: list[float]):
        self.observed_velocity.append(observed_velocity)
        self.commanded_velocity.append(commanded_velocity)

    def compile(self):
        """Average tracking error over all steps for each velocity component."""
        observed = np.array(self.observed_velocity)
        commanded = np.array(self.commanded_velocity)

        error_per_component = np.mean(np.abs(observed - commanded), axis=0)
        return error_per_component

    def save_plot(self, index: int, save_dir: Path):
        """Plot tracking error for x, y, and angular velocity components."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Convert lists to numpy arrays for easier indexing
        cmd_vel = np.array(self.commanded_velocity)
        obs_vel = np.array(self.observed_velocity)
        time_steps = np.arange(len(cmd_vel))

        # Plot x velocity
        ax1.plot(time_steps, cmd_vel[:, 0], label="Commanded", linestyle="-")
        ax1.plot(time_steps, obs_vel[:, 0], label="Observed", linestyle="--")
        ax1.set_title("X Velocity Tracking")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Velocity")
        ax1.legend()

        # Plot y velocity
        ax2.plot(time_steps, cmd_vel[:, 1], label="Commanded", linestyle="-")
        ax2.plot(time_steps, obs_vel[:, 1], label="Observed", linestyle="--")
        ax2.set_title("Y Velocity Tracking")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Velocity")
        ax2.legend()

        # Plot angular velocity
        ax3.plot(time_steps, cmd_vel[:, 2], label="Commanded", linestyle="-")
        ax3.plot(time_steps, obs_vel[:, 2], label="Observed", linestyle="--")
        ax3.set_title("Angular Velocity Tracking")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Velocity")
        ax3.legend()

        plt.tight_layout()
        plt.savefig(save_dir / f"tracking_error_{index}.png")
        plt.close()


class EpisodeDuration(BaseMetric):
    def __init__(self, expected_length: int):
        super().__init__(name=MetricsType.EPISODE_DURATION)
        self.expected_length = expected_length
        self.step_count = 0

    def add_step(self):
        self.step_count += 1

    def compile(self):
        """Average episode duration over all steps."""
        average_duration = self.step_count / self.expected_length
        return average_duration


class PositionError(BaseMetric):
    def __init__(self):
        super().__init__(name=MetricsType.POSITION_ERROR)
        self.commanded_position = []
        self.observed_position = []

    def add_step(self, commanded_position: np.ndarray, observed_position: np.ndarray):
        self.commanded_position.append(commanded_position)
        self.observed_position.append(observed_position)

    def compile(self):
        """Average position error over all steps."""
        average_error = np.mean(self.observed_position - self.commanded_position)

        return average_error

    def save_plot(self, index: int, save_dir: Path):
        plt.plot(self.commanded_position, self.observed_position, "o")
        plt.savefig(save_dir / f"position_error_{index}.png")
        plt.close()


class ContactForces(BaseMetric):
    def __init__(self):
        super().__init__(name=MetricsType.CONTACT_FORCES)
        self.contact_forces = []

    def add_step(self, contact_forces: list[float]):
        self.contact_forces.append(contact_forces)

    def compile(self):
        pass

    def save_plot(self, index: int, save_dir: Path):
        pass


class Metrics:
    def __init__(self, config: OmegaConf, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def compile(self, metrics: list[list[BaseMetric]]):
        assert self.config.eval_envs.locomotion.eval_runs == len(
            metrics
        ), "Number of metrics does not match number of runs"
        save_dir = Path(self.config.logging.log_dir, "locomotion")

        aggregated_metrics = {metric_name: [] for metric_name in metrics[0].keys()}

        # Collect all metric values across runs
        for index, metrics_run in enumerate(metrics):
            for metric_name, metric in metrics_run.items():
                value = metric.compile()
                if value is not None:  # Only add non-None values
                    aggregated_metrics[metric_name].append(value)
                if hasattr(metric, "save_plot"):
                    metric.save_plot(index, save_dir)

        averaged_metrics = {}
        for metric_name, values in aggregated_metrics.items():
            if isinstance(values[0], np.ndarray):
                averaged_metrics[metric_name] = np.mean(values, axis=0).tolist()
            else:
                averaged_metrics[metric_name] = float(np.mean(values))

        # Save to YAML file
        with open(save_dir / "averaged_metrics.yaml", "w") as f:
            yaml.dump(averaged_metrics, f)

        self.logger.info(f"Averaged metrics: {averaged_metrics}")

    def plot(self):
        pass
