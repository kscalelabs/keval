"""Metrics for evaluating the performance of the model."""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

from keval.observation import JOINT_NAMES


class MetricsType(Enum):
    EPISODE_DURATION = "EPISODE_DURATION"
    EPISODE_REWARD = "EPISODE_REWARD"
    TRACKING_ERROR = "TRACKING_ERROR"
    POSITION_ERROR = "POSITION_ERROR"
    CONTACT_FORCES = "CONTACT_FORCES"


class BaseMetric(ABC):
    def __init__(self, name: str | MetricsType) -> None:
        """Basic metric class.

        Args:
            name: The name of the metric.
        """
        self.name = name

    @abstractmethod
    def add_step(self, *args: Any, **kwargs: Any) -> None:
        """Process data to compute the metric.

        Args:
            *args: The arguments to pass to the metric.
            **kwargs: The keyword arguments to pass to the metric.
        """
        raise NotImplementedError("Each metric must implement the add_step method.")

    @abstractmethod
    def compile(self, *args: Any, **kwargs: Any) -> np.ndarray | float | int | None:
        """Compile the metric.

        Args:
            *args: The arguments to pass to the metric.
            **kwargs: The keyword arguments to pass to the metric.

        Returns:
            The compiled metric.
        """
        raise NotImplementedError("Each metric must implement the compile method.")


class TrackingError(BaseMetric):
    def __init__(self) -> None:
        """Initialize the tracking error metric.

        Args:
            name: The name of the metric.
        """
        super().__init__(name=MetricsType.TRACKING_ERROR)
        self.commanded_velocity: list[np.ndarray] = []
        self.observed_velocity: list[np.ndarray] = []

    def add_step(self, observed_velocity: np.ndarray, commanded_velocity: np.ndarray) -> None:
        """Add a step to the tracking error metric.

        Args:
            observed_velocity: The observed velocity.
            commanded_velocity: The commanded velocity.
        """
        self.observed_velocity.append(observed_velocity)
        self.commanded_velocity.append(commanded_velocity)

    def compile(self) -> np.ndarray:
        """Average tracking error over all steps for each velocity component."""
        observed = np.array(self.observed_velocity)
        commanded = np.array(self.commanded_velocity)

        error_per_component = np.mean(np.abs(observed - commanded), axis=0)
        return error_per_component

    def save_plot(self, index: int, save_dir: Path) -> None:
        """Plot tracking error for x, y, and angular velocity components.

        Args:
            index: The index of the rollout.
            save_dir: The directory to save the plot.
        """
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

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
    def __init__(self, expected_length: int) -> None:
        """Initialize the episode duration metric.

        Args:
            expected_length: The expected length of the episode.
        """
        super().__init__(name=MetricsType.EPISODE_DURATION)
        self.expected_length = expected_length
        self.step_count = 0

    def add_step(self) -> None:
        """Add a step to the episode duration metric."""
        self.step_count += 1

    def compile(self) -> float:
        """Average episode duration over all steps."""
        average_duration = self.step_count / self.expected_length
        return average_duration


class PositionError(BaseMetric):
    def __init__(self) -> None:
        """Initialize the position error metric.

        Args:
            name: The name of the metric.
        """
        super().__init__(name=MetricsType.POSITION_ERROR)
        self.predicted_position: list[np.ndarray] = []
        self.observed_position: list[np.ndarray] = []
        self.num_joints = len(JOINT_NAMES)

    def add_step(self, predicted_position: np.ndarray, observed_position: np.ndarray) -> None:
        """Add a step to the position error metric.

        Args:
            predicted_position: The predicted position.
            observed_position: The observed position.
        """
        self.predicted_position.append(predicted_position)
        self.observed_position.append(observed_position)

    def compile(self) -> None:
        """Average position error over all steps."""
        average_error = np.mean(
            np.asarray(self.observed_position)[:, :, : self.num_joints]
            - np.asarray(self.predicted_position)[:, : self.num_joints]
        )

        return average_error

    def save_plot(self, index: int, save_dir: Path) -> None:
        """Save the position error plot.

        Args:
            index: The index of the rollout.
            save_dir: The directory to save the plot.
        """
        # Create a grid of subplots based on number of joints
        rows = int(np.ceil(np.sqrt(self.num_joints)))
        cols = int(np.ceil(self.num_joints / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten()  # Flatten to make indexing easier

        # Convert data to numpy arrays for easier handling
        observed = np.asarray(self.observed_position)[:, 0, : self.num_joints]
        predicted = np.asarray(self.predicted_position)[:, : self.num_joints]
        time_steps = np.arange(len(observed))
        # Create a subplot for each joint
        for joint_idx in range(self.num_joints):
            ax = axes[joint_idx]
            ax.plot(time_steps, predicted[:, joint_idx], label="Predicted", linestyle="-")
            ax.plot(time_steps, observed[:, joint_idx], label="Observed", linestyle="--")
            ax.set_title(f"Joint {joint_idx + 1}")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Position")
            ax.legend()

        # Remove any empty subplots
        for idx in range(self.num_joints, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(save_dir / f"position_error_{index}.png")
        plt.close()


class ContactForces(BaseMetric):
    def __init__(self) -> None:
        """Initialize the contact forces metric.

        Args:
            name: The name of the metric.
        """
        super().__init__(name=MetricsType.CONTACT_FORCES)
        self.contact_forces: list[list[float]] = []

    def add_step(self, contact_forces: list[float]) -> None:
        """Add a step to the contact forces metric.

        Args:
            contact_forces: The contact forces to add.
        """
        self.contact_forces.append(contact_forces)

    def compile(self) -> None:
        """Compile the contact forces metric."""
        pass

    def save_plot(self, index: int, save_dir: Path) -> None:
        """Save the contact forces plot."""
        pass


class Metrics:
    def __init__(self, config: DictConfig | ListConfig | OmegaConf, logger: logging.Logger) -> None:
        """Initialize the metrics class.

        Args:
            config: The configuration.
            logger: The logger.
        """
        self.config = config
        self.logger = logger

    def compile(self, metrics: list[dict[str, BaseMetric]] | list[list[BaseMetric]]) -> None:
        """Compile the metrics.

        Args:
            metrics: The metrics to compile.
        """
        if self.config.eval_suites.locomotion:
            assert self.config.eval_envs.locomotion.eval_runs == len(
                metrics
            ), "Number of metrics does not match number of runs"
            save_dir = Path(self.config.logging.log_dir, "locomotion")
        elif self.config.eval_suites.krec:
            save_dir = Path(self.config.logging.log_dir, "krec")
        else:
            raise ValueError("No evaluation suite selected")
        metrics = [metrics]
        aggregated_metrics: dict[str, list[float | np.ndarray]] = {metric_name: [] for metric_name in metrics[0].keys()}

        # Collect all metric values across runs
        for index, metrics_run in enumerate(metrics):
            for metric_name, metric in metrics_run.items():
                value = metric.compile()
                if value is not None:
                    aggregated_metrics[metric_name].append(value)
                if hasattr(metric, "save_plot"):
                    metric.save_plot(index, save_dir)

        averaged_metrics: dict[str, float | np.ndarray] = {}
        for metric_name, values in aggregated_metrics.items():
            if values and isinstance(values[0], np.ndarray):
                averaged_metrics[metric_name] = np.mean(np.asarray(values), axis=0).tolist()
            else:
                averaged_metrics[metric_name] = float(np.mean(np.asarray(values)))

        # Save to YAML file
        with open(save_dir / "averaged_metrics.yaml", "w") as f:
            yaml.dump(averaged_metrics, f)

        self.logger.info("Averaged metrics: %s", averaged_metrics)

    def plot(self) -> None:
        pass
