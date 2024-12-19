"""Mujoco runner"""

import math
import os
from copy import deepcopy
from pathlib import Path

import mediapy as media
import mujoco
import mujoco_viewer
import numpy as np
import torch
from kinfer import proto as P
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from keval import metrics
from keval.observation import ATTACHED_METADATA, JOINT_NAMES, input_schema
from keval.runners.base_runner import Runner


class MujocoRunner(Runner):
    def __init__(
        self,
        eval_config: OmegaConf,
        model: torch.nn.Module,
        metrics: metrics.Metrics,
    ):
        """Initializes the Mujoco runner.

        Args:
            eval_config: The evaluation configuration.
            model: The model to evaluate.
            metrics: The metrics to use.
        """
        super().__init__(eval_config, model, metrics)
        self.eval_config = eval_config.eval_envs.locomotion
        self.config = eval_config

        self.model_info = ATTACHED_METADATA
        self.model = model

        Path(self.config.logging.log_dir, "locomotion").mkdir(parents=True, exist_ok=True)

        # Init mujoco env
        mujoco_model_path = f"{eval_config.resources_dir}/{eval_config.embodiment}/robot.xml"
        self.mj_model = mujoco.MjModel.from_xml_path(mujoco_model_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Zero-out dynamics
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.mj_data.qvel = np.zeros_like(self.mj_data.qvel)
        self.mj_data.qacc = np.zeros_like(self.mj_data.qacc)

        self.viewer = mujoco_viewer.MujocoViewer(self.mj_model, self.mj_data, "offscreen")
        self.record_video = eval_config.eval_envs.locomotion.record_video
        self.camera_frames = []

        self.x_vel_cmd = 0.4
        self.y_vel_cmd = 0.0
        self.yaw_vel_cmd = 0.0

    def get_joint_offsets(self) -> dict[tuple[float, float]]:
        pass

    def _init_locomotion_metrics(
        self,
    ) -> dict[metrics.MetricsType, metrics.BaseMetric]:
        """Initialize the metrics for the locomotion suite.

        Returns:
            A dictionary of metrics.
        """
        local_metrics = {
            metrics.MetricsType.EPISODE_DURATION.name: metrics.EpisodeDuration(
                int(self.eval_config.sim_duration / self.model_info["sim_dt"])
            ),
            metrics.MetricsType.TRACKING_ERROR.name: metrics.TrackingError(),
            # metrics.MetricsType.POSITION_ERROR.name: metrics.PositionError(),
            # metrics.MetricsType.CONTACT_FORCES.name: metrics.ContactForces(),
        }
        return local_metrics

    def update_locomotion_metrics(
        self,
        local_metrics: dict[metrics.MetricsType, metrics.BaseMetric],
        commanded_velocity: list[float],
    ) -> None:
        """Update the locomotion metrics.

        Args:
            metrics: The metrics to update.
            commanded_velocity: The commanded velocity.

        Returns:
            The updated metrics.
        """
        local_metrics[metrics.MetricsType.EPISODE_DURATION.name].add_step()
        local_metrics[metrics.MetricsType.TRACKING_ERROR.name].add_step(
            observed_velocity=[
                self.mj_data.qvel[0],
                self.mj_data.qvel[1],
                self.mj_data.sensor("angular-velocity").data[2],
            ],
            commanded_velocity=commanded_velocity,
        )

        return local_metrics

    def create_observation(self) -> None:
        """Create the observation for the model.

        Returns:
            The observation for the model.
        """
        inputs = P.IO(
            values=[
                P.Value(
                    value_name="joint_positions",
                    joint_positions=P.JointPositionsValue(
                        values=[
                            P.JointPositionValue(
                                joint_name=name,
                                value=self.mj_data.qpos[-len(JOINT_NAMES) + index],
                                unit=P.JointPositionUnit.RADIANS,
                            )
                            for index, name in enumerate(JOINT_NAMES)
                        ]
                    ),
                ),
                P.Value(
                    value_name="joint_velocities",
                    state_tensor=P.StateTensorValue(
                        data=self.mj_data.qvel[-len(JOINT_NAMES) :].astype(np.float32).tobytes()
                    ),
                ),
            ],
        )

        return inputs

    def run_one_rollout(self, index) -> None:
        """Runs one rollout.

        Args:
            index: The index of the rollout.

        Returns:
            The metrics for the rollout.
        """
        local_metrics = self._init_locomotion_metrics()
        commanded_velocity = [self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd]

        for simulation_step in range(int(self.eval_config.sim_duration / self.model_info["sim_dt"])):
            if simulation_step % self.model_info["sim_decimation"] == 0:
                observation = self.create_observation()
                torques_io = self.model(observation)
                torques = self.model._output_serializer.serialize_io(torques_io, as_dict=True)["joint_torques"]

                if self.record_video:
                    img = self.viewer.read_pixels(camid=0)
                    self.camera_frames.append(img)

            # Update the simulation world
            self.mj_data.ctrl = torques
            mujoco.mj_step(self.mj_model, self.mj_data)

            local_metrics = self.update_locomotion_metrics(local_metrics, commanded_velocity)

        if self.record_video:
            video_path = Path(
                self.config.logging.log_dir,
                "locomotion",
                f"mujoco_simulation_{index}.mp4",
            )

            effective_fps = 1 / (self.model_info["sim_dt"] * self.model_info["sim_decimation"])
            media.write_video(video_path, self.camera_frames, fps=effective_fps)

        return local_metrics

    def run(self) -> None:
        """Runs the Mujoco simulations.

        TODO: add parallization working with ONNX session (thread safe)

        Returns:
            The global metrics for the rollouts.
        """
        all_metrics = []
        for index in range(self.eval_config.eval_runs):
            local_metrics = self.run_one_rollout(index)
            all_metrics.append(local_metrics)

        return all_metrics
