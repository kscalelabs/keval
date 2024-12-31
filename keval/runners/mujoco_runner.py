"""Mujoco runner."""

from enum import Enum
from pathlib import Path

import mediapy as media
import mujoco
import mujoco_viewer
import numpy as np
from kinfer import proto as P
from kinfer.inference.python import ONNXModel
from omegaconf import DictConfig, ListConfig, OmegaConf

from keval import metrics
from keval.observation import MujocoFullObservation
from keval.runners.base_runner import Runner

LEFT_CAMERA_ID = 2
RIGHT_CAMERA_ID = 3

# TODO: to be removed after kinfer updates
ATTACHED_METADATA = {
    "sim_dt": 0.01,
    "sim_decimation": 1,
    "tau_factor": 1.0,
}

class LocomotionTasks(Enum):
    STANDING = "STANDING"
    CONSTANT_COMMAND = "CONSTANT_COMMAND"
    RANDOM_COMMAND = "RANDOM_COMMAND"

class MujocoRunner(Runner):
    def __init__(
        self,
        eval_config: DictConfig | ListConfig | OmegaConf,
        model: ONNXModel,
        metrics: metrics.Metrics,
    ) -> None:
        """Initializes the Mujoco runner."""
        self.eval_config = eval_config.eval_envs.locomotion
        self.config = eval_config

        super().__init__(eval_config, model, metrics)

        # Init mujoco env
        mujoco_model_path = f"{eval_config.resources_dir}/{eval_config.embodiment}/robot.xml"
        self.mj_model = mujoco.MjModel.from_xml_path(mujoco_model_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Zero-out dynamics
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.mj_data.qvel = np.zeros_like(self.mj_data.qvel)
        self.mj_data.qacc = np.zeros_like(self.mj_data.qacc)

        self.viewer = mujoco_viewer.MujocoViewer(
            self.mj_model,
            self.mj_data,
            "offscreen",
            height=480 // 2,
            width=640 // 2,
        )
        
        self.record_video = eval_config.eval_envs.locomotion.record_video
        self.camera_frames: list[np.ndarray] = []
        self.model_info = ATTACHED_METADATA

        # Initialize command variables
        self.x_vel_cmd = 0.4
        self.y_vel_cmd = 0.0
        self.yaw_vel_cmd = 0.0

    def _init_locomotion_metrics(
        self,
        task: LocomotionTasks,
    ) -> dict[str, metrics.BaseMetric]:
        """Initialize the metrics for the locomotion suite.

        Returns:
            A dictionary of metrics.
        """
        local_metrics = {
            metrics.MetricsType.EPISODE_DURATION.name + "_" + task.value: metrics.EpisodeDuration(
                name=metrics.MetricsType.EPISODE_DURATION.name + "_" + task.value,
                expected_length=int(self.config.eval_envs.locomotion.sim_duration / self.model_info["sim_dt"]),
            ),
            metrics.MetricsType.TRACKING_ERROR.name + "_" + task.value: metrics.TrackingError(
                name=metrics.MetricsType.TRACKING_ERROR.name + "_" + task.value
            ),
            # metrics_module.MetricsType.POSITION_ERROR.name: metrics_module.PositionError(),
            # metrics_module.MetricsType.CONTACT_FORCES.name: metrics_module.ContactForces(),
        }
        return local_metrics

    def update_locomotion_metrics(
        self,
        local_metrics: dict[str, metrics.BaseMetric],
        task: LocomotionTasks,
    ) -> dict[str, metrics.BaseMetric]:
        """Update the locomotion metrics.

        Args:
            local_metrics: The metrics to update.
            task: The task to update the metrics for.

        Returns:
            The updated metrics.
        """
        local_metrics[metrics.MetricsType.EPISODE_DURATION.name + "_" + task.value].add_step()
        local_metrics[metrics.MetricsType.TRACKING_ERROR.name + "_" + task.value].add_step(
            observed_velocity=[
                self.mj_data.qvel[0],
                self.mj_data.qvel[1],
                self.mj_data.sensor("angular-velocity").data[2],
            ],
            commanded_velocity=[self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd],
        )

        return local_metrics

    def create_observation(self, simulation_time: float) -> P.IO:
        """Create the observation for the model."""
        # Safely get sensor data with None fallback
        try:
            accel_data = self.mj_data.sensor("accelerometer").data.tolist()
        except (AttributeError, KeyError):
            accel_data = None

        try:
            gyro_data = self.mj_data.sensor("angular-velocity").data.tolist()
        except (AttributeError, KeyError):
            gyro_data = None

        try:
            mag_data = self.mj_data.sensor("magnetometer").data.tolist()
        except (AttributeError, KeyError):
            mag_data = None

        # Get camera frames if viewer exists
        camera_left = (self.viewer.read_pixels(camid=LEFT_CAMERA_ID).tobytes() 
                      if self.viewer is not None else None)
        camera_right = (self.viewer.read_pixels(camid=RIGHT_CAMERA_ID).tobytes()
                       if self.viewer is not None else None)

        raw_obs = MujocoFullObservation(
            qpos=self.mj_data.qpos.tolist(),
            qvel=self.mj_data.qvel.tolist(),
            x_vel_cmd=self.x_vel_cmd,
            y_vel_cmd=self.y_vel_cmd,
            yaw_vel_cmd=self.yaw_vel_cmd,
            accelerometer=accel_data,
            gyroscope=gyro_data,
            magnetometer=mag_data,
            simulation_time=simulation_time,
            camera_frame_left=camera_left,
            camera_frame_right=camera_right,
        )

        return raw_obs.to_proto(self.input_schema)

    def task_setup(self, task: LocomotionTasks, simulation_step: int) -> None:
        if task == LocomotionTasks.STANDING:
            self.x_vel_cmd = 0.0
            self.y_vel_cmd = 0.0
            self.yaw_vel_cmd = 0.0
        elif task == LocomotionTasks.CONSTANT_COMMAND:
            self.x_vel_cmd = 0.4
            self.y_vel_cmd = 0.0
            self.yaw_vel_cmd = 0.0
        elif task == LocomotionTasks.RANDOM_COMMAND:
            if simulation_step % 500 == 0:
                self.x_vel_cmd = np.random.uniform(-0.2, 0.5)
                self.y_vel_cmd = np.random.uniform(-0.3, 0.3)
                self.yaw_vel_cmd = np.random.uniform(-0.3, 0.3)

    def run_one_rollout(
        self,
        index: int,
        task: LocomotionTasks,
    ) -> dict[str, metrics.BaseMetric]:
        """Runs one rollout.

        Args:
            index: The index of the rollout.
            task: The task to update the metrics for.

        Returns:
            The metrics for the rollout.
        """
        local_metrics = self._init_locomotion_metrics(task)

        for simulation_step in range(int(self.config.eval_envs.locomotion.sim_duration / self.model_info["sim_dt"])):
            self.task_setup(task, simulation_step)
            if simulation_step % self.model_info["sim_decimation"] == 0:
                observation = self.create_observation(simulation_step * self.model_info["sim_dt"])
                torques_io = self.model(observation)
                torques = self.model._output_serializer.serialize_io(torques_io, as_dict=True)["joint_torques"]

                if self.record_video:
                    img = self.viewer.read_pixels(camid=0)
                    self.camera_frames.append(img)

            # Update the simulation world
            self.mj_data.ctrl = torques
            mujoco.mj_step(self.mj_model, self.mj_data)

            local_metrics = self.update_locomotion_metrics(
                local_metrics,
                task=task,
            )

        if self.record_video:
            video_path = Path(
                self.config.logging.log_dir,
                "locomotion",
                f"mujoco_simulation_{index}.mp4",
            )

            effective_fps = 1 / (self.model_info["sim_dt"] * self.model_info["sim_decimation"])
            media.write_video(video_path, self.camera_frames, fps=effective_fps)

        return local_metrics

    def run(self) -> list[dict[str, metrics.BaseMetric]]:
        """Runs the Mujoco simulations.

        TODO: add parallization working with ONNX session (thread safe)

        Returns:
            The global metrics for the rollouts.
        """
        all_metrics = []
        for task in LocomotionTasks:
            for index in range(self.config.eval_envs.locomotion.eval_runs):
                local_metrics = self.run_one_rollout(index, task)
                all_metrics.append(local_metrics)

        return all_metrics
