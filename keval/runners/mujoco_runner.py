"""Mujoco runner."""

from pathlib import Path

import mediapy as media
import mujoco
import mujoco_viewer
import numpy as np
from kinfer import proto as P
from kinfer.inference.python import ONNXModel
from omegaconf import DictConfig, ListConfig, OmegaConf

from keval import metrics
from keval.observation import JOINT_NAMES, FullObservation, ValueType
from keval.runners.base_runner import Runner

LEFT_CAMERA_ID = 2
RIGHT_CAMERA_ID = 3

# TODO: to be removed after kinfer updates
ATTACHED_METADATA = {
    "sim_dt": 0.01,
    "sim_decimation": 1,
    "tau_factor": 1.0,
}


class MujocoRunner(Runner):
    def __init__(
        self,
        eval_config: DictConfig | ListConfig | OmegaConf,
        model: ONNXModel,
        metrics: metrics.Metrics,
    ) -> None:
        """Initializes the Mujoco runner.

        Args:
            eval_config: The evaluation configuration.
            model: The model to evaluate.
            metrics: The metrics to use.
        """
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

        self.viewer = mujoco_viewer.MujocoViewer(
            self.mj_model,
            self.mj_data,
            "offscreen",
            height=480 // 2,
            width=640 // 2,
        )
        self.record_video = eval_config.eval_envs.locomotion.record_video
        self.camera_frames: list[np.ndarray] = []

        self.x_vel_cmd = 0.4
        self.y_vel_cmd = 0.0
        self.yaw_vel_cmd = 0.0

    def _init_locomotion_metrics(
        self,
    ) -> dict[str, metrics.BaseMetric]:
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
        local_metrics: dict[str, metrics.BaseMetric],
        commanded_velocity: list[float],
    ) -> dict[str, metrics.BaseMetric]:
        """Update the locomotion metrics.

        Args:
            local_metrics: The metrics to update.
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

    def create_observation(self, simulation_time: float) -> P.IO:
        """Create the observation for the model.

        Args:
            simulation_time: The simulation time.

        Returns:
            The observation for the model.
        """
        joint_positions = P.Value(
            value_name=ValueType.JOINT_POSITIONS.value,
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
        )

        joint_velocities = P.Value(
            value_name=ValueType.JOINT_VELOCITIES.value,
            joint_velocities=P.JointVelocitiesValue(
                values=[
                    P.JointVelocityValue(
                        joint_name=name,
                        value=self.mj_data.qvel[-len(JOINT_NAMES) + index],
                        unit=P.JointVelocityUnit.RADIANS_PER_SECOND,
                    )
                    for index, name in enumerate(JOINT_NAMES)
                ]
            ),
        )

        vector_command = P.Value(
            value_name=ValueType.VECTOR_COMMAND.value,
            vector_command=P.VectorCommandValue(
                values=[self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd],
            ),
        )

        imu = P.Value(
            value_name=ValueType.IMU.value,
            imu=P.ImuValue(
                linear_acceleration=P.ImuAccelerometerValue(
                    x=self.mj_data.sensor("accelerometer").data[0],
                    y=self.mj_data.sensor("accelerometer").data[1],
                    z=self.mj_data.sensor("accelerometer").data[2],
                ),
                angular_velocity=P.ImuGyroscopeValue(
                    x=self.mj_data.sensor("angular-velocity").data[0],
                    y=self.mj_data.sensor("angular-velocity").data[1],
                    z=self.mj_data.sensor("angular-velocity").data[2],
                ),
                magnetic_field=P.ImuMagnetometerValue(
                    x=self.mj_data.sensor("magnetometer").data[0],
                    y=self.mj_data.sensor("magnetometer").data[1],
                    z=self.mj_data.sensor("magnetometer").data[2],
                ),
            ),
        )

        seconds = int(simulation_time)
        nanos = int((simulation_time - seconds) * 1e9)
        timestamp = P.Value(
            value_name=ValueType.TIMESTAMP.value,
            timestamp=P.TimestampValue(
                seconds=seconds,
                nanos=nanos,
            ),
        )

        camera_frame_left = P.Value(
            value_name=ValueType.CAMERA_FRAME_LEFT.value,
            camera_frame=P.CameraFrameValue(
                data=self.viewer.read_pixels(camid=LEFT_CAMERA_ID).tobytes(),
            ),
        )

        camera_frame_right = P.Value(
            value_name=ValueType.CAMERA_FRAME_RIGHT.value,
            camera_frame=P.CameraFrameValue(
                data=self.viewer.read_pixels(camid=RIGHT_CAMERA_ID).tobytes(),
            ),
        )

        full_observation = FullObservation(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            vector_command=vector_command,
            imu=imu,
            timestamp=timestamp,
            camera_frame_left=camera_frame_left,
            camera_frame_right=camera_frame_right,
        )

        # Assemble observation space for the model
        inputs = P.IO(values=[])
        for key in self.model.schema_input_keys:
            try:
                inputs.values.append(getattr(full_observation, key))
            except AttributeError:
                raise AttributeError(f"Attribute {key} not found in FullObservation")

        return inputs

    def run_one_rollout(self, index: int) -> dict[str, metrics.BaseMetric]:
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
                observation = self.create_observation(simulation_step * self.model_info["sim_dt"])
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

    def run(self) -> list[dict[str, metrics.BaseMetric]]:
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
