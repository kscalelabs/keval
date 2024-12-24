"""Defines the krec data runner."""

from kinfer import proto as P
from kinfer.inference.python import ONNXModel
from omegaconf import DictConfig, ListConfig, OmegaConf

from keval import metrics
from keval.observation import JointPositionsValue, ValueType
from keval.runners.base_runner import Runner
from keval.runners.krec_dataloader import KRecDataset, get_krec_dataloader
from keval.observation import JOINT_NAMES


class KrecRunner(Runner):
    def __init__(
        self,
        eval_config: DictConfig | ListConfig | OmegaConf,
        model: ONNXModel,
        metrics: metrics.Metrics,
    ) -> None:
        """Initializes the krec data runner.

        Args:
            eval_config: The evaluation configuration.
            model: The model to evaluate.
            metrics: The metrics to use.
        """
        self.dataset = KRecDataset(eval_config.eval_envs.krec.dataset_path)
        self.dataloader = get_krec_dataloader(eval_config.eval_envs.krec.dataset_path, batch_size=1)
        self.model = model
        self.eval_config = eval_config
        self.metrics = metrics

    def _init_krec_metrics(
        self,
    ) -> dict[str, metrics.BaseMetric]:
        """Initialize the metrics for the locomotion suite.

        Returns:
            A dictionary of metrics.
        """
        local_metrics = {
            metrics.MetricsType.POSITION_ERROR.name: metrics.PositionError(),
        }
        return local_metrics

    def create_observation(self, batch) -> P.IO:
        """Create the observation for the model.

        Args:
            simulation_time: The simulation time.

        Returns:
            The observation for the model.
        """
        schema_batch = []
        for datapoint in batch["joint_pos"]:
            joint_positions = P.Value(
                value_name=ValueType.JOINT_POSITIONS.value,
                joint_positions=P.JointPositionsValue(
                    values=[
                        P.JointPositionValue(
                            joint_name=name,
                            value=datapoint[index],
                            unit=P.JointPositionUnit.RADIANS,
                        )
                        for index, name in enumerate(JOINT_NAMES)
                    ]
                ),
            )

            # joint_velocities = P.Value(
            #     value_name=ValueType.JOINT_VELOCITIES.value,
            #     joint_velocities=P.JointVelocitiesValue(
            #         values=[
            #             P.JointVelocityValue(
            #                 joint_name=name,
            #                 value=datapoint["joint_vel"],
            #                 unit=P.JointVelocityUnit.RADIANS_PER_SECOND,
            #             )
            #             for index, name in enumerate(JOINT_NAMES)
            #         ]
            #     ),
            # )

            # vector_command = P.Value(
            #     value_name=ValueType.VECTOR_COMMAND.value,
            #     vector_command=P.VectorCommandValue(
            #         values=[self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd],
            #     ),
            # )

            # imu = P.Value(
            #     value_name=ValueType.IMU.value,
            #     imu=P.ImuValue(
            #         linear_acceleration=P.ImuAccelerometerValue(
            #             x=self.mj_data.sensor("accelerometer").data[0],
            #             y=self.mj_data.sensor("accelerometer").data[1],
            #             z=self.mj_data.sensor("accelerometer").data[2],
            #         ),
            #         angular_velocity=P.ImuGyroscopeValue(
            #             x=self.mj_data.sensor("angular-velocity").data[0],
            #             y=self.mj_data.sensor("angular-velocity").data[1],
            #             z=self.mj_data.sensor("angular-velocity").data[2],
            #         ),
            #         magnetic_field=P.ImuMagnetometerValue(
            #             x=self.mj_data.sensor("magnetometer").data[0],
            #             y=self.mj_data.sensor("magnetometer").data[1],
            #             z=self.mj_data.sensor("magnetometer").data[2],
            #         ),
            #     ),
            # )

            # seconds = int(simulation_time)
            # nanos = int((simulation_time - seconds) * 1e9)
            # timestamp = P.Value(
            #     value_name=ValueType.TIMESTAMP.value,
            #     timestamp=P.TimestampValue(
            #         seconds=seconds,
            #         nanos=nanos,
            #     ),
            # )

            # camera_frame_left = P.Value(
            #     value_name=ValueType.CAMERA_FRAME_LEFT.value,
            #     camera_frame=P.CameraFrameValue(
            #         data=self.viewer.read_pixels(camid=LEFT_CAMERA_ID).tobytes(),
            #     ),
            # )

            # camera_frame_right = P.Value(
            #     value_name=ValueType.CAMERA_FRAME_RIGHT.value,
            #     camera_frame=P.CameraFrameValue(
            #         data=self.viewer.read_pixels(camid=RIGHT_CAMERA_ID).tobytes(),
            #     ),
            # )

            full_observation = JointPositionsValue(
                joint_positions=joint_positions,
                # joint_velocities=joint_velocities,
                # vector_command=vector_command,
                # imu=imu,
                # timestamp=timestamp,
                # camera_frame_left=camera_frame_left,
                # camera_frame_right=camera_frame_right,
            )

            # Assemble observation space for the model
            inputs = P.IO(values=[])
            for key in self.model.schema_input_keys:
                try:
                    inputs.values.append(getattr(full_observation, key))
                except AttributeError:
                    raise AttributeError(f"Attribute {key} not found in FullObservation")

            schema_batch.append(inputs)
        breakpoint()
        return schema_batch

    def run(self) -> list[dict[str, metrics.BaseMetric]]:
        local_metrics = self._init_krec_metrics()

        for batch in self.dataloader:
            observation = self.create_observation(batch)
            output = self.model(observation)
            local_metrics = self.update_locomotion_metrics(local_metrics, output)

        return []
