"""Defines the krec data runner.

TODO:
1. Add batch handling in the onnx definition.
"""

import numpy as np
from kinfer import proto as P
from kinfer.inference.python import ONNXModel
from omegaconf import DictConfig, ListConfig, OmegaConf

from keval import metrics
from keval.experimental.krec_dataloader import KRecDataset, get_krec_dataloader
from keval.observation import JOINT_NAMES, KrecObservation, ValueType
from keval.runners.base_runner import Runner


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
        local_metrics: dict[str, metrics.BaseMetric] = {
            metrics.MetricsType.POSITION_ERROR.name: metrics.PositionError(),
        }
        return local_metrics

    def update_krec_metrics(
        self,
        local_metrics: dict[str, metrics.BaseMetric],
        batch: dict,
        output: np.ndarray,
    ) -> dict[str, metrics.BaseMetric]:
        """Update the KRec metrics.

        Args:
            local_metrics: The metrics to update.
            batch: The batch of data.
            output: The output of the model.

        Returns:
            The updated metrics.
        """
        local_metrics[metrics.MetricsType.POSITION_ERROR.name].add_step(
            observed_position=batch[ValueType.JOINT_POSITIONS.value],
            predicted_position=output,
        )

        return local_metrics

    def create_observation(self, batch: dict) -> P.IO:
        """Create the observation for the model.

        Args:
            batch: The batch of data.

        Returns:
            The observation for the model.
        """
        for joint_positions, camera_frame_left, camera_frame_right in zip(
            batch[ValueType.JOINT_POSITIONS.value],
            batch[ValueType.CAMERA_FRAME_LEFT.value],
            batch[ValueType.CAMERA_FRAME_RIGHT.value],
        ):
            joint_positions = P.Value(
                value_name=ValueType.JOINT_POSITIONS.value,
                joint_positions=P.JointPositionsValue(
                    values=[
                        P.JointPositionValue(
                            joint_name=name,
                            value=joint_positions[index],
                            unit=P.JointPositionUnit.RADIANS,
                        )
                        for index, name in enumerate(JOINT_NAMES)
                    ]
                ),
            )

            camera_frame_left = P.Value(
                value_name=ValueType.CAMERA_FRAME_LEFT.value,
                camera_frame=P.CameraFrameValue(
                    data=camera_frame_left.numpy().tobytes(),
                ),
            )

            camera_frame_right = P.Value(
                value_name=ValueType.CAMERA_FRAME_RIGHT.value,
                camera_frame=P.CameraFrameValue(
                    data=camera_frame_right.numpy().tobytes(),
                ),
            )

            full_observation = KrecObservation(
                joint_positions=joint_positions,
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

    def run(self) -> list[dict[str, metrics.BaseMetric]]:
        """Run the KRec runner.

        Returns:
            metrics
        """
        local_metrics = self._init_krec_metrics()

        for batch in self.dataloader:
            observation = self.create_observation(batch)
            output_io = self.model(observation)
            output = self.model._output_serializer.serialize_io(output_io, as_dict=True)["joint_torques"]
            local_metrics = self.update_krec_metrics(local_metrics, batch, output)

        return [local_metrics]
