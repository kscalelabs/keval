"""Defines the krec data runner.

TODO:
1. Add batch handling in the onnx definition.
"""

import numpy as np
from kinfer import proto as P
from kinfer.inference.python import ONNXModel
from omegaconf import DictConfig

from keval import metrics
from keval.experimental.krec_dataloader import KRecDataset, get_krec_dataloader
from keval.observation import KrecFullObservation, ValueType
from keval.runners.base_runner import Runner


class KrecRunner(Runner):
    def __init__(self, config: DictConfig, model: ONNXModel, metrics: metrics.Metrics) -> None:
        """Initialize the KrecRunner.

        Args:
            config: The configuration.
            model: The model to evaluate.
            metrics: The metrics to use.
        """
        super().__init__(config, model, metrics)

        # Get camera config from schema
        camera_schema = next(
            value["schema"] for value in self.config.io_schema.input 
            if value.get("type") == "camera_frame" and value.get("value_name") == "camera_frame_left"
        )

        self.dataset = KRecDataset(
            krec_dir=self.config.dataset_path,
            camera_width=camera_schema["width"],
            camera_height=camera_schema["height"],
            camera_channels=camera_schema["channels"],
        )
        self.dataloader = get_krec_dataloader(self.config.dataset_path, batch_size=1)
        self.model = model
        self.metrics = metrics

    def _init_krec_metrics(
        self,
    ) -> dict[str, metrics.BaseMetric]:
        """Initialize the metrics for the locomotion suite.

        Returns:
            A dictionary of metrics.
        """
        local_metrics: dict[str, metrics.BaseMetric] = {
            metrics.MetricsType.POSITION_ERROR.name: metrics.PositionError(config=self.config),
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
        raw_obs = KrecFullObservation(
            joint_positions=batch[ValueType.JOINT_POSITIONS.value].tolist(),
            camera_frame_left=batch[ValueType.CAMERA_FRAME_LEFT.value].numpy().tobytes(),
            camera_frame_right=batch[ValueType.CAMERA_FRAME_RIGHT.value].numpy().tobytes(),
            timestamp_seconds=batch[ValueType.TIMESTAMP.value][0],
            timestamp_nanos=batch[ValueType.TIMESTAMP.value][1],
        )

        return raw_obs.to_proto(self.input_schema)

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
