"""Defines the krec data runner.

TODO:
1. Add batch handling in the onnx definition.
"""

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

    def update_krec_metrics(
        self,
        local_metrics: dict[str, metrics.BaseMetric],
        batch: dict,
        output: list[float],
    ) -> dict[str, metrics.BaseMetric]:
        """Update the locomotion metrics.

        Args:
            local_metrics: The metrics to update.
            commanded_velocity: The commanded velocity.

        Returns:
            The updated metrics.
        """
        local_metrics[metrics.MetricsType.POSITION_ERROR.name].add_step(
            observed_position=batch["joint_pos"],
            commanded_position=output,
        )

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

            full_observation = JointPositionsValue(
                joint_positions=joint_positions,
            )

            # Assemble observation space for the model
            inputs = P.IO(values=[])
            for key in self.model.schema_input_keys:
                try:
                    inputs.values.append(getattr(full_observation, key))
                except AttributeError:
                    raise AttributeError(f"Attribute {key} not found in FullObservation")

            schema_batch.append(inputs)

        if len(schema_batch) == 1:
            return schema_batch[0]
        else:
            return schema_batch

    def run(self) -> list[dict[str, metrics.BaseMetric]]:
        local_metrics = self._init_krec_metrics()

        for batch in self.dataloader:
            observation = self.create_observation(batch)
            output_io = self.model(observation)
            output = self.model._output_serializer.serialize_io(output_io, as_dict=True)["joint_torques"]
            local_metrics = self.update_krec_metrics(local_metrics, batch, output)

        return local_metrics
