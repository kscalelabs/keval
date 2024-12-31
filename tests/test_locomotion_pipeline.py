"""E2E tests on random models."""

import logging
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
import torch
from kinfer import proto as P
from kinfer.export.pytorch import export_model
from kinfer.inference.python import ONNXModel
from omegaconf import DictConfig, ListConfig, OmegaConf

from keval.evaluator import Evaluator
from keval.observation import INPUT_SCHEMA, OUTPUT_SCHEMA
from keval.runners.mujoco_runner import ATTACHED_METADATA

TEST_MODEL_PATH = Path("test_model.onnx")


@dataclass
class ModelConfig:
    joint_positions_length: int = 10
    joint_velocities_length: int = 10
    vector_command_length: int = 3
    imu_length: int = 9
    timestamp_length: int = 1
    hidden_size: int = 10
    num_layers: int = 2
    camera_frame_left_length: int = 640 * 480 * 3
    camera_frame_right_length: int = 640 * 480 * 3


class SimpleModel(torch.nn.Module):
    """A simple neural network model for demonstration."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        layers = []

        total_input_features = (
            config.joint_positions_length
            + config.joint_velocities_length
            + config.vector_command_length
            + config.imu_length
            + config.timestamp_length
            + config.camera_frame_left_length
            + config.camera_frame_right_length
        )
        in_features = total_input_features

        for _ in range(config.num_layers):
            layers.extend([torch.nn.Linear(in_features, config.hidden_size), torch.nn.ReLU()])
            in_features = config.hidden_size

        layers.append(torch.nn.Linear(config.hidden_size, config.joint_positions_length))
        self.net = torch.nn.Sequential(*layers)

    def forward(
        self,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        vector_command: torch.Tensor,
        imu: torch.Tensor,
        timestamp: torch.Tensor,
        camera_frame_left: torch.Tensor,
        camera_frame_right: torch.Tensor,
    ) -> torch.Tensor:
        combined_input = torch.cat(
            [
                joint_positions,
                joint_velocities,
                vector_command,
                imu.reshape(-1),
                timestamp,
                camera_frame_left.reshape(-1),
                camera_frame_right.reshape(-1),
            ],
            dim=-1,
        )
        joint_torques = self.net(combined_input)
        return joint_torques


def create_model(save_path: Path) -> Path | str:
    """Create and export a test model.

    Args:
        save_path: The path to save the model.

    Returns:
        The path to the saved model.
    """
    config = ModelConfig()
    model = SimpleModel(config)

    jit_model = torch.jit.script(model)

    exported_model = export_model(
        model=jit_model,
        schema=P.ModelSchema(
            input_schema=INPUT_SCHEMA,
            output_schema=OUTPUT_SCHEMA,
        ),
    )
    onnx.save_model(exported_model, save_path)

    return save_path


class TestEvalLocomotionPipeline(unittest.TestCase):
    """Test the full evaluation pipeline."""

    @contextmanager
    def temp_test_dir(self) -> Iterator[Path]:
        """Create a temporary directory for test artifacts.

        Yields:
            Path to temporary directory that is automatically cleaned up
        """
        with TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def setUp(self) -> None:
        """Set up test environment."""
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config()

    def _load_config(self) -> ListConfig | DictConfig | OmegaConf:
        """Load and merge configuration files."""
        base_config = OmegaConf.load("configs/base.yaml")
        test_config = OmegaConf.load("tests/test_locomotion_config.yaml")
        return OmegaConf.merge(base_config, test_config)

    def test_locomotion_pipeline(self) -> None:
        """Test locomotion pipeline."""
        with self.temp_test_dir() as test_dir:
            model_path = test_dir / "test_model.onnx"
            create_model(model_path)
            model = ONNXModel(model_path)
            # TODO: Remove this
            model.attached_metadata = ATTACHED_METADATA  # type: ignore[assignment]

            evaluator = Evaluator(self.config, model, self.logger)
            evaluator.run_eval()

            data_dir = Path(self.config.logging.log_dir, "locomotion")
            video_files = list(data_dir.glob("*.mp4"))
            self.assertEqual(
                len(video_files),
                2,
                f"Expected 2 video files in {data_dir}, found {len(video_files)}",
            )

            # Check if metrics file exists
            metrics_file = Path(data_dir, "averaged_metrics.yaml")
            self.assertTrue(
                metrics_file.exists(),
                f"Expected metrics file at {metrics_file} does not exist",
            )

            # Check if tracking error file exists
            tracking_error_plots = list(data_dir.glob("tracking_error_*.png"))
            self.assertEqual(
                len(tracking_error_plots),
                6,
                f"Expected 6 tracking error plots in {data_dir}, found {len(tracking_error_plots)}",
            )


if __name__ == "__main__":
    unittest.main()
