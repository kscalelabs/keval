"""E2E tests on random models."""

import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from kinfer import proto as P
from kinfer.export.pytorch import export_model
from kinfer.inference.python import ONNXModel

from keval.observation import (
    ATTACHED_METADATA,
    JOINT_NAMES,
    input_schema,
    output_schema,
)

TEST_MODEL_PATH = Path("test_model.onnx")


@dataclass
class ModelConfig:
    in_features: int = 10
    out_features: int = 10
    hidden_size: int = 10
    num_layers: int = 2


class SimpleModel(torch.nn.Module):
    """A simple neural network model for demonstration."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        layers = []

        total_input_features = config.in_features * 2
        in_features = total_input_features

        for _ in range(config.num_layers):
            layers.extend([torch.nn.Linear(in_features, config.hidden_size), torch.nn.ReLU()])
            in_features = config.hidden_size

        layers.append(torch.nn.Linear(config.hidden_size, config.out_features))
        self.net = torch.nn.Sequential(*layers)

    def forward(
        self,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
    ) -> torch.Tensor:
        # Concatenate positions and velocities
        combined_input = torch.cat([joint_positions, joint_velocities[0]], dim=-1)
        joint_torques = self.net(combined_input)
        return joint_torques


def create_model(save_path: Path) -> str:
    """Create and export a test model."""
    # Create and export model
    config = ModelConfig()
    model = SimpleModel(config)

    jit_model = torch.jit.script(model)

    exported_model = export_model(
        model=jit_model,
        schema=P.ModelSchema(
            input_schema=input_schema,
            output_schema=output_schema,
        ),
    )
    onnx.save_model(exported_model, save_path)

    return save_path


class TestRandomModel(unittest.TestCase):
    def setUp(self):
        pass

    def test_random_loading(self) -> None:
        """Test basic model loading functionality."""
        save_path = create_model(TEST_MODEL_PATH)
        model = ONNXModel(save_path)

        # TODO: Remove this
        model.attached_metadata = ATTACHED_METADATA
        radians = P.JointPositionUnit.RADIANS
        inputs = P.IO(
            values=[
                P.Value(
                    value_name="joint_positions",
                    joint_positions=P.JointPositionsValue(
                        values=[
                            P.JointPositionValue(
                                joint_name=name,
                                value=float(np.random.randn()),
                                unit=radians,
                            )
                            for name in JOINT_NAMES
                        ]
                    ),
                ),
                P.Value(
                    value_name="joint_velocities",
                    state_tensor=P.StateTensorValue(data=np.random.randn(1, 10).astype(np.float32).tobytes()),
                ),
            ],
        )
        outputs = model(inputs)

        # outputs_np = model._output_serializer.serialize_io(outputs, as_dict=True)
        assert isinstance(outputs, P.IO)


if __name__ == "__main__":
    unittest.main()
