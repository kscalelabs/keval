from dataclasses import dataclass
from typing import Any, Dict

from kinfer import proto as P
from omegaconf import DictConfig, ListConfig


@dataclass
class SchemaConverter:
    """Converts YAML schema definitions to kinfer IOSchema objects."""

    @staticmethod
    def _create_value_schema(value_def: Dict[str, Any]) -> P.ValueSchema:
        """Creates a ValueSchema from a value definition."""
        value_type = value_def["type"]
        schema = value_def["schema"]

        match value_type:
            case "joint_positions":
                return P.ValueSchema(
                    value_name=value_def["value_name"],
                    joint_positions=P.JointPositionsSchema(
                        joint_names=schema["joint_names"],
                        unit=P.JointPositionUnit.RADIANS if schema["unit"] == "radians" 
                            else P.JointPositionUnit.DEGREES,
                    )
                )
            case "joint_velocities":
                return P.ValueSchema(
                    value_name=value_def["value_name"],
                    joint_velocities=P.JointVelocitiesSchema(
                        joint_names=schema["joint_names"],
                        unit=P.JointVelocityUnit.RADIANS_PER_SECOND 
                            if schema["unit"] == "radians_per_second"
                            else P.JointVelocityUnit.DEGREES_PER_SECOND,
                    )
                )
            case "joint_torques":
                return P.ValueSchema(
                    value_name=value_def["value_name"],
                    joint_torques=P.JointTorquesSchema(
                        joint_names=schema["joint_names"],
                        unit=P.JointTorqueUnit.NEWTON_METERS,
                    )
                )
            case "joint_commands":
                return P.ValueSchema(
                    value_name=value_def["value_name"],
                    joint_commands=P.JointCommandsSchema(
                        joint_names=schema["joint_names"],
                        torque_unit=P.JointTorqueUnit.NEWTON_METERS,
                        velocity_unit=P.JointVelocityUnit.RADIANS_PER_SECOND
                            if schema["velocity_unit"] == "radians_per_second"
                            else P.JointVelocityUnit.DEGREES_PER_SECOND,
                        position_unit=P.JointPositionUnit.RADIANS
                            if schema["position_unit"] == "radians"
                            else P.JointPositionUnit.DEGREES,
                    )
                )
            case "vector_command":
                return P.ValueSchema(
                    value_name=value_def["value_name"],
                    vector_command=P.VectorCommandSchema(
                        dimensions=schema["dimensions"],
                    )
                )
            case "imu":
                return P.ValueSchema(
                    value_name=value_def["value_name"],
                    imu=P.ImuSchema(
                        use_accelerometer=schema["use_accelerometer"],
                        use_gyroscope=schema["use_gyroscope"],
                        use_magnetometer=schema["use_magnetometer"],
                    )
                )
            case "camera_frame":
                return P.ValueSchema(
                    value_name=value_def["value_name"],
                    camera_frame=P.CameraFrameSchema(
                        width=schema["width"],
                        height=schema["height"],
                        channels=schema["channels"],
                    )
                )
            case "audio_frame":
                return P.ValueSchema(
                    value_name=value_def["value_name"],
                    audio_frame=P.AudioFrameSchema(
                        channels=schema["channels"],
                        sample_rate=schema["sample_rate"],
                        dtype=getattr(P.DType, schema["dtype"].upper()),
                    )
                )
            case "timestamp":
                return P.ValueSchema(
                    value_name=value_def["value_name"],
                    timestamp=P.TimestampSchema(
                        start_seconds=schema["start_seconds"],
                        start_nanos=schema["start_nanos"],
                    )
                )
            case _:
                raise ValueError(f"Unknown value type: {value_type}")

    @staticmethod
    def create_schema(config: DictConfig | ListConfig) -> tuple[P.IOSchema, P.IOSchema]:
        """Creates input and output schemas from config."""
        if "io_schema" not in config:
            raise ValueError("No io_schema section found in config")

        schema_config = config.io_schema

        input_values = [
            SchemaConverter._create_value_schema(value_def)
            for value_def in schema_config.input
        ]

        output_values = [
            SchemaConverter._create_value_schema(value_def)
            for value_def in schema_config.output
        ]

        return (
            P.IOSchema(values=input_values),
            P.IOSchema(values=output_values)
        )
