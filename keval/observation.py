"""Observation schema for the evaluation."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from kinfer import proto as P


class ValueType(Enum):
    JOINT_POSITIONS = "joint_positions"
    JOINT_VELOCITIES = "joint_velocities"
    VECTOR_COMMAND = "vector_command"
    IMU = "imu"
    CAMERA_FRAME_LEFT = "camera_frame_left"
    CAMERA_FRAME_RIGHT = "camera_frame_right"
    TIMESTAMP = "timestamp"
    JOINT_TORQUES = "joint_torques"

@dataclass
class KrecFullObservation:
    """Raw observation from KREC dataset."""
    # Joint states
    joint_positions: list[float]

    # Camera frames
    camera_frame_left: bytes
    camera_frame_right: bytes

    # Time
    timestamp_seconds: int = 0
    timestamp_nanos: int = 0

    def to_proto(self, schema: P.IOSchema) -> P.IO:
        """Converts raw observation to proto format based on schema."""
        io = P.IO(values=[])

        for value_schema in schema.values:
            value_name = value_schema.value_name
            value_type = value_schema.WhichOneof("value_type")

            match value_type:
                case "joint_positions":
                    io.values.append(P.Value(
                        value_name=value_name,
                        joint_positions=P.JointPositionsValue(
                            values=[
                                P.JointPositionValue(
                                    joint_name=name,
                                    value=self.joint_positions[i],
                                    unit=P.JointPositionUnit.RADIANS,
                                )
                                for i, name in enumerate(value_schema.joint_positions.joint_names)
                            ]
                        )
                    ))
                case "camera_frame":
                    match value_name:
                        case "camera_frame_left":
                            frame = self.camera_frame_left
                        case "camera_frame_right":
                            frame = self.camera_frame_right
                        case _:
                            raise ValueError(f"Invalid camera frame name: {value_name}")

                    io.values.append(P.Value(
                        value_name=value_name,
                        camera_frame=P.CameraFrameValue(
                            data=frame,
                        )
                    ))
                case "timestamp":
                    io.values.append(P.Value(
                        value_name=value_name,
                        timestamp=P.TimestampValue(
                            seconds=self.timestamp_seconds,
                            nanos=self.timestamp_nanos,
                        )
                    ))
                case _:
                    raise ValueError(f"KrecFullObservation does not support value type: {value_type}")

        return io

@dataclass
class MujocoFullObservation:
    """Raw observation from Mujoco simulation."""
    # Joint states
    qpos: list[float]
    qvel: list[float]

    # Commands
    x_vel_cmd: float
    y_vel_cmd: float
    yaw_vel_cmd: float

    # IMU readings
    accelerometer: Optional[list[float]] = None
    gyroscope: Optional[list[float]] = None
    magnetometer: Optional[list[float]] = None

    # Time
    simulation_time: float = 0.0

    # Camera frames
    camera_frame_left: Optional[bytes] = None
    camera_frame_right: Optional[bytes] = None

    def to_proto(self, schema: P.IOSchema) -> P.IO:
        """Converts raw observation to proto format based on schema."""
        io = P.IO(values=[])

        for value_schema in schema.values:
            value_name = value_schema.value_name
            value_type = value_schema.WhichOneof("value_type")

            match value_type:
                case "joint_positions":
                    io.values.append(P.Value(
                        value_name=value_name,
                        joint_positions=P.JointPositionsValue(
                            values=[
                                P.JointPositionValue(
                                    joint_name=name,
                                    value=self.qpos[-len(value_schema.joint_positions.joint_names) + i],
                                    unit=P.JointPositionUnit.RADIANS,
                                )
                                for i, name in enumerate(value_schema.joint_positions.joint_names)
                            ]
                        )
                    ))
                case "joint_velocities":
                    io.values.append(P.Value(
                        value_name=value_name,
                        joint_velocities=P.JointVelocitiesValue(
                            values=[
                                P.JointVelocityValue(
                                    joint_name=name,
                                    value=self.qvel[-len(value_schema.joint_velocities.joint_names) + i],
                                    unit=P.JointVelocityUnit.RADIANS_PER_SECOND,
                                )
                                for i, name in enumerate(value_schema.joint_velocities.joint_names)
                            ]
                        )
                    ))
                case "vector_command":
                    io.values.append(P.Value(
                        value_name=value_name,
                        vector_command=P.VectorCommandValue(
                            values=[self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd],
                        )
                    ))
                case "imu":
                    if value_schema.imu.use_accelerometer:
                        if type(self.accelerometer) is not list:
                            raise ValueError("Schema requires accelerometer but data not available")
                        if len(self.accelerometer) != 3:
                            raise ValueError("Schema requires accelerometer but data not available")

                    if value_schema.imu.use_gyroscope:
                        if type(self.gyroscope) is not list:
                            raise ValueError("Schema requires gyroscope but data not available")
                        if len(self.gyroscope) != 3:
                            raise ValueError("Schema requires gyroscope but data not available")

                    if value_schema.imu.use_magnetometer:
                        if type(self.magnetometer) is not list:
                            raise ValueError("Schema requires magnetometer but data not available")
                        if len(self.magnetometer) != 3:
                            raise ValueError("Schema requires magnetometer but data not available")

                    io.values.append(P.Value(
                        value_name=value_name,
                        imu=P.ImuValue(
                            linear_acceleration=P.ImuAccelerometerValue( 
                                x=self.accelerometer[0], y=self.accelerometer[1], z=self.accelerometer[2]  # type: ignore[index]
                            ) if value_schema.imu.use_accelerometer else None,
                            angular_velocity=P.ImuGyroscopeValue(
                                x=self.gyroscope[0], y=self.gyroscope[1], z=self.gyroscope[2]  # type: ignore[index]
                            ) if value_schema.imu.use_gyroscope else None,
                            magnetic_field=P.ImuMagnetometerValue(
                                x=self.magnetometer[0], y=self.magnetometer[1], z=self.magnetometer[2]  # type: ignore[index]
                            ) if value_schema.imu.use_magnetometer else None,
                        )
                    ))
                case "timestamp":
                    io.values.append(P.Value(
                        value_name=value_name,
                        timestamp=P.TimestampValue(
                            seconds=int(self.simulation_time),
                            nanos=int((self.simulation_time - int(self.simulation_time)) * 1e9),
                        )
                    ))
                case "camera_frame":
                    match value_name:
                        case "camera_frame_left":
                            if self.camera_frame_left is None:
                                raise ValueError("Schema requires camera frame left but data not available")
                            frame = self.camera_frame_left
                        case "camera_frame_right":
                            if self.camera_frame_right is None:
                                raise ValueError("Schema requires camera frame right but data not available")
                            frame = self.camera_frame_right
                        case _:
                            raise ValueError(f"Invalid camera frame name: {value_name}")

                    io.values.append(P.Value(
                        value_name=value_name,
                        camera_frame=P.CameraFrameValue(
                            data=frame,
                        )
                    ))
                case _:
                    raise ValueError(f"MujocoFullObservation does not support value type: {value_type}")

        return io
