"""Observation schema for the evaluation."""

from dataclasses import dataclass
from enum import Enum

from kinfer import proto as P

JOINT_NAMES = [
    "L_hip_y",
    "L_hip_z",
    "L_hip_x",
    "L_knee",
    "L_ankle",
    "R_hip_y",
    "R_hip_z",
    "R_hip_x",
    "R_knee",
    "R_ankle",
]


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
class DefaultCameraSetup:
    width: int = 640
    height: int = 480
    channels: int = 3


@dataclass
class KrecObservation:
    joint_positions: P.JointPositionsValue
    camera_frame_left: P.CameraFrameValue
    camera_frame_right: P.CameraFrameValue


@dataclass
class FullObservation:
    joint_positions: P.JointPositionsValue
    joint_velocities: P.JointVelocitiesValue
    vector_command: P.VectorCommandValue
    imu: P.ImuValue
    timestamp: P.TimestampValue
    camera_frame_left: P.CameraFrameValue
    camera_frame_right: P.CameraFrameValue


KREC_INPUT_SCHEMA = P.IOSchema(
    values=[
        P.ValueSchema(
            value_name=ValueType.JOINT_POSITIONS.value,
            joint_positions=P.JointPositionsSchema(
                joint_names=JOINT_NAMES,
                unit=P.JointPositionUnit.RADIANS,
            ),
        ),
        P.ValueSchema(
            value_name=ValueType.CAMERA_FRAME_LEFT.value,
            camera_frame=P.CameraFrameSchema(
                width=DefaultCameraSetup.width,
                height=DefaultCameraSetup.height,
                channels=DefaultCameraSetup.channels,
            ),
        ),
        P.ValueSchema(
            value_name=ValueType.CAMERA_FRAME_RIGHT.value,
            camera_frame=P.CameraFrameSchema(
                width=DefaultCameraSetup.width,
                height=DefaultCameraSetup.height,
                channels=DefaultCameraSetup.channels,
            ),
        ),
    ]
)


INPUT_SCHEMA = P.IOSchema(
    values=[
        P.ValueSchema(
            value_name=ValueType.JOINT_POSITIONS.value,
            joint_positions=P.JointPositionsSchema(
                joint_names=JOINT_NAMES,
                unit=P.JointPositionUnit.RADIANS,
            ),
        ),
        P.ValueSchema(
            value_name=ValueType.JOINT_VELOCITIES.value,
            joint_velocities=P.JointVelocitiesSchema(
                joint_names=JOINT_NAMES,
                unit=P.JointVelocityUnit.RADIANS_PER_SECOND,
            ),
        ),
        P.ValueSchema(
            value_name=ValueType.VECTOR_COMMAND.value,
            vector_command=P.VectorCommandSchema(
                dimensions=3,
            ),
        ),
        P.ValueSchema(
            value_name=ValueType.IMU.value,
            imu=P.ImuSchema(
                use_accelerometer=True,
                use_gyroscope=True,
                use_magnetometer=True,
            ),
        ),
        P.ValueSchema(
            value_name=ValueType.TIMESTAMP.value,
            timestamp=P.TimestampSchema(
                start_seconds=0,
                start_nanos=0,
            ),
        ),
        P.ValueSchema(
            value_name=ValueType.CAMERA_FRAME_LEFT.value,
            camera_frame=P.CameraFrameSchema(
                width=640,
                height=480,
                channels=3,
            ),
        ),
        P.ValueSchema(
            value_name=ValueType.CAMERA_FRAME_RIGHT.value,
            camera_frame=P.CameraFrameSchema(
                width=DefaultCameraSetup.width,
                height=DefaultCameraSetup.height,
                channels=DefaultCameraSetup.channels,
            ),
        ),
    ]
)


OUTPUT_SCHEMA = P.IOSchema(
    values=[
        P.ValueSchema(
            value_name=ValueType.JOINT_TORQUES.value,
            joint_torques=P.JointTorquesSchema(
                joint_names=JOINT_NAMES,
                unit=P.JointTorqueUnit.NEWTON_METERS,
            ),
        ),
    ]
)
