"""Observation schema for the evaluation."""

from kinfer import proto as P

ATTACHED_METADATA = {
    "sim_dt": 0.01,
    "sim_decimation": 1,
    "tau_factor": 1.0,
}


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


input_schema = P.IOSchema(
    values=[
        P.ValueSchema(
            value_name="joint_positions",
            joint_positions=P.JointPositionsSchema(
                joint_names=JOINT_NAMES,
                unit=P.JointPositionUnit.RADIANS,
            ),
        ),
        P.ValueSchema(
            value_name="joint_velocities",
            joint_velocities=P.JointVelocitiesSchema(
                joint_names=JOINT_NAMES,
                unit=P.JointVelocityUnit.RADIANS_PER_SECOND,
            ),
        ),
        P.ValueSchema(
            value_name="vector_command",
            vector_command=P.VectorCommandSchema(
                dimensions=3,
            ),
        ),
        P.ValueSchema(
            value_name="imu",
            imu=P.ImuSchema(
                use_accelerometer=True,
                use_gyroscope=True,
                use_magnetometer=True,
            ),
        ),
        P.ValueSchema(
            value_name="timestamp",
            timestamp=P.TimestampSchema(
                start_seconds=0,
                start_nanos=0,
            ),
        ),
        P.ValueSchema(
            value_name="camera_frame_left",
            camera_frame=P.CameraFrameSchema(
                width=640,
                height=480,
                channels=3,
            ),
        ),
        P.ValueSchema(
            value_name="camera_frame_right",
            camera_frame=P.CameraFrameSchema(
                width=640,
                height=480,
                channels=3,
            ),
        ),
    ]
)


# output_schema = P.IOSchema(
#     values=[
#         P.ValueSchema(
#             value_name="joint_torques",
#             state_tensor=P.StateTensorSchema(
#                 shape=[1, 10],
#                 dtype=P.DType.FP32,
#             ),
#         ),
#     ]
# )

output_schema = P.IOSchema(
    values=[
        P.ValueSchema(
            value_name="joint_torques",
            joint_torques=P.JointTorquesSchema(
                joint_names=JOINT_NAMES,
                unit=P.JointTorqueUnit.NEWTON_METERS,
            ),
        ),
    ]
)
