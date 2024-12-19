"""Observation schema for the evaluation."""

from dataclasses import dataclass, field
from typing import List

import numpy as np
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
            state_tensor=P.StateTensorSchema(
                shape=[1, 10],
                dtype=P.DType.FP32,
            ),
        ),
    ]
)


output_schema = P.IOSchema(
    values=[
        P.ValueSchema(
            value_name="joint_torques",
            state_tensor=P.StateTensorSchema(
                shape=[1, 10],
                dtype=P.DType.FP32,
            ),
        ),
    ]
)
