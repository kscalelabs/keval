"""Temporary KRec dataloader before moving back to krec."""

import os
from glob import glob
from pathlib import Path
from typing import Union

import krec
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset
from torchcodec.decoders import VideoDecoder

from keval.observation import ValueType

TEMP_JOINT_MAPPING = {
    11: 0,
    12: 1,
    13: 2,
    14: 3,
    15: 4,
    21: 5,
    22: 6,
    23: 7,
    24: 8,
    25: 9,
}


class KRecDataset(Dataset):
    """Dataset class for loading KREC data."""

    def __init__(
        self,
        krec_dir: str,
        chunk_size: int = 64,
        image_history_size: int = 2,
        device: str = "cpu",
        has_video: bool = False,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_channels: int = 3,
    ) -> None:
        """Initialize the KRecDataset.

        Args:
            krec_dir: Directory containing the KREC files
            chunk_size: Number of frames in the action chunk
            image_history_size: Number of frames in the image history
            device: Device to load the data on ('cpu' or 'cuda')
            has_video: Whether the dataset has video frames
            camera_width: Width of camera frames
            camera_height: Height of camera frames
            camera_channels: Number of channels in camera frames
        """
        self.krec_dir: Path = Path(krec_dir)
        self.chunk_size = chunk_size
        self.image_history_size = max(1, image_history_size)
        self.device = device
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_channels = camera_channels

        self.filepaths: list[str] = sorted(
            glob(os.path.join(krec_dir, "*.krec")) + glob(os.path.join(krec_dir, "*.krec.mkv"))
        )
        self.has_video = has_video
        if not self.filepaths:
            raise ValueError(f"No .krec files found in {krec_dir}")

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> dict[str, Union[str, int, npt.NDArray]]:
        """Retrieve a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing:
                - camera_frame_left (NDArray): RGB video frame (H,W,C)
                - camera_frame_right (NDArray): RGB video frame (H,W,C)
                - timestamp (NDArray): Timestamp in seconds
                - joint_positions (NDArray): Joint positions array (num_joints,)
                - joint_velocities (NDArray): Joint velocities array (num_joints,)
                - imu (NDArray): IMU values
                - action_chunk (NDArray): Array of current and future actions (chunk_size, num_joints)
        """
        curr_fp = self.filepaths[idx]
        curr_krec_obj = KRecDataset.load_krec(curr_fp)

        # Randomly sample a frame index from the KREC frames
        krec_frame_idx = torch.randint(0, len(curr_krec_obj) - self.chunk_size, (1,)).item()
        krec_frame = curr_krec_obj[krec_frame_idx]

        # Create frame history as a single numpy array
        if self.has_video:
            video_decoder = VideoDecoder(curr_fp, device=self.device)
            video_frame = video_decoder[krec_frame.video_frame_number]
        else:
            video_frame = np.zeros((self.camera_height, self.camera_width, self.camera_channels), dtype=np.uint8)

        # Initialize arrays
        actuator_states = krec_frame.get_actuator_states()

        joint_positions = np.zeros(len(TEMP_JOINT_MAPPING), dtype=np.float32)
        joint_velocities = np.zeros(len(TEMP_JOINT_MAPPING), dtype=np.float32)
        curr_actions = np.zeros(len(TEMP_JOINT_MAPPING), dtype=np.float32)

        # States
        for state in actuator_states:
            joint_positions[TEMP_JOINT_MAPPING[state.actuator_id]] = state.position
            joint_velocities[TEMP_JOINT_MAPPING[state.actuator_id]] = state.velocity

        # Current actions
        for cmd in krec_frame.get_actuator_commands():
            curr_actions[TEMP_JOINT_MAPPING[cmd.actuator_id]] = cmd.position

        # IMU values
        imu_values = krec_frame.get_imu_values() if krec_frame.get_imu_values() is not None else np.zeros(6)

        # Get action chunk (current + future actions)
        actions = np.zeros((self.chunk_size, len(TEMP_JOINT_MAPPING)), dtype=np.float32)

        # Fill action chunk with available future actions
        for chunk_idx in range(self.chunk_size):
            future_frame_idx = krec_frame_idx + chunk_idx

            assert future_frame_idx < len(
                curr_krec_obj
            ), f"action chunk frame index {future_frame_idx} is out of bounds for len {len(curr_krec_obj)}"

            future_frame = curr_krec_obj[future_frame_idx]
            for cmd in future_frame.get_actuator_commands():
                actions[chunk_idx, TEMP_JOINT_MAPPING[cmd.actuator_id]] = cmd.position

        return {
            ValueType.CAMERA_FRAME_LEFT.value: video_frame,
            ValueType.CAMERA_FRAME_RIGHT.value: video_frame,
            ValueType.TIMESTAMP.value: np.array(
                [krec_frame.video_timestamp * 1e-9], dtype=np.float32
            ),  # Convert ns to seconds
            ValueType.JOINT_POSITIONS.value: joint_positions,
            ValueType.JOINT_VELOCITIES.value: joint_velocities,
            ValueType.IMU.value: imu_values,
            # ValueType.JOINT_TORQUES.value: actions,
        }

    @staticmethod
    def load_krec(file_path: str, verbose: bool = True) -> krec.KRec:
        """Smart loader that handles both direct KREC and MKV-embedded KREC files."""
        if file_path.endswith(".krec"):
            return krec.KRec.load(file_path)
        elif file_path.endswith(".krec.mkv"):
            return krec.extract_from_video(file_path, verbose=verbose)
        else:
            raise ValueError(f"Invalid file extension. Expected '.krec' or '.krec.mkv', got: {file_path}")


def get_krec_dataloader(
    krec_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True,
    device: str = "cpu",
) -> DataLoader:
    """Creates a DataLoader for KREC video files in a directory.

    Args:
        krec_dir: Directory containing .krec.mkv files
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the dataset
        device: Device to load data to ('cpu' or 'cuda')

    Returns:
        DataLoader configured for the KREC dataset
    """
    dataset = KRecDataset(krec_dir=krec_dir, device=device)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device == "cuda",
    )
