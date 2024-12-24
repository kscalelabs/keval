"""KREC dataloader."""

import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import decord
import krec
import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from datetime import datetime

NUM_ACTUATORS = 40


def pretty_print_batch(sample: Dict[str, Union[str, int, npt.NDArray]], batch_idx: Optional[int] = None) -> None:
    """Pretty print a sample from the KRec dataset.

    Args:
        sample: Dictionary containing the batch data with keys:
            - filepath (str): Path to the source file
            - frame_number (int): Frame number in the video
            - t (NDArray): Timestamp in seconds
            - joint_pos (NDArray): Joint positions array
            - joint_vel (NDArray): Joint velocities array
            - prev_actions (NDArray): Previous joint torques
            - curr_actions (NDArray): Current joint torques
            - ang_vel (NDArray): Angular velocities from IMU
            - euler_rotation (NDArray): Euler angles from IMU quaternion
            - frame_idx (int): Index of the frame
            - video_frame (NDArray): The RGB video frame
        batch_idx (Optional[int]): Index of the batch, if printing multiple batches
    """

    print(f"\n========== Batch: {batch_idx if batch_idx is not None else ''} ==========")
    print(f"Filepath: {sample['filepath']}")
    print(f"Frame Number: {sample['frame_number']}")
    print(f"Frame Index: {sample['frame_idx']}")
    print(f"t: {sample['t']}")
    print(f"Joint Position: {sample['joint_pos']}")
    print(f"Joint Velocity: {sample['joint_vel']}")
    print(f"Prev Actions: {sample['prev_actions']}")
    print(f"Curr Actions: {sample['curr_actions']}")
    print(f"Action Chunk: {sample['action_chunk']}")
    print(f"Ang Vel: {sample['ang_vel']}")
    print(f"Euler Rotation: {sample['euler_rotation']}")
    print(f"Video frame shape: {sample['video_frame'].shape}")
    print(f"Frame history shape: {sample['frame_history'].shape}")



""" two  """
def get_krec_file_type(file_path: str) -> str:
    """Determine if the file is a direct KREC file or MKV-embedded KREC.

    Returns:
        'krec' for .krec files
        'mkv' for .krec.mkv files
        raises RuntimeError for invalid extensions
    """
    if file_path.endswith(".krec"):
        return "krec"
    elif file_path.endswith(".krec.mkv"):
        return "mkv"
    else:
        error_msg = f"Invalid file extension. Expected '.krec' or '.krec.mkv', got: {file_path}"
        raise RuntimeError(error_msg)


def load_krec_direct(krec_file_path: str) -> krec.KRec:
    """Load a KREC file directly."""
    return krec.KRec.load(krec_file_path)


def load_krec_from_mkv(mkv_file_path: str, verbose: bool) -> krec.KRec:
    """Load a KREC file from an MKV file into a manually created temp directory."""
    if not os.path.exists(mkv_file_path):
        raise FileNotFoundError(f"File not found: {mkv_file_path}")

    return krec.extract_from_video(mkv_file_path, verbose=verbose)


def load_krec(file_path: str, verbose: bool = True) -> krec.KRec:
    """Smart loader that handles both direct KREC and MKV-embedded KREC files."""
    file_type = get_krec_file_type(file_path)
    return load_krec_direct(file_path) if file_type == "krec" else load_krec_from_mkv(file_path, verbose)


class KRecDataset(Dataset):
    """Dataset class for loading KREC (Kinova Recording) data.

    This dataset loads .krec.mkv files containing synchronized robot state and video data.
    Each sample contains robot joint states, IMU data, and corresponding video frames.

    Attributes:
        krec_dir (Path): Directory containing the KREC files
        filepaths (List[str]): List of paths to all .krec.mkv files
    """

    def __init__(
        self,
        krec_dir: str,
        chunk_size: int = 64,
        image_history_size: int = 2,
        device: str = "cpu",
        has_video: bool = False,
    ) -> None:
        """Initialize the KRecDataset.

        Args:
            krec_dir: Directory containing the KREC files
            chunk_size: Number of frames in the action chunk
            image_history_size: Number of frames in the image history
            has_video: Whether the dataset has video frames
            device: Device to load the data on ('cpu' or 'cuda')

        Raises:
            ValueError: If no .krec.mkv files are found in the directory
        """
        self.krec_dir: Path = Path(krec_dir)
        self.chunk_size = chunk_size
        self.image_history_size = max(1, image_history_size)

        # TODO: remove it
        self.filepaths: List[str] = sorted(glob(os.path.join(krec_dir, "*.krec")) + glob(os.path.join(krec_dir, "*.krec.mkv")))
        self.has_video = has_video
        if not self.filepaths:
            raise ValueError(f"No .krec files found in {krec_dir}")

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, int, npt.NDArray]]:
        """Retrieve a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing:
                - video_frame (NDArray): RGB video frame (H,W,C)
                - t (NDArray): Timestamp in seconds
                - joint_pos (NDArray): Joint positions array (num_joints,)
                - joint_vel (NDArray): Joint velocities array (num_joints,)
                - curr_actions (NDArray): Current joint torques (num_joints,)
                - prev_actions (NDArray): Previous joint torques (num_joints,)
                - ang_vel (NDArray): Angular velocities from IMU (3,)
                - euler_rotation (NDArray): Euler angles from quaternion (3,)
                - frame_idx (int): Index of the frame
                - frame_number (int): Frame number in video
                - filepath (str): Source file path
                - action_chunk (NDArray): Array of current and future actions (chunk_size, num_joints)

        Raises:
            ValueError: If IMU values are missing in the KREC frame
        """
        curr_fp = self.filepaths[idx]
        curr_krec_obj = load_krec(curr_fp)
        
        if self.has_video:
            video_reader = decord.VideoReader(curr_fp, ctx=decord.cpu(0))

        # Randomly sample a frame index from the KREC frames
        krec_frame_idx = torch.randint(0, len(curr_krec_obj) - self.chunk_size, (1,)).item()
        krec_frame = curr_krec_obj[krec_frame_idx]

        # Create frame history as a single numpy array
        if self.has_video:
            frame_history = np.zeros((self.image_history_size,) + video_reader[0].asnumpy().shape, dtype=np.uint8)
        else:
            frame_history = np.zeros((self.image_history_size,), dtype=np.uint8)

        for hist_idx in range(self.image_history_size):
            history_frame_idx = max(0, krec_frame.video_frame_number - hist_idx)
            if self.has_video:
                frame_history[hist_idx] = np.zeros_like(frame_history[0])

        if self.has_video:
            video_frame = video_reader[krec_frame.video_frame_number].asnumpy()

        # Initialize arrays
        actuator_states = krec_frame.get_actuator_states()
        joint_pos = np.zeros(NUM_ACTUATORS, dtype=np.float32)
        joint_vel = np.zeros(NUM_ACTUATORS, dtype=np.float32)
        curr_actions = np.zeros(NUM_ACTUATORS, dtype=np.float32)
        prev_actions = np.zeros(NUM_ACTUATORS, dtype=np.float32)

        # States
        for state in actuator_states:
            joint_pos[state.actuator_id] = state.position
            joint_vel[state.actuator_id] = state.velocity

        # Current actions
        for cmd in krec_frame.get_actuator_commands():
            curr_actions[cmd.actuator_id - 1] = cmd.position

        # Previous actions
        krec_prev_frame = curr_krec_obj[max(0, krec_frame_idx - 1)]
        for prev_cmd in krec_prev_frame.get_actuator_commands():
            prev_actions[prev_cmd.actuator_id - 1] = prev_cmd.position

        # IMU values
        imu_values = krec_frame.get_imu_values()
        if imu_values:
            angular_velocity = np.array([imu_values.gyro.x, imu_values.gyro.y, imu_values.gyro.z], dtype=np.float32)
            quaternion = np.array(
                [
                    imu_values.quaternion.x,
                    imu_values.quaternion.y,
                    imu_values.quaternion.z,
                    imu_values.quaternion.w,
                ]
            )
        else:
            angular_velocity = np.zeros(3, dtype=np.float32)
            quaternion = np.zeros(4, dtype=np.float32)

        # Get action chunk (current + future actions)
        action_chunk = np.zeros((self.chunk_size, NUM_ACTUATORS), dtype=np.float32)

        # Fill action chunk with available future actions
        for chunk_idx in range(self.chunk_size):
            future_frame_idx = krec_frame_idx + chunk_idx

            assert future_frame_idx < len(
                curr_krec_obj
            ), f"action chunk frame index {future_frame_idx} is out of bounds for len {len(curr_krec_obj)}"

            future_frame = curr_krec_obj[future_frame_idx]
            for cmd in future_frame.get_actuator_commands():
                action_chunk[chunk_idx, cmd.actuator_id - 1] = cmd.position

        if not self.has_video:
            video_frame = np.zeros_like(frame_history[0])

        return {
            "video_frame": video_frame,
            "frame_history": frame_history,
            "t": np.array([krec_frame.video_timestamp * 1e-9], dtype=np.float32),  # Convert ns to seconds
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "prev_actions": prev_actions,
            "curr_actions": curr_actions,
            "ang_vel": angular_velocity,
            "quaternion": quaternion,
            "frame_idx": krec_frame_idx,
            "frame_number": krec_frame.video_frame_number,
            "filepath": curr_fp,
            "action_chunk": action_chunk,
        }


def get_krec_dataloader(
    krec_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    transform: Optional[T.Compose] = None,
    shuffle: bool = True,
    device: str = "cpu",
) -> DataLoader:
    """Creates a DataLoader for KREC video files in a directory.

    Args:
        krec_dir: Directory containing .krec.mkv files
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        transform: Optional torchvision transforms to apply to video frames
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


if __name__ == "__main__":
    # Example usage
    krec_dir = "recordings"

    # Create dataset and dataloader
    dataset = KRecDataset(krec_dir)
    dataloader = get_krec_dataloader(krec_dir, batch_size=1)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of KREC files: {len(dataset.filepaths)}")

    # import pdb; pdb.set_trace()

    # Print first 5 samples
    for i, batch in enumerate(dataloader):
        pretty_print_batch(batch, batch_idx=i)
        if i >= 5:
            break
