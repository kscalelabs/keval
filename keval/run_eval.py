"""Run evaluation for a given model and dataset.

Run:
    python keval/run_eval.py --config_path configs/locomotion.yaml \
        --model_path examples/gpr_walking.kinfer \
        --device cpu

    python keval/run_eval.py --config_path configs/locomotion.yaml \
        --model_path test_model.onnx \
        --device cpu
"""

import argparse
import logging
from pathlib import Path

import colorlogging
import onnxruntime as ort
import torch
from kinfer.inference.python import ONNXModel
from omegaconf import OmegaConf

from keval.evaluator import Evaluator

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

BASE_CONFIG_PATH = "configs/base.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=False)
    args = parser.parse_args()

    colorlogging.configure()
    logger = logging.getLogger(__name__)

    parsed_args = parser.parse_args()
    model_path: Path = parsed_args.model_path

    base_config = OmegaConf.load(BASE_CONFIG_PATH)
    updated_config = OmegaConf.load(parsed_args.config_path)
    config = OmegaConf.merge(base_config, updated_config)

    if parsed_args.device == "cuda":
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": torch.cuda.current_device(),
                    "user_compute_stream": str(torch.cuda.current_stream().cuda_stream),
                },
            )
        ]
    else:
        providers = ["CPUExecutionProvider"]

    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = 10
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)

    logger.warning("Using old inference session")
    model = ONNXModel(model_path)
    model.session = sess

    evaluator = Evaluator(config, model, logger)
    evaluator.run_eval()
