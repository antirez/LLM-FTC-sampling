import glob
import json
import logging
from pathlib import Path
from typing import Generator, Tuple

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np
import random, time
import matplotlib.pyplot as plt

# Local imports
from .models import llama, phi2
from .models.base import BaseModelArgs

# Constants
MODEL_MAPPING = {
    "llama": llama,
    "mistral": llama,  # mistral is compatible with llama
    "phi": phi2,
}


def _get_classes(config: dict):
    """
    Retrieve the model and model args classes based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    model_type = config["model_type"]
    if model_type not in MODEL_MAPPING:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

    arch = MODEL_MAPPING[model_type]
    return arch.Model, arch.ModelArgs


def get_model_path(path_or_hf_repo: str) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                allow_patterns=["*.json", "*.safetensors", "*.py", "tokenizer.model"],
            )
        )
    return model_path


def generate_step(
    prompt: mx.array, model: nn.Module, cutoff: float = 7.0
) -> Generator[mx.array, None, None]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling. If temp is 0, use max sampling.

    Yields:
        Generator[mx.array]: A generator producing one token per call.
    """

    def sample(logits: mx.array) -> mx.array:
        random.seed(time.monotonic_ns())
        mx.random.seed(time.monotonic_ns())

        logits = mx.softmax(logits)
        np_logits = np.array(logits) # MX -> NumPy
        np_logits = np_logits.flatten()
        sorted_indices = np.argsort(np_logits)
        sorted_indices = sorted_indices[::-1]

        j = 1
        t0 = np_logits[sorted_indices[0]]
        while 1 - (np_logits[sorted_indices[j]] / t0) < cutoff and j < len(np_logits):
            j += 1
        accepted_logits = []
        for i in range(0,j):
            accepted_logits.append(float(np_logits[sorted_indices[j]]))
        accepted_logits = mx.array(accepted_logits)
        idx = mx.random.categorical(accepted_logits)
        idx = int(np.array(idx)) # We can't convert zero-dim array without passing from numpy
        c = sorted_indices[idx]
        return (t0,mx.array(np.array([c])))

    y = prompt
    cache = None
    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        t0,y = sample(logits)
        yield (t0,y)


def generate(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    temp: float = 0.0,
    max_tokens: int = 100,
    verbose: bool = False,
) -> str:
    """
    Generate text from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       temp (float): The temperature for sampling (default 0).
       max_tokens (int): The maximum number of tokens (default 100).
    """

    prompt = mx.array(tokenizer.encode(prompt))

    tokens = []
    skip = 0
    for token, _ in zip(generate_step(prompt, model, temp), range(max_tokens)):
        if token == tokenizer.eos_token_id:
            break

        tokens.append(token.item())

        if verbose:
            s = tokenizer.decode(tokens)
            print(s[skip:], end="", flush=True)
            skip = len(s)

    tokens = tokenizer.decode(tokens)[skip:]
    if verbose:
        print(tokens, flush=True)
    return tokens


def load(path_or_hf_repo: str) -> Tuple[nn.Module, PreTrainedTokenizer]:
    """
    Load the model from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (str): The path or the huggingface repository to load the model from.

    Returns:
        Tuple[nn.Module, PreTrainedTokenizer]: The loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    model_path = get_model_path(path_or_hf_repo)

    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
            quantization = config.get("quantization", None)
    except FileNotFoundError:
        logging.error(f"Config file not found in {model_path}")
        raise
    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if not weight_files:
        logging.error(f"No safetensors found in {model_path}")
        raise FileNotFoundError(f"No safetensors found in {model_path}")
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model_class, model_args_class = _get_classes(config=config)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    if quantization is not None:
        nn.QuantizedLinear.quantize_module(model, **quantization)

    model.load_weights(list(weights.items()))

    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer
