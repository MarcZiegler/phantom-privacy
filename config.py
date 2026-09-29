from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Literal


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Base model and fine-tuning configuration."""

    base_model: str = "EleutherAI/gpt-neo-125M"
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    max_seq_length: int = 512
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters."""

    output_dir: Path = Path("models/finetuned")
    num_epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    warmup_steps: int = 200
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    seed: int = 42
    device: Literal["auto", "cuda", "cpu"] = "auto"


@dataclasses.dataclass(frozen=True)
class AttackConfig:
    """Attack evaluation hyperparameters."""

    model_path: Path = Path("models/finetuned")
    data_path: Path = Path("preprocessed/enron.jsonl")
    max_users: int | None = None
    max_samples_per_user: int = 100
    batch_size: int = 8
    device: Literal["auto", "cuda", "cpu"] = "auto"
    seed: int = 42


@dataclasses.dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing defaults."""

    min_examples_per_user: int = 28
    holdout_fraction: float = 0.10
    min_body_words: int = 5
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Model-family helpers
# ---------------------------------------------------------------------------
def get_lora_targets(model_name: str) -> tuple[str, ...]:
    """Return the default LoRA target modules for a given model family."""
    name_lower = model_name.lower()

    if "gpt-neo" in name_lower:
        return (
            "attn.attention.q_proj",
            "attn.attention.k_proj",
            "attn.attention.v_proj",
            "attn.attention.out_proj",
        )
    if "gpt2" in name_lower:
        return ("c_attn", "c_proj")
    if "gpt-j" in name_lower or "gptj" in name_lower:
        return ("q_proj", "v_proj")

    # Default to Llama-style attention modules.
    return ("q_proj", "k_proj", "v_proj", "o_proj")


def model_name_to_dir(model_name: str) -> str:
    """Convert a HuggingFace model ID to a local folder name."""
    name = model_name.split("/")[-1].lower()
    name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    # Use a dot-free parameter-size label for GPT-Neo-1.3B to match the 125M cache.
    if name == "gpt-neo-1-3b":
        return "gpt-neo-1300m"
    return name


DEFAULT_SEED = 42
