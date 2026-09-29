from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Literal

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import disable_progress_bar, enable_progress_bar

from config import ModelConfig, get_lora_targets


def _is_distributed() -> bool:
    """Return True if this process is part of a torch.distributed run.

    ``torchrun`` / ``torch.distributed.launch`` set ``WORLD_SIZE``, ``RANK``
    and ``LOCAL_RANK`` before the script starts, but ``torch.distributed`` is
    often not initialized yet when this helper is called (the HuggingFace
    ``Trainer`` initializes it later). Detecting distributed mode from the
    environment avoids incorrectly falling back to single-GPU
    ``device_map="auto"`` on every rank, which causes all ranks to try to
    load the full model on the same GPUs and hang.
    """
    if not torch.distributed.is_available():
        return False
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size > 1


def get_device(preference: Literal["auto", "cuda", "cpu"] = "auto") -> torch.device:
    """Return the torch device to use."""
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(model_name: str, max_length: int = 512) -> AutoTokenizer:
    """Load and configure a tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = max_length
    return tokenizer


def load_base_model(
    model_name: str,
    config: ModelConfig | None = None,
    device: torch.device | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a base causal LM in bf16/fp16."""
    config = config or ModelConfig()
    device = device or get_device()

    tokenizer = load_tokenizer(model_name, max_length=config.max_seq_length)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

    # Use bf16/fp16 directly (no 4-bit quantization) to maximize model capacity
    # and match the paper reproduction.
    bnb_config = None

    # Under torchrun/DDP/FSDP, accelerate's device_map must not be used and the
    # Trainer handles device placement. Otherwise use device_map="auto" on CUDA.
    is_distributed = _is_distributed()
    device_map = None if is_distributed else ("auto" if device.type == "cuda" else None)

    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        target_device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        print(
            f"Distributed run detected (world_size={world_size}, rank={rank}, "
            f"local_rank={local_rank}). Loading model with device_map={device_map}; "
            f"Trainer will place model on {target_device}."
        )
        # With 8 ranks loading concurrently, the interleaved Transformers progress
        # bars make the log unreadable and can slow down the terminal. Disable
        # them during weight loading; we will re-enable afterwards.
        with contextlib.suppress(Exception):
            disable_progress_bar()
        # Stagger model loading so all ranks do not hit the shared filesystem
        # at the exact same moment. Each rank waits local_rank*5 seconds.
        sleep_seconds = local_rank * 5
        if sleep_seconds:
            print(
                f"Rank {rank} (local {local_rank}): sleeping {sleep_seconds}s "
                "before loading weights to reduce filesystem contention..."
            )
            import time

            time.sleep(sleep_seconds)
        print(f"Rank {rank}: loading base model weights (target device: {target_device})...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )

    if is_distributed:
        with contextlib.suppress(Exception):
            enable_progress_bar()
        print(
            f"Rank {rank}: base model weights loaded (currently on "
            f"{next(model.parameters()).device}; Trainer will move to {target_device})."
        )

    # Only move the model manually in single-process runs. In distributed runs
    # the Trainer / FSDP will place the model on the correct device.
    if device.type == "cpu" or (device.type == "cuda" and not is_distributed):
        model = model.to(device)

    model.eval()
    return model, tokenizer


def add_lora(
    model: AutoModelForCausalLM,
    config: ModelConfig | None = None,
) -> PeftModel:
    """Wrap a base model with LoRA adapters."""
    config = config or ModelConfig()
    target_modules = get_lora_targets(config.base_model)
    lora_cfg = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    return model


def _summarize_trainable_params(model: torch.nn.Module) -> tuple[int, float]:
    """Return the number of trainable params and their mean absolute value."""
    total = 0
    weighted_sum = 0.0
    for p in model.parameters():
        if p.requires_grad:
            n = p.numel()
            total += n
            weighted_sum += float(p.abs().mean()) * n
    mean_abs = weighted_sum / total if total > 0 else 0.0
    return total, mean_abs


def infer_base_model_name(
    model_path: Path,
    fallback: str | None = None,
) -> str:
    """Infer the base model name from a checkpoint.

    If the checkpoint contains a LoRA adapter config, read the
    ``base_model_name_or_path`` field. Otherwise return ``fallback`` or the
    project default.

    Args:
        model_path: Directory containing the fine-tuned model or adapter.
        fallback: Base model name to use when no adapter config is found.

    Returns:
        The inferred base model identifier.
    """
    adapter_config_path = model_path / "adapter_config.json"
    if adapter_config_path.exists():
        with adapter_config_path.open("r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        return adapter_cfg.get("base_model_name_or_path", fallback or ModelConfig().base_model)
    return fallback or ModelConfig().base_model


def load_finetuned_model(
    model_path: Path,
    base_model_name: str | None = None,
    config: ModelConfig | None = None,
    device: torch.device | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a fine-tuned model (full weights or LoRA adapter)."""
    config = config or ModelConfig()
    device = device or get_device()

    if base_model_name is None:
        base_model_name = infer_base_model_name(model_path, fallback=config.base_model)

    tokenizer = load_tokenizer(base_model_name, max_length=config.max_seq_length)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

    # Use bf16/fp16 directly (no 4-bit quantization).
    bnb_config = None

    is_distributed = _is_distributed()
    device_map = None if is_distributed else ("auto" if device.type == "cuda" else None)

    has_adapter = (model_path / "adapter_config.json").exists()
    has_full_weights = any(
        (model_path / fname).exists()
        for fname in (
            "pytorch_model.bin",
            "model.safetensors",
            "model.safetensors.index.json",
        )
    )

    if has_adapter:
        # LoRA: load base model first, then adapter weights.
        print(f"Loading LoRA adapter from {model_path} ...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        if device.type == "cpu" or (device.type == "cuda" and not is_distributed):
            model = model.to(device)
        model = PeftModel.from_pretrained(model, str(model_path))
        trainable, mean_abs = _summarize_trainable_params(model)
        print(f"  LoRA adapter loaded: {trainable:,} trainable parameters")
        print(f"  Mean absolute adapter weight: {mean_abs:.6f}")
    elif has_full_weights:
        # Full fine-tune: load the saved weights directly from model_path.
        print(f"Loading full fine-tuned weights from {model_path} ...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        if device.type == "cpu" or (device.type == "cuda" and not is_distributed):
            model = model.to(device)
        total, mean_abs = _summarize_trainable_params(model)
        print(f"  Full model loaded: {total:,} parameters")
        print(f"  Mean absolute weight: {mean_abs:.6f}")
    else:
        raise FileNotFoundError(
            f"No adapter or full weights found at {model_path}. "
            "Expected one of: adapter_config.json, pytorch_model.bin, "
            "model.safetensors, or model.safetensors.index.json."
        )

    model.eval()
    return model, tokenizer


def count_trainable_params(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
