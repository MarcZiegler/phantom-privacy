#!/usr/bin/env python
"""Fine-tune a base LLM on preprocessed user data.

Supports full-parameter fine-tuning (paper default) and optional LoRA for
faster experimentation.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from config import ModelConfig, TrainConfig
from datasets import Dataset as HFDataset
from utils.model_utils import (
    add_lora,
    get_device,
    infer_base_model_name,
    load_base_model,
    load_tokenizer,
)
from utils.training_logger import TrainingLogger
from utils.training_viz_utils import format_training_summary, plot_training_curves
from utils.utility_utils import (
    evaluate_texts,
    format_utility_results,
    load_split_texts,
    save_utility_metrics,
)


def _is_distributed_run() -> bool:
    """Return True if this is a multi-GPU torchrun/DDP/FSDP run."""
    if not torch.distributed.is_available():
        return False
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size > 1


def _disable_inductor_for_distributed() -> None:
    """Disable TorchInductor compile workers on multi-GPU runs.

    The Inductor subprocess pool is forked by PyTorch and can deadlock when
    CUDA has already been initialized in the parent process. This only
    affects distributed runs; single-GPU runs are left unchanged.
    """
    if not _is_distributed_run():
        return
    try:
        import torch._inductor.config as inductor_config

        inductor_config.compile_threads = 0
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        print(
            f"Disabled TorchInductor compile workers for distributed run "
            f"(world_size={world_size})."
        )
    except Exception:
        pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_function(examples: dict, tokenizer: AutoTokenizer, max_length: int):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


class CausalLMCollator:
    """Collator that sets labels and masks padding tokens with -100."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


class CheckpointSyncCallback(TrainerCallback):
    """Copy the latest checkpoint to a persistent directory after each save."""

    def __init__(self, sync_dir: str | Path | None):
        self.sync_dir = Path(sync_dir) if sync_dir else None

    def on_save(self, args, state, control, **kwargs):
        if not self.sync_dir or not state.is_world_process_zero:
            return control

        checkpoint_dirs = [
            d
            for d in Path(args.output_dir).iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]
        if not checkpoint_dirs:
            return control

        latest = max(checkpoint_dirs, key=lambda p: int(p.name.split("-")[1]))

        # Maintain a fixed-name copy for easy resume.
        latest_dest = self.sync_dir / "checkpoint-latest"
        if latest_dest.exists():
            shutil.rmtree(latest_dest)
        shutil.copytree(latest, latest_dest)

        print(f"Synced checkpoint {latest.name} to {self.sync_dir}")
        return control


class TrainingHistoryCallback(TrainerCallback):
    """Log training metrics to a JSONL file for post-hoc visualization."""

    def __init__(self, logger: TrainingLogger):
        self.logger = logger
        self._pending_eval_loss: float | None = None

    def on_log(self, args, state, control, logs: dict | None = None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return control

        metrics: dict[str, float] = {}
        if "loss" in logs:
            metrics["train_loss"] = float(logs["loss"])
        if "learning_rate" in logs:
            metrics["learning_rate"] = float(logs["learning_rate"])
        if "grad_norm" in logs:
            metrics["grad_norm"] = float(logs["grad_norm"])
        if self._pending_eval_loss is not None:
            metrics["eval_loss"] = self._pending_eval_loss
            metrics["eval_perplexity"] = math.exp(self._pending_eval_loss)
            self._pending_eval_loss = None

        if metrics:
            self.logger.log(state.global_step, metrics)
        return control

    def on_evaluate(self, args, state, control, metrics: dict | None = None, **kwargs):
        if metrics and "eval_loss" in metrics:
            self._pending_eval_loss = float(metrics["eval_loss"])
        return control


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a base LLM on preprocessed user data. "
        "Full fine-tune by default; pass --lora to use LoRA."
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to preprocessed JSONL.")
    parser.add_argument("--output-dir", type=Path, default=Path("models/finetuned"))
    parser.add_argument("--base-model", type=str, default="EleutherAI/gpt-neo-125M")
    parser.add_argument("--lora", action="store_true", help="Use LoRA instead of full fine-tuning.")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (default: 1).",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps.")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from.",
    )
    parser.add_argument(
        "--checkpoint-sync-dir",
        type=str,
        default=None,
        help="Directory to sync checkpoints to after each save.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep in the output directory (default: 2).",
    )
    parser.add_argument(
        "--fsdp",
        nargs="+",
        default=None,
        help="FSDP mode(s), e.g. 'full_shard auto_wrap' (for multi-GPU full fine-tuning).",
    )
    parser.add_argument(
        "--fsdp-config",
        type=Path,
        default=None,
        help="Path to an FSDP config JSON file.",
    )
    parser.add_argument(
        "--ddp-backend",
        choices=["nccl", "gloo", "mpi"],
        default=None,
        help="Distributed backend for multi-GPU training (default: nccl on CUDA). "
        "Use 'gloo' if NCCL hangs on older kernels.",
    )
    parser.add_argument(
        "--ddp-find-unused-parameters",
        action="store_true",
        help="Set find_unused_parameters=True for DDP (useful for LoRA workloads).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--skip-utility-eval",
        action="store_true",
        help="Skip post-training utility evaluation on test/holdout splits.",
    )
    parser.add_argument(
        "--no-load-best-at-end",
        action="store_true",
        help="Do not load the best checkpoint at the end of training. "
             "Useful for large distributed LoRA runs that OOM during finalization.",
    )
    args = parser.parse_args()

    # Disable TorchInductor compile workers early on multi-GPU runs. The forked
    # worker pool can deadlock if CUDA is initialized before the fork.
    _disable_inductor_for_distributed()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    model_cfg = ModelConfig(
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
    )
    train_cfg = TrainConfig(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
        device=args.device,
    )

    train_texts = load_split_texts(
        args.data,
        split="train",
        max_users=args.max_users,
    )
    val_texts = load_split_texts(
        args.data,
        split="val",
        max_users=args.max_users,
    )

    tokenizer = load_tokenizer(model_cfg.base_model, max_length=model_cfg.max_seq_length)
    print(f"Loading base model: {model_cfg.base_model} ...")
    model, _ = load_base_model(model_cfg.base_model, config=model_cfg, device=device)

    if args.lora:
        print(f"Adding LoRA (r={model_cfg.lora_rank}) ...")
        model = add_lora(model, config=model_cfg)
        model.print_trainable_parameters()
    else:
        print("Using full-parameter fine-tuning.")

    def tokenize(examples: dict) -> dict:
        return tokenize_function(examples, tokenizer, model_cfg.max_seq_length)

    train_dataset = HFDataset.from_dict({"text": train_texts}).map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )
    eval_dataset = None
    if val_texts:
        eval_dataset = HFDataset.from_dict({"text": val_texts}).map(
            tokenize,
            batched=True,
            remove_columns=["text"],
        )

    bf16 = device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = device.type == "cuda" and not bf16

    extra_args: dict = {}
    if args.fsdp:
        extra_args["fsdp"] = args.fsdp
    if args.fsdp_config:
        extra_args["fsdp_config"] = str(args.fsdp_config)
    if args.ddp_backend:
        extra_args["ddp_backend"] = args.ddp_backend
    if args.ddp_find_unused_parameters:
        extra_args["ddp_find_unused_parameters"] = True
    if _is_distributed_run():
        # torch.compile can also spawn Inductor workers that deadlock in
        # distributed runs; keep it off for multi-GPU training only.
        extra_args["torch_compile"] = False

    final_path = train_cfg.output_dir / "final"
    training_logger = TrainingLogger(final_path)

    print("\nBuilding TrainingArguments ...")
    training_args = TrainingArguments(
        output_dir=str(train_cfg.output_dir),
        num_train_epochs=train_cfg.num_epochs,
        per_device_train_batch_size=train_cfg.batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        warmup_steps=train_cfg.warmup_steps,
        logging_steps=train_cfg.logging_steps,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        load_best_model_at_end=(eval_dataset is not None and not args.no_load_best_at_end),
        metric_for_best_model=("eval_loss" if eval_dataset is not None and not args.no_load_best_at_end else None),
        greater_is_better=False,
        seed=train_cfg.seed,
        fp16=fp16,
        bf16=bf16,
        report_to="none",
        **extra_args,
    )

    callbacks = [TrainingHistoryCallback(training_logger)]
    if args.checkpoint_sync_dir:
        callbacks.append(CheckpointSyncCallback(args.checkpoint_sync_dir))

    data_collator = CausalLMCollator(tokenizer)

    print("Initializing Trainer (this may take a while with DDP/FSDP) ...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    print("Trainer initialized.\n")

    print("Starting training ...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nModel saved to {final_path}")

    is_main = (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )
    if is_main:
        plots_dir = final_path / "training_plots"
        generated = plot_training_curves(training_logger.log_path, plots_dir)
        if generated:
            print(f"Generated training plots in {plots_dir}")
        print(format_training_summary(training_logger.log_path))

    if not args.skip_utility_eval:
        is_main = (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
        if is_main:
            print("\nRunning post-training utility evaluation ...")

            def _eval_model(eval_model, label: str) -> dict[str, dict]:
                eval_model.eval()
                results: dict[str, dict] = {}
                print(f"\nEvaluating {label} model ...")
                for split in ("test", "holdout"):
                    texts = load_split_texts(
                        args.data, split=split, max_users=args.max_users
                    )
                    if not texts:
                        continue
                    results[split] = evaluate_texts(
                        eval_model,
                        tokenizer,
                        texts,
                        device,
                        max_length=args.max_seq_length,
                        batch_size=args.batch_size,
                    )
                if results:
                    print(format_utility_results(results))
                return results

            utility_results = _eval_model(model, "fine-tuned")

            if utility_results:
                # Free the fine-tuned model before loading the foundation model
                # so that both do not have to reside in GPU memory simultaneously.
                del model, trainer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                base_model_name = infer_base_model_name(
                    final_path, fallback=model_cfg.base_model
                )
                print(f"\nLoading foundation model: {base_model_name} ...")
                base_model, _ = load_base_model(
                    base_model_name, config=model_cfg, device=device
                )
                baseline_results = _eval_model(base_model, "foundation")

                metrics_out = {
                    "dataset_name": args.data.stem,
                    "base_model": model_cfg.base_model,
                    "max_seq_length": args.max_seq_length,
                    "splits": utility_results,
                    "baseline_splits": baseline_results,
                }
                save_utility_metrics(
                    final_path / "utility_metrics.json", metrics_out
                )
            else:
                print("No test/holdout splits found; skipping utility save.")


if __name__ == "__main__":
    main()
