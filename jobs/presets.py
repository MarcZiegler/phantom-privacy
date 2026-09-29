from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import TrainConfig, model_name_to_dir  # noqa: E402
from utils.env_config import get_log_dir  # noqa: E402


@dataclass(frozen=True)
class SlurmConfig:
    """Slurm resource and notification configuration."""

    partition: str
    time: str
    nodes: int = 1
    ntasks_per_node: int = 8
    mem_per_cpu: int | None = None
    mem: str | None = None
    cpus_per_task: int | None = None
    gpus_per_node: int | None = None
    constraint: str | None = None
    account: str | None = None
    mail_type: str = "ALL"
    mail_user: str | None = None
    hint: str | None = "nomultithread"


@dataclass(frozen=True)
class AttackJobConfig:
    """Attack-specific configuration bundled with a fine-tuning preset."""

    max_users: int | None = None
    max_samples: int = 100
    strip_content_type_prefix: bool = False
    max_words: int | None = None
    device: str = "auto"
    pii_weight: float = 1.0
    pii_only: bool = False
    pii_source: str = "auto"
    num_distractors: int = 20
    pii_types: tuple[str, ...] = ("email", "phone")
    prompt_set: str = "D"
    name_map: Path | None = None
    context_mode: str = "any"
    mask_token: str = ""
    max_context_tokens: int = 256
    pii_graph_path: Path | None = None
    exposure_split: str = "train"
    exposure_from_held_in_only: bool = False
    exposure_from_target_only: bool = False
    distractor_mode: str = "real"
    email_local_only: bool = False
    mask_pii_types_in_prefix: tuple[str, ...] | None = None
    mask_placeholder: str = "<MASK>"
    max_new_tokens: int = 64
    bert_model_path: Path | None = None
    slurm: SlurmConfig | None = None


@dataclass(frozen=True)
class AutocompleteJobConfig:
    """Autocomplete PII extraction attack configuration."""

    max_users: int | None = None
    max_samples: int = 20
    pii_types: tuple[str, ...] = ("phone", "job_title")
    pii_graph_path: Path | None = None
    max_new_tokens: int = 15
    num_beams: int = 1
    do_sample: bool = False
    prompt_sets: tuple[str, ...] = ("target", "cross", "synthetic")
    device: str = "auto"
    email_local_only: bool = False
    mask_pii_types_in_prefix: tuple[str, ...] | None = None
    mask_placeholder: str = "<MASK>"
    slurm: SlurmConfig | None = None


@dataclass(frozen=True)
class PIIGraphJobConfig:
    """PII-graph extraction configuration bundled with a fine-tuning preset."""

    max_users: int | None = None
    max_records_per_user: int | None = None
    max_chars: int = 2000
    coref: str = "stanza"
    device: str = "auto"
    relation_extractor: str = "none"
    entity_threshold: float | None = None
    relation_threshold: float | None = None
    require_ner_tail: bool = True
    fallback_proximity_chars: int = 300
    auto_aliases: bool = True
    common_words_list: str = "configs/pii_graph_common_words.txt"
    job_titles_list: str = "configs/pii_graph_job_titles.txt"
    slurm: SlurmConfig | None = None


@dataclass(frozen=True)
class FinetunePreset:
    """A complete fine-tuning configuration: model, data, training, and Slurm."""

    name: str
    model_id: str
    dataset_name: str
    slurm: SlurmConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    max_seq_length: int = 512
    attack_max_seq_length: int | None = None
    use_torchrun: bool = False
    nproc_per_node: int = 1
    use_fsdp: bool = False
    fsdp_config: Path | None = None
    save_total_limit: int = 2
    extra_train_args: tuple[str, ...] = ()
    extra_env_vars: dict[str, str] = field(default_factory=dict)
    attack: AttackJobConfig = field(default_factory=AttackJobConfig)
    autocomplete: AutocompleteJobConfig | None = None
    pii_graph: PIIGraphJobConfig = field(default_factory=PIIGraphJobConfig)
    finetuned_dir_name_override: str | None = None

    @property
    def finetuned_dir_name(self) -> str:
        """Persistent directory name under models/ for this preset."""
        if self.finetuned_dir_name_override is not None:
            return self.finetuned_dir_name_override
        return f"finetuned-{self.name}"

    @property
    def model_dir_name(self) -> str:
        """Local base model directory name derived from the HuggingFace ID."""
        return model_name_to_dir(self.model_id)

    @property
    def dataset_file(self) -> str:
        """Full dataset filename in preprocessed/."""
        return f"{self.dataset_name}.jsonl"

    @property
    def log_file(self) -> Path:
        """Preset-specific log file used to detect duplicate submissions."""
        return get_log_dir() / f"finetune-{self.name}.log"

    def attack_log_file(self, attack_tag: str = "default") -> Path:
        """Preset- and attack-tag-specific log file for attack jobs."""
        return get_log_dir() / f"attack-{self.name}-{attack_tag}.log"

    def pii_graph_log_file(self, pii_graph_tag: str = "default") -> Path:
        """Preset- and tag-specific log file for PII-graph extraction jobs."""
        return get_log_dir() / f"pii-graph-{self.name}-{pii_graph_tag}.log"

    def autocomplete_log_file(self, autocomplete_tag: str = "default") -> Path:
        """Preset- and tag-specific log file for autocomplete PII extraction jobs."""
        return get_log_dir() / f"autocomplete-{self.name}-{autocomplete_tag}.log"


FINETUNE_PRESETS: dict[str, FinetunePreset] = {

    "neo125-enron-strict-lora": FinetunePreset(
        name="neo125-enron-strict-lora",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="enron",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
    ),
    "neo1300-enron-strict-lora-test": FinetunePreset(
        name="neo1300-enron-strict-lora-test",
        model_id="EleutherAI/gpt-neo-1.3B",
        dataset_name="enron",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-02:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=1,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
    ),
    "neo1300-enron-strict-lora": FinetunePreset(
        name="neo1300-enron-strict-lora",
        model_id="EleutherAI/gpt-neo-1.3B",
        dataset_name="enron",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="2-00:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("name", "phone", "job_title", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=True,
            distractor_mode="synthetic",
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-16:00:00",
                mem_per_cpu=3900,
            ),
        ),
        autocomplete=AutocompleteJobConfig(
            max_samples=20,
            pii_types=("name", "phone", "job_title", "email"),
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            max_new_tokens=15,
            num_beams=1,
            do_sample=False,
            prompt_sets=("target", "cross", "synthetic"),
            device="auto",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-12:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo1300-mallinckrodt-lora": FinetunePreset(
        name="neo1300-mallinckrodt-lora",
        model_id="EleutherAI/gpt-neo-1.3B",
        dataset_name="mallinckrodt",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="2-00:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("phone", "job_title", "name", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=True,
            distractor_mode="synthetic",
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-16:00:00",
                mem_per_cpu=3900,
            ),
        ),
        autocomplete=AutocompleteJobConfig(
            max_samples=20,
            pii_types=("phone", "job_title", "name"),
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            max_new_tokens=15,
            num_beams=1,
            do_sample=False,
            prompt_sets=("target", "cross", "synthetic"),
            device="auto",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-12:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo1300-enron-strict-lora-pii-autocomplete-masked": FinetunePreset(
        name="neo1300-enron-strict-lora-pii-autocomplete-masked",
        model_id="EleutherAI/gpt-neo-1.3B",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-neo1300-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        autocomplete=AutocompleteJobConfig(
            max_samples=20,
            pii_types=("name", "phone", "job_title", "email"),
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            max_new_tokens=15,
            num_beams=1,
            do_sample=False,
            prompt_sets=("target", "cross", "synthetic"),
            device="auto",
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-12:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo1300-mallinckrodt-lora-pii-autocomplete-masked": FinetunePreset(
        name="neo1300-mallinckrodt-lora-pii-autocomplete-masked",
        model_id="EleutherAI/gpt-neo-1.3B",
        dataset_name="mallinckrodt",
        finetuned_dir_name_override="finetuned-neo1300-mallinckrodt-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        autocomplete=AutocompleteJobConfig(
            max_samples=20,
            pii_types=("phone", "job_title", "name"),
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            max_new_tokens=15,
            num_beams=1,
            do_sample=False,
            prompt_sets=("target", "cross", "synthetic"),
            device="auto",
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-12:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo1300-enron-strict-lora-pii-autocomplete": FinetunePreset(
        name="neo1300-enron-strict-lora-pii-autocomplete",
        model_id="EleutherAI/gpt-neo-1.3B",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-neo1300-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        autocomplete=AutocompleteJobConfig(
            max_samples=20,
            pii_types=("name", "phone", "job_title", "email"),
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            max_new_tokens=15,
            num_beams=1,
            do_sample=False,
            prompt_sets=("target", "cross", "synthetic"),
            device="auto",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-12:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo1300-mallinckrodt-lora-pii-autocomplete": FinetunePreset(
        name="neo1300-mallinckrodt-lora-pii-autocomplete",
        model_id="EleutherAI/gpt-neo-1.3B",
        dataset_name="mallinckrodt",
        finetuned_dir_name_override="finetuned-neo1300-mallinckrodt-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        autocomplete=AutocompleteJobConfig(
            max_samples=20,
            pii_types=("phone", "job_title", "name"),
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            max_new_tokens=15,
            num_beams=1,
            do_sample=False,
            prompt_sets=("target", "cross", "synthetic"),
            device="auto",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-12:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo1300-enron-strict-lora-pii-held-out-records-any": FinetunePreset(
        name="neo1300-enron-strict-lora-pii-held-out-records-any",
        model_id="EleutherAI/gpt-neo-1.3B",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-neo1300-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("name", "phone", "job_title", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=False,
            exposure_from_target_only=True,
            distractor_mode="synthetic",
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-16:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo1300-mallinckrodt-pii-graph-attack-held-out-records": FinetunePreset(
        name="neo1300-mallinckrodt-pii-graph-attack-held-out-records",
        model_id="EleutherAI/gpt-neo-1.3B",
        dataset_name="mallinckrodt",
        finetuned_dir_name_override="finetuned-neo1300-mallinckrodt-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("phone", "job_title", "name", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=False,
            exposure_from_target_only=True,
            distractor_mode="synthetic",
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-16:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo1300-enron-strict-lora-pii-limited-masked": FinetunePreset(
        name="neo1300-enron-strict-lora-pii-limited-masked",
        model_id="EleutherAI/gpt-neo-1.3B",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-neo1300-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("name", "phone", "job_title", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=True,
            distractor_mode="synthetic",
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-16:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo1300-enron-strict-lora-pii-limited-masked-real": FinetunePreset(
        name="neo1300-enron-strict-lora-pii-limited-masked-real",
        model_id="EleutherAI/gpt-neo-1.3B",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-neo1300-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("name", "phone", "job_title", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=True,
            distractor_mode="real",
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-16:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo1300-enron-strict-lora-pii-held-out-records-any-real": FinetunePreset(
        name="neo1300-enron-strict-lora-pii-held-out-records-any-real",
        model_id="EleutherAI/gpt-neo-1.3B",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-neo1300-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("name", "phone", "job_title", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=False,
            exposure_from_target_only=True,
            distractor_mode="real",
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-16:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo1300-mallinckrodt-pii-graph-attack-limited-masked": FinetunePreset(
        name="neo1300-mallinckrodt-pii-graph-attack-limited-masked",
        model_id="EleutherAI/gpt-neo-1.3B",
        dataset_name="mallinckrodt",
        finetuned_dir_name_override="finetuned-neo1300-mallinckrodt-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("phone", "job_title", "name", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=True,
            distractor_mode="synthetic",
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-16:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo1300-mallinckrodt-pii-graph-attack-limited-masked-real": FinetunePreset(
        name="neo1300-mallinckrodt-pii-graph-attack-limited-masked-real",
        model_id="EleutherAI/gpt-neo-1.3B",
        dataset_name="mallinckrodt",
        finetuned_dir_name_override="finetuned-neo1300-mallinckrodt-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("phone", "job_title", "name", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=True,
            distractor_mode="real",
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-16:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo1300-mallinckrodt-pii-graph-attack-held-out-records-real": FinetunePreset(
        name="neo1300-mallinckrodt-pii-graph-attack-held-out-records-real",
        model_id="EleutherAI/gpt-neo-1.3B",
        dataset_name="mallinckrodt",
        finetuned_dir_name_override="finetuned-neo1300-mallinckrodt-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("phone", "job_title", "name", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=False,
            exposure_from_target_only=True,
            distractor_mode="real",
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-16:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "llama8b-enron-strict-lora-test": FinetunePreset(
        name="llama8b-enron-strict-lora-test",
        model_id="NousResearch/Meta-Llama-3-8B",
        dataset_name="enron",
        slurm=SlurmConfig(
            partition="gpu-8xA100",
            time="0-12:00:00",
            nodes=1,
            ntasks_per_node=8,
            mem="256G",
            gpus_per_node=8,
        ),
        train=TrainConfig(
            num_epochs=1,
            batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            warmup_steps=200,
            logging_steps=1,
        ),
        max_seq_length=512,
        use_torchrun=True,
        nproc_per_node=2,
        use_fsdp=False,
        fsdp_config=Path("configs/fsdp_llama3_8b.json"),
        save_total_limit=1,
        extra_train_args=(
            "--lora",
            "--lora-rank", "32",
            "--lora-alpha", "64",
            "--max-users", "50",
            "--skip-utility-eval",
            "--ddp-find-unused-parameters",
        ),
        extra_env_vars={},
    ),
    "llama8b-enron-strict-lora": FinetunePreset(
        name="llama8b-enron-strict-lora",
        model_id="NousResearch/Meta-Llama-3-8B",
        dataset_name="enron",
        slurm=SlurmConfig(
            partition="gpu-8xA100",
            time="1-12:00:00",
            nodes=1,
            ntasks_per_node=8,
            mem="256G",
            gpus_per_node=8,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=5e-5,
            warmup_steps=200,
            logging_steps=10,
        ),
        max_seq_length=512,
        use_torchrun=True,
        nproc_per_node=8,
        use_fsdp=False,
        fsdp_config=Path("configs/fsdp_llama3_8b.json"),
        save_total_limit=2,
        extra_train_args=(
            "--lora",
            "--lora-rank", "32",
            "--lora-alpha", "64",
            "--ddp-find-unused-parameters",
            "--no-load-best-at-end",
        ),
        extra_env_vars={},
    ),
    "llama8b-mallinckrodt-lora": FinetunePreset(
        name="llama8b-mallinckrodt-lora",
        model_id="NousResearch/Meta-Llama-3-8B",
        dataset_name="mallinckrodt",
        slurm=SlurmConfig(
            partition="gpu-8xA100",
            time="1-12:00:00",
            nodes=1,
            ntasks_per_node=8,
            mem="256G",
            gpus_per_node=8,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=5e-5,
            warmup_steps=200,
            logging_steps=10,
        ),
        max_seq_length=512,
        use_torchrun=True,
        nproc_per_node=8,
        use_fsdp=False,
        fsdp_config=Path("configs/fsdp_llama3_8b.json"),
        save_total_limit=2,
        extra_train_args=(
            "--lora",
            "--lora-rank", "32",
            "--lora-alpha", "64",
            "--ddp-find-unused-parameters",
            "--no-load-best-at-end",
        ),
        extra_env_vars={},
    ),
    # --- Llama-3-8B Enron attack presets ---
    "llama8b-enron-strict-lora-pii-limited-masked": FinetunePreset(
        name="llama8b-enron-strict-lora-pii-limited-masked",
        model_id="NousResearch/Meta-Llama-3-8B",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-llama8b-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(num_epochs=10, batch_size=4, learning_rate=5e-5),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("name", "phone", "job_title", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=True,
            distractor_mode="synthetic",
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            device="cuda",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-08:00:00",
                mem_per_cpu=3900,
            ),
        ),
        autocomplete=None,
    ),
    "llama8b-enron-strict-lora-pii-limited-masked-real": FinetunePreset(
        name="llama8b-enron-strict-lora-pii-limited-masked-real",
        model_id="NousResearch/Meta-Llama-3-8B",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-llama8b-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(num_epochs=10, batch_size=4, learning_rate=5e-5),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("name", "phone", "job_title", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=True,
            distractor_mode="real",
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            device="cuda",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-08:00:00",
                mem_per_cpu=3900,
            ),
        ),
        autocomplete=None,
    ),
    "llama8b-enron-strict-lora-pii-held-out-records-any": FinetunePreset(
        name="llama8b-enron-strict-lora-pii-held-out-records-any",
        model_id="NousResearch/Meta-Llama-3-8B",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-llama8b-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(num_epochs=10, batch_size=4, learning_rate=5e-5),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("name", "phone", "job_title", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=False,
            exposure_from_target_only=True,
            distractor_mode="synthetic",
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            device="cuda",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-08:00:00",
                mem_per_cpu=3900,
            ),
        ),
        autocomplete=None,
    ),
    "llama8b-enron-strict-lora-pii-held-out-records-any-real": FinetunePreset(
        name="llama8b-enron-strict-lora-pii-held-out-records-any-real",
        model_id="NousResearch/Meta-Llama-3-8B",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-llama8b-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(num_epochs=10, batch_size=4, learning_rate=5e-5),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("name", "phone", "job_title", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=False,
            exposure_from_target_only=True,
            distractor_mode="real",
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            device="cuda",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-08:00:00",
                mem_per_cpu=3900,
            ),
        ),
        autocomplete=None,
    ),
    "llama8b-enron-strict-lora-pii-autocomplete": FinetunePreset(
        name="llama8b-enron-strict-lora-pii-autocomplete",
        model_id="NousResearch/Meta-Llama-3-8B",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-llama8b-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-12:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(num_epochs=10, batch_size=4, learning_rate=5e-5),
        autocomplete=AutocompleteJobConfig(
            max_samples=20,
            pii_types=("name", "phone", "job_title", "email"),
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            max_new_tokens=15,
            num_beams=1,
            do_sample=False,
            prompt_sets=("target", "cross", "synthetic"),
            device="cuda",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-08:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "llama8b-enron-strict-lora-pii-autocomplete-masked": FinetunePreset(
        name="llama8b-enron-strict-lora-pii-autocomplete-masked",
        model_id="NousResearch/Meta-Llama-3-8B",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-llama8b-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-12:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(num_epochs=10, batch_size=4, learning_rate=5e-5),
        autocomplete=AutocompleteJobConfig(
            max_samples=20,
            pii_types=("name", "phone", "job_title", "email"),
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            max_new_tokens=15,
            num_beams=1,
            do_sample=False,
            prompt_sets=("target", "cross", "synthetic"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            device="cuda",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-08:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "llama8b-mallinckrodt-lora-pii-limited-masked": FinetunePreset(
        name="llama8b-mallinckrodt-lora-pii-limited-masked",

        model_id="NousResearch/Meta-Llama-3-8B",
        dataset_name="mallinckrodt",
        finetuned_dir_name_override="finetuned-llama8b-mallinckrodt-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="1-00:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(num_epochs=10, batch_size=4, learning_rate=5e-5),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("phone", "job_title", "name", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=True,
            distractor_mode="synthetic",
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            device="cuda",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-12:00:00",
                mem_per_cpu=3900,
            ),
        ),
        autocomplete=None,
    ),
    "llama8b-mallinckrodt-lora-pii-limited-masked-real": FinetunePreset(
        name="llama8b-mallinckrodt-lora-pii-limited-masked-real",
        model_id="NousResearch/Meta-Llama-3-8B",
        dataset_name="mallinckrodt",
        finetuned_dir_name_override="finetuned-llama8b-mallinckrodt-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="1-00:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(num_epochs=10, batch_size=4, learning_rate=5e-5),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("phone", "job_title", "name", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=True,
            distractor_mode="real",
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            device="cuda",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-12:00:00",
                mem_per_cpu=3900,
            ),
        ),
        autocomplete=None,
    ),
    "llama8b-mallinckrodt-lora-pii-held-out-records": FinetunePreset(
        name="llama8b-mallinckrodt-lora-pii-held-out-records",
        model_id="NousResearch/Meta-Llama-3-8B",
        dataset_name="mallinckrodt",
        finetuned_dir_name_override="finetuned-llama8b-mallinckrodt-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="1-00:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(num_epochs=10, batch_size=4, learning_rate=5e-5),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("phone", "job_title", "name", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=False,
            exposure_from_target_only=True,
            distractor_mode="synthetic",
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            device="cuda",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-12:00:00",
                mem_per_cpu=3900,
            ),
        ),
        autocomplete=None,
    ),
    "llama8b-mallinckrodt-lora-pii-held-out-records-real": FinetunePreset(
        name="llama8b-mallinckrodt-lora-pii-held-out-records-real",
        model_id="NousResearch/Meta-Llama-3-8B",
        dataset_name="mallinckrodt",
        finetuned_dir_name_override="finetuned-llama8b-mallinckrodt-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="1-00:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(num_epochs=10, batch_size=4, learning_rate=5e-5),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("phone", "job_title", "name", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=False,
            exposure_from_target_only=True,
            distractor_mode="real",
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            device="cuda",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-12:00:00",
                mem_per_cpu=3900,
            ),
        ),
        autocomplete=None,
    ),
    "llama8b-mallinckrodt-lora-pii-autocomplete": FinetunePreset(
        name="llama8b-mallinckrodt-lora-pii-autocomplete",
        model_id="NousResearch/Meta-Llama-3-8B",
        dataset_name="mallinckrodt",
        finetuned_dir_name_override="finetuned-llama8b-mallinckrodt-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-12:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(num_epochs=10, batch_size=4, learning_rate=5e-5),
        autocomplete=AutocompleteJobConfig(
            max_samples=20,
            pii_types=("phone", "job_title", "name"),
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            max_new_tokens=15,
            num_beams=1,
            do_sample=False,
            prompt_sets=("target", "cross", "synthetic"),
            device="cuda",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-08:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "llama8b-mallinckrodt-lora-pii-autocomplete-masked": FinetunePreset(
        name="llama8b-mallinckrodt-lora-pii-autocomplete-masked",
        model_id="NousResearch/Meta-Llama-3-8B",
        dataset_name="mallinckrodt",
        finetuned_dir_name_override="finetuned-llama8b-mallinckrodt-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-12:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(num_epochs=10, batch_size=4, learning_rate=5e-5),
        autocomplete=AutocompleteJobConfig(
            max_samples=20,
            pii_types=("phone", "job_title", "name"),
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            max_new_tokens=15,
            num_beams=1,
            do_sample=False,
            prompt_sets=("target", "cross", "synthetic"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            device="cuda",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-08:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-enron-strict-lora-pii-limited-masked": FinetunePreset(
        name="neo125-enron-strict-lora-pii-limited-masked",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-neo125-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("name", "phone", "job_title", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=True,
            distractor_mode="synthetic",
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-12:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-enron-strict-lora-pii-limited-masked-real": FinetunePreset(
        name="neo125-enron-strict-lora-pii-limited-masked-real",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-neo125-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("name", "phone", "job_title", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=True,
            distractor_mode="real",
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-12:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-enron-strict-lora-pii-limited-masked-structured": FinetunePreset(
        name="neo125-enron-strict-lora-pii-limited-masked-structured",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-neo125-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("name", "phone", "job_title", "email"),
            context_mode="structured",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=True,
            distractor_mode="synthetic",
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-12:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-enron-strict-lora-pii-limited-masked-header": FinetunePreset(
        name="neo125-enron-strict-lora-pii-limited-masked-header",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-neo125-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("name", "phone", "job_title", "email"),
            context_mode="header",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=True,
            distractor_mode="synthetic",
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-12:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-enron-strict-lora-pii-held-out-records-any": FinetunePreset(
        name="neo125-enron-strict-lora-pii-held-out-records-any",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-neo125-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("name", "phone", "job_title", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=False,
            exposure_from_target_only=True,
            distractor_mode="synthetic",
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            bert_model_path=Path("models/base/roberta-base"),
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-16:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-enron-strict-lora-pii-held-out-records-structured": FinetunePreset(
        name="neo125-enron-strict-lora-pii-held-out-records-structured",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-neo125-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("name", "phone", "job_title", "email"),
            context_mode="structured",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=False,
            exposure_from_target_only=True,
            distractor_mode="synthetic",
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            bert_model_path=Path("models/base/roberta-base"),
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-16:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-enron-strict-lora-pii-held-out-records-header": FinetunePreset(
        name="neo125-enron-strict-lora-pii-held-out-records-header",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-neo125-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("name", "phone", "job_title", "email"),
            context_mode="header",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=False,
            exposure_from_target_only=True,
            distractor_mode="synthetic",
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            bert_model_path=Path("models/base/roberta-base"),
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-16:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-enron-strict-lora-pii-held-out-records-any-real": FinetunePreset(
        name="neo125-enron-strict-lora-pii-held-out-records-any-real",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-neo125-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("name", "phone", "job_title", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=False,
            exposure_from_target_only=True,
            distractor_mode="real",
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            bert_model_path=Path("models/base/roberta-base"),
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-16:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-enron-strict-lora-pii-autocomplete": FinetunePreset(
        name="neo125-enron-strict-lora-pii-autocomplete",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-neo125-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        autocomplete=AutocompleteJobConfig(
            max_samples=20,
            pii_types=("name", "phone", "job_title"),
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            max_new_tokens=15,
            num_beams=1,
            do_sample=False,
            prompt_sets=("target", "cross", "synthetic"),
            device="auto",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-08:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-enron-strict-lora-pii-autocomplete-masked": FinetunePreset(
        name="neo125-enron-strict-lora-pii-autocomplete-masked",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="enron",
        finetuned_dir_name_override="finetuned-neo125-enron-strict-lora",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-16:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        autocomplete=AutocompleteJobConfig(
            max_samples=20,
            pii_types=("name", "phone", "job_title"),
            pii_graph_path=Path("preprocessed/pii_graph_enron_with_email.json"),
            max_new_tokens=15,
            num_beams=1,
            do_sample=False,
            prompt_sets=("target", "cross", "synthetic"),
            device="auto",
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-08:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-mallinckrodt": FinetunePreset(
        name="neo125-mallinckrodt",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="mallinckrodt",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-12:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
    ),
    "neo125-enron-strict-pii-graph": FinetunePreset(
        name="neo125-enron-strict-pii-graph",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="enron",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-01:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=1,
            batch_size=1,
        ),
        max_seq_length=512,
        pii_graph=PIIGraphJobConfig(
            max_users=None,
            max_records_per_user=None,
            coref="stanza",
            device="auto",
            relation_extractor="none",
            max_chars=1350,
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-12:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-mallinckrodt-pii-graph": FinetunePreset(
        name="neo125-mallinckrodt-pii-graph",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="mallinckrodt",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-01:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=1,
            batch_size=1,
        ),
        max_seq_length=512,
        pii_graph=PIIGraphJobConfig(
            max_users=None,
            max_records_per_user=None,
            coref="stanza",
            device="auto",
            relation_extractor="none",
            max_chars=1350,
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="1-00:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-mallinckrodt-pii-autocomplete": FinetunePreset(
        name="neo125-mallinckrodt-pii-autocomplete",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="mallinckrodt",
        finetuned_dir_name_override="finetuned-neo125-mallinckrodt",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-08:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        autocomplete=AutocompleteJobConfig(
            max_samples=20,
            pii_types=("phone", "job_title", "name"),
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            max_new_tokens=15,
            num_beams=1,
            do_sample=False,
            prompt_sets=("target", "cross", "synthetic"),
            device="auto",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-08:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-mallinckrodt-pii-graph-attack-limited-masked": FinetunePreset(
        name="neo125-mallinckrodt-pii-graph-attack-limited-masked",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="mallinckrodt",
        finetuned_dir_name_override="finetuned-neo125-mallinckrodt",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-12:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("phone", "job_title", "name"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=True,
            distractor_mode="synthetic",
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title"),
            mask_placeholder="<MASK>",
            bert_model_path=Path("models/base/roberta-base"),
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-12:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-mallinckrodt-pii-graph-attack-limited-masked-real": FinetunePreset(
        name="neo125-mallinckrodt-pii-graph-attack-limited-masked-real",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="mallinckrodt",
        finetuned_dir_name_override="finetuned-neo125-mallinckrodt",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-12:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("phone", "job_title", "name"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=True,
            distractor_mode="real",
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title"),
            mask_placeholder="<MASK>",
            bert_model_path=Path("models/base/roberta-base"),
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-12:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-mallinckrodt-pii-graph-attack-limited-masked-structured": FinetunePreset(
        name="neo125-mallinckrodt-pii-graph-attack-limited-masked-structured",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="mallinckrodt",
        finetuned_dir_name_override="finetuned-neo125-mallinckrodt",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-12:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("phone", "job_title", "name"),
            context_mode="structured",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=True,
            distractor_mode="synthetic",
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title"),
            mask_placeholder="<MASK>",
            bert_model_path=Path("models/base/roberta-base"),
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-12:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-mallinckrodt-pii-graph-attack-limited-masked-header": FinetunePreset(
        name="neo125-mallinckrodt-pii-graph-attack-limited-masked-header",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="mallinckrodt",
        finetuned_dir_name_override="finetuned-neo125-mallinckrodt",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-12:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("phone", "job_title", "name"),
            context_mode="header",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=True,
            distractor_mode="synthetic",
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title"),
            mask_placeholder="<MASK>",
            bert_model_path=Path("models/base/roberta-base"),
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-12:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-mallinckrodt-pii-autocomplete-masked": FinetunePreset(
        name="neo125-mallinckrodt-pii-autocomplete-masked",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="mallinckrodt",
        finetuned_dir_name_override="finetuned-neo125-mallinckrodt",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-08:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        autocomplete=AutocompleteJobConfig(
            max_samples=20,
            pii_types=("phone", "job_title", "name"),
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            max_new_tokens=15,
            num_beams=1,
            do_sample=False,
            prompt_sets=("target", "cross", "synthetic"),
            device="auto",
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-08:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    'neo125-mallinckrodt-pii-graph-attack-limited-masked-email': FinetunePreset(
        name='neo125-mallinckrodt-pii-graph-attack-limited-masked-email',
        model_id='EleutherAI/gpt-neo-125M',
        dataset_name='mallinckrodt',
        finetuned_dir_name_override='finetuned-neo125-mallinckrodt',
        slurm=SlurmConfig(
            partition='gpu-1xA100',
            time='0-12:00:00',
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=('--lora',),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=('phone', 'job_title', 'name', 'email'),
            context_mode='any',
            mask_token='',
            max_context_tokens=256,
            exposure_split='attacker',
            exposure_from_held_in_only=True,
            distractor_mode='synthetic',
            pii_graph_path=Path('preprocessed/pii_graph_mallinckrodt_with_email.json'),
            mask_pii_types_in_prefix=('name', 'phone', 'job_title', 'email'),
            mask_placeholder='<MASK>',
            bert_model_path=Path('models/base/roberta-base'),
            slurm=SlurmConfig(
                partition='gpu-1xA100',
                time='0-12:00:00',
                mem_per_cpu=3900,
            ),
        ),
    ),
    'neo125-mallinckrodt-pii-graph-attack-held-out-records': FinetunePreset(
        name='neo125-mallinckrodt-pii-graph-attack-held-out-records',
        model_id='EleutherAI/gpt-neo-125M',
        dataset_name='mallinckrodt',
        finetuned_dir_name_override='finetuned-neo125-mallinckrodt',
        slurm=SlurmConfig(
            partition='gpu-1xA100',
            time='0-12:00:00',
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=('--lora',),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=('phone', 'job_title', 'name', 'email'),
            context_mode='any',
            mask_token='',
            max_context_tokens=256,
            exposure_split='attacker',
            exposure_from_held_in_only=False,
            exposure_from_target_only=True,
            distractor_mode='synthetic',
            pii_graph_path=Path('preprocessed/pii_graph_mallinckrodt_with_email.json'),
            mask_pii_types_in_prefix=('name', 'phone', 'job_title', 'email'),
            mask_placeholder='<MASK>',
            bert_model_path=Path('models/base/roberta-base'),
            slurm=SlurmConfig(
                partition='gpu-1xA100',
                time='0-12:00:00',
                mem_per_cpu=3900,
            ),
        ),
    ),
    'neo125-mallinckrodt-pii-graph-attack-held-out-records-structured': FinetunePreset(
        name='neo125-mallinckrodt-pii-graph-attack-held-out-records-structured',
        model_id='EleutherAI/gpt-neo-125M',
        dataset_name='mallinckrodt',
        finetuned_dir_name_override='finetuned-neo125-mallinckrodt',
        slurm=SlurmConfig(
            partition='gpu-1xA100',
            time='0-12:00:00',
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=('--lora',),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=('phone', 'job_title', 'name', 'email'),
            context_mode='structured',
            mask_token='',
            max_context_tokens=256,
            exposure_split='attacker',
            exposure_from_held_in_only=False,
            exposure_from_target_only=True,
            distractor_mode='synthetic',
            pii_graph_path=Path('preprocessed/pii_graph_mallinckrodt_with_email.json'),
            mask_pii_types_in_prefix=('name', 'phone', 'job_title', 'email'),
            mask_placeholder='<MASK>',
            bert_model_path=Path('models/base/roberta-base'),
            slurm=SlurmConfig(
                partition='gpu-1xA100',
                time='0-12:00:00',
                mem_per_cpu=3900,
            ),
        ),
    ),
    'neo125-mallinckrodt-pii-graph-attack-held-out-records-header': FinetunePreset(
        name='neo125-mallinckrodt-pii-graph-attack-held-out-records-header',
        model_id='EleutherAI/gpt-neo-125M',
        dataset_name='mallinckrodt',
        finetuned_dir_name_override='finetuned-neo125-mallinckrodt',
        slurm=SlurmConfig(
            partition='gpu-1xA100',
            time='0-12:00:00',
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=('--lora',),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=('phone', 'job_title', 'name', 'email'),
            context_mode='header',
            mask_token='',
            max_context_tokens=256,
            exposure_split='attacker',
            exposure_from_held_in_only=False,
            exposure_from_target_only=True,
            distractor_mode='synthetic',
            pii_graph_path=Path('preprocessed/pii_graph_mallinckrodt_with_email.json'),
            mask_pii_types_in_prefix=('name', 'phone', 'job_title', 'email'),
            mask_placeholder='<MASK>',
            bert_model_path=Path('models/base/roberta-base'),
            slurm=SlurmConfig(
                partition='gpu-1xA100',
                time='0-12:00:00',
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-mallinckrodt-pii-graph-attack-held-out-records-real": FinetunePreset(
        name="neo125-mallinckrodt-pii-graph-attack-held-out-records-real",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="mallinckrodt",
        finetuned_dir_name_override="finetuned-neo125-mallinckrodt",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-12:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_samples=100,
            num_distractors=20,
            pii_types=("phone", "job_title", "name", "email"),
            context_mode="any",
            mask_token="",
            max_context_tokens=256,
            exposure_split="attacker",
            exposure_from_held_in_only=False,
            exposure_from_target_only=True,
            distractor_mode="real",
            pii_graph_path=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
            mask_pii_types_in_prefix=("name", "phone", "job_title", "email"),
            mask_placeholder="<MASK>",
            bert_model_path=Path("models/base/roberta-base"),
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-12:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-mallinckrodt-xassoc-train": FinetunePreset(
        name="neo125-mallinckrodt-xassoc-train",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="mallinckrodt_xassoc",
        max_seq_length=2048,
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="1-00:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
    ),
    "neo125-mallinckrodt-xassoc-attack": FinetunePreset(
        name="neo125-mallinckrodt-xassoc-attack",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="mallinckrodt_xassoc_attack",
        finetuned_dir_name_override="finetuned-neo125-mallinckrodt-xassoc-train",
        max_seq_length=2048,
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-12:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_new_tokens=96,
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-10:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-mallinckrodt-xassoc-synthetic-train": FinetunePreset(
        name="neo125-mallinckrodt-xassoc-synthetic-train",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="mallinckrodt_xassoc_synthetic",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-12:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
    ),
    "neo125-mallinckrodt-xassoc-synthetic-attack": FinetunePreset(
        name="neo125-mallinckrodt-xassoc-synthetic-attack",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="mallinckrodt_xassoc_synthetic_attack",
        finetuned_dir_name_override="finetuned-neo125-mallinckrodt-xassoc-synthetic-train",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-12:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            max_new_tokens=64,
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-04:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-mallinckrodt-recipient-qid": FinetunePreset(
        name="neo125-mallinckrodt-recipient-qid",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="mallinckrodt_recipient_qid",
        max_seq_length=1024,
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-12:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=10,
            batch_size=8,
            learning_rate=5e-5,
            warmup_steps=200,
        ),
        extra_train_args=("--lora",),
    ),
    "neo125-mallinckrodt-recipient-qid-attack": FinetunePreset(
        name="neo125-mallinckrodt-recipient-qid-attack",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="mallinckrodt_recipient_qid",
        finetuned_dir_name_override="finetuned-neo125-mallinckrodt-recipient-qid",
        max_seq_length=1024,
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-01:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=1,
            batch_size=1,
        ),
        extra_train_args=("--lora",),
        attack=AttackJobConfig(
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-08:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-enron-strict-pii-graph-test": FinetunePreset(
        name="neo125-enron-strict-pii-graph-test",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="enron",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-01:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=1,
            batch_size=1,
        ),
        max_seq_length=512,
        pii_graph=PIIGraphJobConfig(
            max_users=50,
            max_records_per_user=50,
            coref="stanza",
            device="auto",
            relation_extractor="none",
            max_chars=2000,
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-06:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
    "neo125-mallinckrodt-pii-graph-test": FinetunePreset(
        name="neo125-mallinckrodt-pii-graph-test",
        model_id="EleutherAI/gpt-neo-125M",
        dataset_name="mallinckrodt",
        slurm=SlurmConfig(
            partition="gpu-1xA100",
            time="0-01:00:00",
            mem_per_cpu=3900,
        ),
        train=TrainConfig(
            num_epochs=1,
            batch_size=1,
        ),
        max_seq_length=512,
        pii_graph=PIIGraphJobConfig(
            max_users=10,
            max_records_per_user=50,
            coref="stanza",
            device="auto",
            relation_extractor="none",
            max_chars=1350,
            slurm=SlurmConfig(
                partition="gpu-1xA100",
                time="0-06:00:00",
                mem_per_cpu=3900,
            ),
        ),
    ),
}
