"""Tests for jobs/launch_pii_extraction_attack.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobs.launch_pii_extraction_attack import render_slurm_script
from jobs.presets import FINETUNE_PRESETS, SlurmConfig

_DUMMY_SLURM = SlurmConfig(
    partition="gpu-test",
    time="0-01:00:00",
    mail_user="test@example.com",
    account="test-account",
)


@pytest.fixture
def masked_preset():
    return FINETUNE_PRESETS["neo125-mallinckrodt-pii-graph-attack-limited-masked"]


@pytest.fixture
def plain_preset():
    return FINETUNE_PRESETS["neo125-enron-strict-lora"]


def test_render_includes_preset_bert_model(masked_preset) -> None:
    script = render_slurm_script(
        masked_preset,
        _DUMMY_SLURM,
        attack_tag="bert-test",
        max_users=None,
        num_distractors=None,
        device=None,
        pii_types=None,
        max_samples=None,
        max_context_tokens=None,
        mask_token=None,
        context_mode=None,
        pii_graph_path=None,
    )
    assert "--bert-model /models/bert/roberta-base" in script
    assert 'cp -r "${PROJECT_DIR}/models/base/roberta-base"' in script
    assert "Copying BERT model to scratch..." in script


def test_render_override_bert_model_path(masked_preset) -> None:
    script = render_slurm_script(
        masked_preset,
        _DUMMY_SLURM,
        attack_tag="bert-override",
        max_users=None,
        num_distractors=None,
        device=None,
        pii_types=None,
        max_samples=None,
        max_context_tokens=None,
        mask_token=None,
        context_mode=None,
        pii_graph_path=None,
        bert_model_path=Path("models/base/roberta-large"),
    )
    assert "--bert-model /models/bert/roberta-large" in script
    assert 'cp -r "${PROJECT_DIR}/models/base/roberta-large"' in script


def test_render_omits_bert_when_not_configured(plain_preset) -> None:
    script = render_slurm_script(
        plain_preset,
        _DUMMY_SLURM,
        attack_tag="no-bert",
        max_users=None,
        num_distractors=None,
        device=None,
        pii_types=None,
        max_samples=None,
        max_context_tokens=None,
        mask_token=None,
        context_mode=None,
        pii_graph_path=None,
    )
    assert "--bert-model" not in script
    assert "Copying BERT model to scratch..." not in script
