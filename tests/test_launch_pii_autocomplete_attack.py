"""Tests for jobs/launch_pii_autocomplete_attack.py."""

from __future__ import annotations

import pytest

from jobs.launch_pii_autocomplete_attack import render_slurm_script
from jobs.presets import FINETUNE_PRESETS, SlurmConfig

_DUMMY_SLURM = SlurmConfig(
    partition="gpu-test",
    time="0-01:00:00",
    mail_user="test@example.com",
    account="test-account",
)


@pytest.fixture
def autocomplete_preset():
    return FINETUNE_PRESETS["neo125-enron-strict-lora-pii-autocomplete"]


def test_render_includes_autocomplete_command(autocomplete_preset) -> None:
    script = render_slurm_script(
        autocomplete_preset,
        _DUMMY_SLURM,
        autocomplete_tag="default",
        max_users=None,
        max_samples=None,
        max_new_tokens=None,
        num_beams=None,
        do_sample=None,
        device=None,
        pii_types=None,
        prompt_sets=None,
        pii_graph_path=None,
    )
    assert "pii_autocomplete_attack.py" in script
    assert "--pii-types name phone job_title" in script
    assert "--prompt-sets target cross synthetic" in script
    assert "--model-path /models/finetuned/latest" in script


def test_render_override_prompt_sets(autocomplete_preset) -> None:
    script = render_slurm_script(
        autocomplete_preset,
        _DUMMY_SLURM,
        autocomplete_tag="target-only",
        max_users=None,
        max_samples=None,
        max_new_tokens=None,
        num_beams=None,
        do_sample=None,
        device=None,
        pii_types=None,
        prompt_sets=("target",),
        pii_graph_path=None,
    )
    assert "--prompt-sets target" in script
    assert "cross" not in script.split("prompt-sets")[1].split("\\")[0]


def test_render_includes_pii_graph_copy(autocomplete_preset) -> None:
    script = render_slurm_script(
        autocomplete_preset,
        _DUMMY_SLURM,
        autocomplete_tag="graph",
        max_users=None,
        max_samples=None,
        max_new_tokens=None,
        num_beams=None,
        do_sample=None,
        device=None,
        pii_types=None,
        prompt_sets=None,
        pii_graph_path=None,
    )
    assert "Copying PII graph to scratch..." in script
    assert "--pii-graph /data/pii_graph.json" in script


def test_render_raises_for_preset_without_autocomplete() -> None:
    preset = FINETUNE_PRESETS["neo125-enron-strict-lora"]
    with pytest.raises(ValueError, match="no autocomplete configuration"):
        render_slurm_script(
            preset,
            _DUMMY_SLURM,
            autocomplete_tag="default",
            max_users=None,
            max_samples=None,
            max_new_tokens=None,
            num_beams=None,
            do_sample=None,
            device=None,
            pii_types=None,
            prompt_sets=None,
            pii_graph_path=None,
        )
