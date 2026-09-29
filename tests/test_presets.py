"""Lightweight consistency checks for the preset registry."""

from __future__ import annotations

from jobs.presets import FINETUNE_PRESETS


def test_all_preset_keys_match_names() -> None:
    """Every preset dict key must equal the preset's declared name."""
    for key, preset in FINETUNE_PRESETS.items():
        assert key == preset.name, f"key {key!r} != name {preset.name!r}"


def test_all_presets_have_model_and_dataset() -> None:
    for key, preset in FINETUNE_PRESETS.items():
        assert preset.model_id and isinstance(preset.model_id, str), key
        assert preset.dataset_name and isinstance(preset.dataset_name, str), key


def test_attack_presets_have_required_fields() -> None:
    for key, preset in FINETUNE_PRESETS.items():
        if preset.attack is None:
            continue
        assert isinstance(preset.attack.pii_types, tuple), key
        assert preset.attack.distractor_mode in {"real", "synthetic"}, key
        assert preset.attack.context_mode in {"any", "header", "structured"}, key


def test_autocomplete_presets_have_required_fields() -> None:
    for key, preset in FINETUNE_PRESETS.items():
        if preset.autocomplete is None:
            continue
        assert isinstance(preset.autocomplete.pii_types, tuple), key
        assert isinstance(preset.autocomplete.prompt_sets, tuple), key
        assert all(ps in {"target", "cross", "synthetic"} for ps in preset.autocomplete.prompt_sets), key


def test_renamed_llama_presets_exist() -> None:
    """The Llama training-cross presets use the -limited-masked suffix."""
    assert "llama8b-enron-strict-lora-pii-limited-masked" in FINETUNE_PRESETS
    assert "llama8b-enron-strict-lora-pii-limited-masked-real" in FINETUNE_PRESETS
    assert "llama8b-mallinckrodt-lora-pii-limited-masked" in FINETUNE_PRESETS
    assert "llama8b-mallinckrodt-lora-pii-limited-masked-real" in FINETUNE_PRESETS
