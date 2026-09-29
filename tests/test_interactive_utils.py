"""Tests for interactive preset selection."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from utils.interactive_utils import select_preset


def _make_mock_preset(model_id: str, dataset_name: str) -> MagicMock:
    preset = MagicMock()
    preset.model_id = model_id
    preset.dataset_name = dataset_name
    return preset


def test_select_preset_returns_selected_preset(monkeypatch) -> None:
    presets = {
        "p1": _make_mock_preset("model-a", "data-a"),
        "p2": _make_mock_preset("model-b", "data-b"),
    }
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "2")

    selected = select_preset(presets, "test")
    assert selected is presets["p2"]


def test_select_preset_loops_on_invalid_choice(monkeypatch) -> None:
    presets = {
        "p1": _make_mock_preset("model-a", "data-a"),
        "p2": _make_mock_preset("model-b", "data-b"),
    }
    inputs = iter(["invalid", "3", "1"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    selected = select_preset(presets, "test")
    assert selected is presets["p1"]


def test_select_preset_quit_on_q(monkeypatch) -> None:
    presets = {"p1": _make_mock_preset("model-a", "data-a")}
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "q")

    with pytest.raises(SystemExit) as exc_info:
        select_preset(presets, "test")
    assert exc_info.value.code == 0


def test_select_preset_exits_when_not_interactive(monkeypatch) -> None:
    presets = {"p1": _make_mock_preset("model-a", "data-a")}
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    with pytest.raises(SystemExit) as exc_info:
        select_preset(presets, "test")
    assert exc_info.value.code == 1
