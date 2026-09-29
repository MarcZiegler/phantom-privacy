"""Tests for utils/env_config.py."""

from __future__ import annotations

import pytest

from utils import env_config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove the variables under test before each test."""
    for name in (
        "SLURM_MAIL_USER",
        "SLURM_ACCOUNT",
        "SLURM_PARTITION",
        "PHANTOM_PRIVACY_PROJECT_DIR",
        "PHANTOM_PRIVACY_SCRATCH_ROOT",
        "PHANTOM_PRIVACY_SIF_PATH",
        "PHANTOM_PRIVACY_BASE_MODEL_CACHE",
        "PHANTOM_PRIVACY_LOG_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def test_is_placeholder_detects_missing_and_placeholder_values() -> None:
    assert env_config.is_placeholder(None)
    assert env_config.is_placeholder("")
    assert env_config.is_placeholder("   ")
    assert env_config.is_placeholder("your.email@example.com")
    assert env_config.is_placeholder("YOUR-SLURM-ACCOUNT")
    assert env_config.is_placeholder("<PLACEHOLDER>")
    assert not env_config.is_placeholder("marc25@ru.is")
    assert not env_config.is_placeholder("gpu-1xA100")


def test_validate_env_raises_on_missing_variables(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_PARTITION", "gpu-1xA100")
    # SLURM_MAIL_USER missing
    with pytest.raises(RuntimeError, match="SLURM_MAIL_USER"):
        env_config.validate_env()


def test_validate_env_raises_on_placeholder_values(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_MAIL_USER", "marc25@ru.is")
    monkeypatch.setenv("SLURM_PARTITION", "<PLACEHOLDER>")
    with pytest.raises(RuntimeError, match="SLURM_PARTITION"):
        env_config.validate_env()


def test_validate_env_returns_resolved_values(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_MAIL_USER", "marc25@ru.is")
    monkeypatch.setenv("SLURM_PARTITION", "gpu-1xA100")
    result = env_config.validate_env()
    assert result == {
        "SLURM_MAIL_USER": "marc25@ru.is",
        "SLURM_PARTITION": "gpu-1xA100",
    }


def test_get_account_returns_empty_when_unset_or_placeholder(monkeypatch) -> None:
    assert env_config.get_account() == ""
    monkeypatch.setenv("SLURM_ACCOUNT", "your-slurm-account")
    assert env_config.get_account() == ""


def test_get_account_returns_value_when_set(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_ACCOUNT", "test-account")
    assert env_config.get_account() == "test-account"


def test_get_required_env_returns_placeholder_when_missing() -> None:
    assert env_config.get_required_env("SLURM_PARTITION") == "<SLURM_PARTITION>"


def test_get_required_env_returns_value_when_set(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_PARTITION", "gpu-1xA100")
    assert env_config.get_required_env("SLURM_PARTITION") == "gpu-1xA100"


def test_optional_paths_default_to_project_root() -> None:
    # With no env set, optional paths fall back to the repository layout.
    expected_root = env_config.Path(__file__).resolve().parent.parent
    assert env_config.get_project_dir() == expected_root
    assert env_config.get_log_dir() == expected_root / "jobs" / "logs"
    assert env_config.get_sif_path() == expected_root / "images" / "priv_bench.sif"


def test_optional_paths_respect_env(monkeypatch) -> None:
    monkeypatch.setenv("PHANTOM_PRIVACY_PROJECT_DIR", "/opt/phantom")
    monkeypatch.setenv("PHANTOM_PRIVACY_LOG_DIR", "/var/log/phantom")
    assert env_config.get_project_dir() == env_config.Path("/opt/phantom")
    assert env_config.get_log_dir() == env_config.Path("/var/log/phantom")
