"""Unit tests for extract_mallinckrodt_profiles.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import extract_mallinckrodt_profiles as extract


def test_email_candidates_from_name_handles_last_first_format() -> None:
    """Generate plausible email local parts from a Mallinckrodt display name."""
    candidates = extract._email_candidates_from_name("Muhlenkap, Kate M (Neely)")
    assert "kate.muhlenkap" in candidates
    assert "k.muhlenkap" in candidates
    assert "kate.neely" in candidates


def test_name_variants_from_display_name() -> None:
    """Name variants include first, last, and initials."""
    variants = extract._name_variants_from_display_name("Abbey, Michael A")
    assert "michael abbey" in variants
    assert "michael" in variants
    assert "abbey" in variants
    assert "m. abbey" in variants


def test_extract_text_emails_filters_imceaex() -> None:
    """Internal Exchange identifiers are ignored."""
    text = (
        "Contact kate.muhlenkap@covidien.com or "
        "IMCEAEX-_O=THCG_CN=KATE@tycohealthcare.com"
    )
    emails = extract._extract_text_emails(text, extract._DEFAULT_ALLOWED_DOMAINS)
    assert emails == ["kate.muhlenkap@covidien.com"]


def test_detect_user_emails_uses_from_header() -> None:
    """The authoritative email in the From header is preferred."""
    records = [
        {
            "user": "Muhlenkap, Kate",
            "text": "Hello world.",
            "from": '"Muhlenkap, Kate" <kate.muhlenkap@covidien.com>',
            "sender": "",
        },
        {
            "user": "Muhlenkap, Kate",
            "text": "Another email.",
            "from": '"Muhlenkap, Kate" <kate.muhlenkap@covidien.com>',
            "sender": "",
        },
    ]
    emails = extract._detect_user_emails(records, extract._DEFAULT_ALLOWED_DOMAINS)
    assert emails["Muhlenkap, Kate"] == "kate.muhlenkap@covidien.com"


def test_detect_user_phones_picks_most_common() -> None:
    """The most frequent 10-digit phone number per user is selected."""
    records = [
        {"user": "U1", "text": "Call 111-222-3333 or 111-222-3333."},
        {"user": "U1", "text": "Reach me at 444-555-6666."},
    ]
    phones = extract._detect_user_phones(records)
    assert phones["U1"] == "111-222-3333"


def test_build_graph_contains_expected_edges() -> None:
    """The derived graph links users to ID nodes."""
    profiles = {
        "users": {
            "U1": {
                "ids": {
                    "name": "Alice Smith",
                    "email": "alice@example.com",
                    "phone": "123-456-7890",
                    "job_title": "engineer",
                },
                "aliases": {"name": [], "email": [], "phone": []},
            }
        }
    }
    graph = extract._build_graph(profiles)
    relations = {e["relation"] for e in graph["edges"]}
    assert relations == {
        "has_name",
        "has_email_address",
        "has_phone_number",
        "has_job_title",
    }


def test_end_to_end_alias_extraction(tmp_path: Path) -> None:
    """Run the full script with a mocked GLiNER model."""
    input_path = tmp_path / "input.jsonl"
    graph_path = tmp_path / "graph.json"
    output_dir = tmp_path / "out"

    records = [
        {
            "user": "Abbey, Michael",
            "text": "Hello from Michael Abbey. Contact michael.abbey@mallinckrodt.com or 111-222-3333.",
            "from": '"Abbey, Michael" <michael.abbey@mallinckrodt.com>',
            "sender": "",
            "split": "train",
        },
        {
            "user": "Becker, Steven",
            "text": "Steven Becker here. steven.becker@covidien.com 444-555-6666.",
            "from": '"Becker, Steven" <steven.becker@covidien.com>',
            "sender": "",
            "split": "attacker",
        },
    ]
    input_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    graph_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "USER:Abbey, Michael", "type": "USER"},
                    {"id": "USER:Becker, Steven", "type": "USER"},
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )

    mock_model = MagicMock()
    # Return one PERSON span per text.
    mock_model.batch_predict_entities = MagicMock(
        side_effect=lambda texts, labels, threshold: [
            [{"text": "Michael Abbey", "label": "PERSON", "score": 0.99}]
            if "Michael" in t
            else [{"text": "Steven Becker", "label": "PERSON", "score": 0.99}]
            for t in texts
        ]
    )

    with patch("extract_mallinckrodt_profiles.GLiNER") as mock_gliner_cls:
        mock_gliner_cls.from_pretrained.return_value = mock_model
        import sys

        old_argv = sys.argv
        try:
            sys.argv = [
                "extract_mallinckrodt_profiles.py",
                "--input",
                str(input_path),
                "--pii-graph",
                str(graph_path),
                "--output-dir",
                str(output_dir),
                "--device",
                "cpu",
                "--seed",
                "7",
            ]
            extract.main()
        finally:
            sys.argv = old_argv

    profiles = json.loads((output_dir / "mallinckrodt_xassoc_profiles.json").read_text())
    assert "Abbey, Michael" in profiles["users"]
    assert profiles["users"]["Abbey, Michael"]["ids"]["email"] == "michael.abbey@mallinckrodt.com"
    assert profiles["users"]["Abbey, Michael"]["ids"]["phone"] == "111-222-3333"
    assert "Michael Abbey" in profiles["users"]["Abbey, Michael"]["aliases"]["name"]

    graph = json.loads((output_dir / "pii_graph_mallinckrodt_xassoc.json").read_text())
    assert any(e["relation"] == "has_email_address" for e in graph["edges"])
