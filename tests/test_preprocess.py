"""Unit tests for the unified preprocessing pipeline."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import preprocess


def test_resolve_user_identity_groups_by_user_field() -> None:
    records = [
        preprocess.TextRecord(user="alice", sender="alice", body="hello"),
        preprocess.TextRecord(user="alice", sender="alice", body="world"),
        preprocess.TextRecord(user="bob", sender="bob", body="hi"),
    ]
    grouped = preprocess.resolve_user_identity(records)
    assert set(grouped.keys()) == {"alice", "bob"}
    assert len(grouped["alice"]) == 2
    assert len(grouped["bob"]) == 1


def test_resolve_user_identity_drops_empty_users() -> None:
    records = [
        preprocess.TextRecord(user="alice", sender="alice", body="hello"),
        preprocess.TextRecord(user="", sender="", body="orphan"),
    ]
    grouped = preprocess.resolve_user_identity(records)
    assert set(grouped.keys()) == {"alice"}


def test_parse_idl_ocr_headers_extracts_fields_and_body() -> None:
    ocr = (
        "?Subject: Project Update  From: Alice Smith <alice@example.com>  "
        "Date: Mon, 01 Jan 2024 09:00:00 +0000  "
        "To: Bob Jones <bob@example.com>  "
        "Cc: Carol Lee <carol@example.com>      "
        "Hi Bob, here is the update."
    )
    parsed = preprocess._parse_idl_ocr_headers(ocr)
    assert parsed["subject"] == "Project Update"
    assert parsed["from"] == "Alice Smith <alice@example.com>"
    assert parsed["date"] == "Mon, 01 Jan 2024 09:00:00 +0000"
    assert parsed["to"] == "Bob Jones <bob@example.com>"
    assert parsed["cc"] == "Carol Lee <carol@example.com>"
    assert parsed["body"] == "Hi Bob, here is the update."


def test_parse_idl_ocr_headers_truncates_bloated_subject() -> None:
    ocr = (
        "?Subject: FW: Hydromorphone ER Orders  Kellen Goodwin  "
        "From: Cardetti, Lisa M  Sent: Tuesday, May 20, 2014 11:54 AM  "
        "To: Pharmaceuticals  "
        "Cc:      body text"
    )
    parsed = preprocess._parse_idl_ocr_headers(ocr)
    assert parsed["subject"] == "FW: Hydromorphone ER Orders  Kellen Goodwin"
    assert "Cardetti" not in parsed["subject"]


def test_clean_idl_subject_leaves_short_subjects_unchanged() -> None:
    assert preprocess._clean_idl_subject("Project Update") == "Project Update"
    assert preprocess._clean_idl_subject("RE: Hello") == "RE: Hello"


def test_clean_idl_subject_truncates_long_subjects_at_forward() -> None:
    raw = (
        "FW: Hydromorphone ER Orders  Kellen Goodwin  "
        "From: Cardetti, Lisa M  Sent: Tuesday, May 20, 2014  "
        "To: Pharmaceuticals  Cc: someone@example.com  "
        "This is extra body text that should not be in the subject. " * 3
    )
    cleaned = preprocess._clean_idl_subject(raw)
    assert cleaned == "FW: Hydromorphone ER Orders  Kellen Goodwin"
    assert "Cardetti" not in cleaned


def test_parse_idl_ocr_headers_handles_missing_cc() -> None:
    ocr = (
        "?Subject: Hello  From: Alice <alice@example.com>  "
        "Date: Mon, 01 Jan 2024 09:00:00 +0000  "
        "To: Bob <bob@example.com>      Body text"
    )
    parsed = preprocess._parse_idl_ocr_headers(ocr)
    assert parsed["subject"] == "Hello"
    assert "cc" not in parsed
    assert parsed["body"] == "Body text"


def test_extract_emails_from_text_deduplicates_and_lowercases() -> None:
    text = "Alice <Alice@Example.com>, Bob <bob@example.com>, Alice <alice@example.com>"
    emails = preprocess._extract_emails_from_text(text)
    assert emails == ["alice@example.com", "bob@example.com"]


def _make_idl_csv(path: Path, rows: list[dict]) -> None:
    """Write a small pipe-delimited IDL metadata CSV."""
    header = "|".join(
        [
            "id",
            "type",
            "author",
            "recipient",
            "copied",
            "subject",
            "document_date",
            "custodian",
            "ocr_text",
        ]
    )
    lines = [header]
    for row in rows:
        lines.append(
            "|".join(
                [
                    row.get("id", ""),
                    row.get("type", ""),
                    row.get("author", ""),
                    row.get("recipient", ""),
                    row.get("copied", ""),
                    row.get("subject", ""),
                    row.get("document_date", ""),
                    row.get("custodian", ""),
                    row.get("ocr_text", ""),
                ]
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def test_load_idl_records_loads_emails_and_uses_custodian() -> None:
    rows = [
        {
            "id": "doc1",
            "type": "Document, Email",
            "author": "Alice Smith <alice@example.com>",
            "recipient": "Bob <bob@example.com>",
            "copied": "",
            "subject": "",
            "document_date": "Mon, 01 Jan 2024 09:00:00 +0000",
            "custodian": "Alice Smith",
            "ocr_text": (
                "?Subject: Hello  From: Alice Smith <alice@example.com>  "
                "Date: Mon, 01 Jan 2024 09:00:00 +0000  "
                "To: Bob <bob@example.com>      Hi Bob"
            ),
        },
        {
            "id": "doc2",
            "type": "Document, Email",
            "author": "Alice Smith <alice@example.com>",
            "recipient": "Charlie <charlie@example.com>",
            "copied": "",
            "subject": "",
            "document_date": "Tue, 02 Jan 2024 10:00:00 +0000",
            "custodian": "Alice Smith",
            "ocr_text": (
                "?Subject: Re: Hello  From: Alice Smith <alice@example.com>  "
                "Date: Tue, 02 Jan 2024 10:00:00 +0000  "
                "To: Charlie <charlie@example.com>      Hi Charlie"
            ),
        },
        {
            "id": "doc3",
            "type": "Document",
            "author": "",
            "recipient": "",
            "copied": "",
            "subject": "",
            "document_date": "",
            "custodian": "Alice Smith",
            "ocr_text": "This is a memo, not an email.",
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "Mallinckrodt_Litigation_Documents.csv"
        _make_idl_csv(csv_path, rows)
        records = preprocess.load_idl_records(
            Path(tmpdir), min_examples=1, min_body_words=1
        )

    assert len(records) == 2
    assert all(r.user == "Alice Smith" for r in records)
    assert {r.to[0] for r in records} == {"bob@example.com", "charlie@example.com"}
    assert all(r.sender == "Alice Smith <alice@example.com>" for r in records)


def test_load_idl_records_respects_max_users_and_max_examples() -> None:
    rows = []
    for i in range(5):
        for j in range(3):
            rows.append(
                {
                    "id": f"doc{i}_{j}",
                    "type": "Document, Email",
                    "author": f"user{i}@example.com",
                    "recipient": "other@example.com",
                    "copied": "",
                    "subject": "",
                    "document_date": "",
                    "custodian": f"User {i}",
                    "ocr_text": (
                        f"?Subject: s{j}  From: user{i}@example.com  "
                        "Date: Mon, 01 Jan 2024 09:00:00 +0000  "
                        "To: other@example.com      body"
                    ),
                }
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "Mallinckrodt_Litigation_Documents.csv"
        _make_idl_csv(csv_path, rows)
        records = preprocess.load_idl_records(
            Path(tmpdir),
            min_examples=1,
            min_body_words=1,
            max_users=2,
            max_examples_per_user=2,
        )

    assert len(records) <= 4
    users = {r.user for r in records}
    assert len(users) <= 2
    for user in users:
        assert sum(1 for r in records if r.user == user) <= 2


def test_match_sender_to_custodian_requires_first_and_last_name() -> None:
    custodians = {"Becker, Steven", "Becker, Kevin", "Cardetti, Lisa"}
    assert preprocess._match_sender_to_custodian(
        '"Becker, Steven A" <steven.becker@covidien.com>', custodians
    ) == "Becker, Steven"
    assert preprocess._match_sender_to_custodian(
        '"Becker, Kevin" <kevin.becker@covidien.com>', custodians
    ) == "Becker, Kevin"
    assert (
        preprocess._match_sender_to_custodian(
            '"Becker, Mark" <mark.becker@covidien.com>', custodians
        )
        is None
    )
    assert preprocess._match_sender_to_custodian("lisa.cardetti@mallinckrodt.com", custodians) == "Cardetti, Lisa"
    assert preprocess._match_sender_to_custodian("unknown@example.com", custodians) is None


def test_load_idl_records_cross_mailbox_mode_assigns_by_sender() -> None:
    # Two custodians.  Alice's mailbox contains an email she sent; Bob's mailbox
    # contains the same email (as a recipient copy).  In idl-kandpal mode the
    # email should be assigned to Alice based on the author, not to Bob based on
    # the mailbox owner.
    rows = [
        {
            "id": "doc1",
            "type": "Document, Email",
            "author": '"Smith, Alice" <alice.smith@example.com>',
            "recipient": '"Doe, Bob" <bob.doe@example.com>',
            "copied": "",
            "subject": "hello",
            "document_date": "",
            "custodian": "Smith, Alice",
            "ocr_text": (
                "From: alice.smith@example.com  To: bob.doe@example.com  "
                "Subject: hello      body one"
            ),
        },
        {
            "id": "doc2",
            "type": "Document, Email",
            "author": '"Smith, Alice" <alice.smith@example.com>',
            "recipient": '"Doe, Bob" <bob.doe@example.com>',
            "copied": "",
            "subject": "hello",
            "document_date": "",
            "custodian": "Doe, Bob",
            "ocr_text": (
                "From: alice.smith@example.com  To: bob.doe@example.com  "
                "Subject: hello      body one"
            ),
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "Mallinckrodt_Litigation_Documents.csv"
        _make_idl_csv(csv_path, rows)
        records = preprocess.load_idl_records(
            Path(tmpdir),
            min_examples=1,
            min_body_words=1,
            user_resolver="idl-kandpal",
        )

    assert len(records) == 2
    assert all(r.user == "Smith, Alice" for r in records)


def test_build_config_mallinckrodt_preset() -> None:
    old_argv = sys.argv
    try:
        sys.argv = ["preprocess.py", "--preset", "mallinckrodt"]
        args = preprocess.parse_args()
        cfg = preprocess.build_config(args)
    finally:
        sys.argv = old_argv

    assert cfg["source"] == "idl"
    assert cfg["user_resolver"] == "idl-kandpal"
    assert cfg["text_format"] == "headers_body"
    assert cfg["min_examples"] == 100
    assert cfg["data_dir"] == "datasets/Mallinckrodt_Litigation_Documents"


def test_default_output_for_mallinckrodt_preset() -> None:
    assert preprocess._default_output_for_preset("mallinckrodt") == Path(
        "preprocessed/mallinckrodt_unsplit.jsonl"
    )


def test_record_to_dict_includes_from_field() -> None:
    record = preprocess.TextRecord(
        sender="alice@example.com",
        body="hello",
        subject="hi",
    )
    row = preprocess.record_to_dict(record)
    assert row["sender"] == "alice@example.com"
    assert row["from"] == "alice@example.com"
    # The text field should not include the sender by default (body_only here).
    assert "alice@example.com" not in row["text"]
