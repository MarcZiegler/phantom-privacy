"""Tests for augment_pii_graph_with_email.py."""

from __future__ import annotations

import json
from pathlib import Path

import augment_pii_graph_with_email as aug


def _make_graph() -> dict:
    return {
        "schema": {},
        "nodes": [
            {"id": "USER:Alice", "type": "USER", "label": "Alice"},
            {"id": "NAME:Alice", "type": "NAME", "label": "Alice"},
        ],
        "edges": [
            {
                "source": "USER:Alice",
                "target": "NAME:Alice",
                "relation": "has_name",
                "score": 1.0,
            }
        ],
        "stats": {"n_nodes": 2, "n_edges": 1},
    }


def _make_profiles() -> dict:
    return {
        "users": {
            "Alice": {
                "ids": {
                    "email": "alice@example.com",
                    "phone": "555-123-4567",
                    "name": "Alice",
                }
            },
            "Bob": {
                "ids": {
                    "email": "bob@example.com",
                    "name": "Bob",
                }
            },
        }
    }


def test_augment_adds_email_nodes_and_edges() -> None:
    graph = _make_graph()
    profiles = _make_profiles()

    out = aug.augment_graph_with_emails(graph, profiles)

    email_node_ids = {
        n["id"] for n in out["nodes"] if n.get("type") == "EMAIL_ADDRESS"
    }
    assert "EMAIL_ADDRESS:alice@example.com" in email_node_ids

    email_edges = [
        e for e in out["edges"] if e.get("relation") == "has_email_address"
    ]
    assert len(email_edges) == 1
    assert email_edges[0]["source"] == "USER:Alice"
    assert email_edges[0]["target"] == "EMAIL_ADDRESS:alice@example.com"
    assert out["stats"]["email_nodes_added"] == 1
    assert out["stats"]["email_edges_added"] == 1


def test_augment_is_idempotent() -> None:
    graph = _make_graph()
    profiles = _make_profiles()

    first = aug.augment_graph_with_emails(graph, profiles)
    second = aug.augment_graph_with_emails(first, profiles)

    assert second["stats"]["email_nodes_added"] == 0
    assert second["stats"]["email_edges_added"] == 0
    assert len([n for n in second["nodes"] if n.get("type") == "EMAIL_ADDRESS"]) == 1
    assert len([e for e in second["edges"] if e.get("relation") == "has_email_address"]) == 1


def test_augment_ignores_missing_email() -> None:
    graph = {
        "schema": {},
        "nodes": [{"id": "USER:Charlie", "type": "USER", "label": "Charlie"}],
        "edges": [],
        "stats": {},
    }
    profiles = {
        "users": {
            "Charlie": {"ids": {"name": "Charlie"}},
        }
    }

    out = aug.augment_graph_with_emails(graph, profiles)

    assert out["stats"]["email_nodes_added"] == 0
    assert out["stats"]["email_edges_added"] == 0


def test_cli_writes_output(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    profiles_path = tmp_path / "profiles.json"
    output_path = tmp_path / "out.json"

    graph_path.write_text(json.dumps(_make_graph()))
    profiles_path.write_text(json.dumps(_make_profiles()))

    import sys
    argv_old = sys.argv
    try:
        sys.argv = [
            "augment_pii_graph_with_email.py",
            "--graph", str(graph_path),
            "--profiles", str(profiles_path),
            "--output", str(output_path),
        ]
        aug.main()
    finally:
        sys.argv = argv_old

    out = json.loads(output_path.read_text())
    assert out["stats"]["email_nodes_added"] == 1
    assert out["stats"]["email_edges_added"] == 1
