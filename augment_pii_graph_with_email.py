#!/usr/bin/env python3
"""Add email-address edges to a Mallinckrodt PII graph.

Reads a PII graph and a profile file (e.g. produced by a separate email-profile
extractor) and injects one ``has_email_address`` edge per user from the profile's
canonical email. Existing nodes and edges are preserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _user_node_id(user: str) -> str:
    return f"USER:{user}"


def _email_node_id(email: str) -> str:
    return f"EMAIL_ADDRESS:{email.lower()}"


def augment_graph_with_emails(
    graph: dict[str, Any],
    profiles: dict[str, Any],
) -> dict[str, Any]:
    """Return a new graph with email edges added from profiles.

    Args:
        graph: PII graph with ``nodes`` and ``edges`` lists.
        profiles: Profile dict with ``users`` -> ``ids`` -> ``email``.

    Returns:
        A new graph dict including ``EMAIL_ADDRESS`` nodes and
        ``has_email_address`` edges.
    """
    out_graph: dict[str, Any] = {
        "schema": graph.get("schema", {}),
        "nodes": list(graph.get("nodes", [])),
        "edges": list(graph.get("edges", [])),
        "stats": dict(graph.get("stats", {})),
    }

    existing_nodes: set[str] = {n.get("id", "") for n in out_graph["nodes"]}
    existing_edges: set[tuple[str, str, str]] = {
        (e.get("source", ""), e.get("target", ""), e.get("relation", ""))
        for e in out_graph["edges"]
    }

    users = profiles.get("users", {})
    added_nodes = 0
    added_edges = 0

    for user, data in users.items():
        email = data.get("ids", {}).get("email", "").strip().lower()
        if not email:
            continue

        user_node = _user_node_id(user)
        if user_node not in existing_nodes:
            # Skip users that do not already exist in the graph.
            continue

        email_node = _email_node_id(email)

        if email_node not in existing_nodes:
            out_graph["nodes"].append(
                {
                    "id": email_node,
                    "type": "EMAIL_ADDRESS",
                    "label": email,
                    "canonical_value": email,
                }
            )
            existing_nodes.add(email_node)
            added_nodes += 1

        edge_key = (user_node, email_node, "has_email_address")
        if edge_key not in existing_edges:
            out_graph["edges"].append(
                {
                    "source": user_node,
                    "target": email_node,
                    "relation": "has_email_address",
                    "score": 1.0,
                    "provenance": {
                        "source": "augment_pii_graph_with_email.py",
                        "profile_source": "mallinckrodt_xassoc_profiles.json",
                    },
                }
            )
            existing_edges.add(edge_key)
            added_edges += 1

    out_graph["stats"]["email_nodes_added"] = added_nodes
    out_graph["stats"]["email_edges_added"] = added_edges
    return out_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add email edges to a Mallinckrodt PII graph."
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("preprocessed/pii_graph_mallinckrodt.json"),
        help="Input PII graph JSON.",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("preprocessed/mallinckrodt_xassoc_profiles.json"),
        help="Profile JSON with canonical emails per user.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("preprocessed/pii_graph_mallinckrodt_with_email.json"),
        help="Output graph JSON.",
    )
    args = parser.parse_args()

    print(f"Loading graph from {args.graph} ...")
    graph = json.loads(args.graph.read_text(encoding="utf-8"))

    print(f"Loading profiles from {args.profiles} ...")
    profiles = json.loads(args.profiles.read_text(encoding="utf-8"))

    out_graph = augment_graph_with_emails(graph, profiles)

    args.output.write_text(json.dumps(out_graph, indent=2), encoding="utf-8")
    print(
        f"Wrote {args.output}: "
        f"added {out_graph['stats']['email_nodes_added']} email nodes and "
        f"{out_graph['stats']['email_edges_added']} email edges."
    )


if __name__ == "__main__":
    main()
