#!/usr/bin/env python3
"""Extract a user-level PII graph from a JSONL dataset.

The pipeline uses GLiNER for NER, GLiREL for relation extraction, and Stanza (or
a rule-based/null fallback) for coreference.  Output is a JSON graph with user
and PII-value nodes and (user, relation, value) edges.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from utils.data_utils import load_jsonl
from utils.pii_graph_utils import (
    GLiNERNER,
    GLiRELExtractor,
    NullCorefResolver,
    NullRelationExtractor,
    PIIGraph,
    PIIGraphBuilder,
    PIIGraphSchema,
    RuleCorefResolver,
    StanzaCorefResolver,
    UserLinker,
    _normalize_for_match,
    build_user_aliases,
    load_word_set,
)


def _maybe_cleanup_memory(step: int, every: int = 1000) -> None:
    """Run periodic GC and GPU cache clear to fight memory leaks."""
    if step == 0 or step % every != 0:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a user-level PII graph from email-style JSONL records.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the preprocessed JSONL dataset.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output graph JSON file.",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Path to a custom PII schema JSON (default: configs/pii_graph_schema.json).",
    )
    parser.add_argument(
        "--gliner-model",
        default="models/ner/gliner_small-v2.5",
        help="Local path to the GLiNER model directory.",
    )
    parser.add_argument(
        "--glirel-model",
        default="models/ner/glirel-large-v0",
        help="Local path to the GLiREL model directory.",
    )
    parser.add_argument(
        "--relation-extractor",
        choices=["glirel", "none"],
        default="none",
        help="Relation extractor backend (default: none; fallback heuristics only).",
    )
    parser.add_argument(
        "--stanza-dir",
        default="models/ner/stanza_resources",
        help="Local path to the Stanza resources directory.",
    )
    parser.add_argument(
        "--coref",
        choices=["stanza", "rule", "null"],
        default="stanza",
        help="Coreference resolver backend (default: stanza).",
    )
    parser.add_argument(
        "--disable-coref",
        action="store_true",
        help="Use the null coreference resolver (same as --coref null).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for GLiNER/GLiREL/Stanza (default: cpu).",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=0,
        help="Limit evaluation to N users (0 = unlimited).",
    )
    parser.add_argument(
        "--max-records-per-user",
        type=int,
        default=0,
        help="Limit each selected user to N records (0 = unlimited). Records are sampled randomly with a fixed seed.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Maximum characters to process per record text.",
    )
    parser.add_argument(
        "--entity-threshold",
        type=float,
        default=None,
        help="Override GLiNER entity threshold.",
    )
    parser.add_argument(
        "--relation-threshold",
        type=float,
        default=None,
        help="Override GLiREL relation threshold.",
    )
    parser.add_argument(
        "--require-ner-tail",
        action="store_true",
        default=True,
        help="Only keep GLiREL predictions whose tail overlaps a GLiNER entity.",
    )
    parser.add_argument(
        "--no-require-ner-tail",
        dest="require_ner_tail",
        action="store_false",
        help="Allow GLiREL predictions to create raw tail nodes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load models and process one record, then print stats without writing output.",
    )
    parser.add_argument(
        "--user-aliases",
        default=None,
        help="Optional JSON file mapping user IDs to alias lists.",
    )
    parser.add_argument(
        "--fallback-proximity-chars",
        type=int,
        default=300,
        help="Max char distance between a fallback PII mention and the nearest user-linked PERSON mention (default: 300).",
    )
    parser.add_argument(
        "--auto-aliases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically build user aliases from record metadata (default: True).",
    )
    parser.add_argument(
        "--common-words-list",
        default="configs/pii_graph_common_words.txt",
        help="Path to a newline-delimited list of common English words for ORGANIZATION filtering.",
    )
    parser.add_argument(
        "--job-titles-list",
        default="configs/pii_graph_job_titles.txt",
        help="Path to a newline-delimited list of valid job titles for JOB_TITLE filtering.",
    )
    return parser.parse_args()


def load_user_aliases(path: str | None) -> dict[str, list[str]]:
    """Load optional user alias dictionary."""
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("User aliases file must contain a JSON object.")
    return {str(k): [str(v) for v in vals] for k, vals in data.items()}


def main() -> None:
    args = parse_args()

    if args.disable_coref:
        args.coref = "null"

    schema = PIIGraphSchema(args.schema)

    print("Loading GLiNER model...", file=sys.stderr)
    ner = GLiNERNER(args.gliner_model, device=args.device)

    if args.relation_extractor == "glirel":
        print("Loading GLiREL model...", file=sys.stderr)
        relation_extractor = GLiRELExtractor(args.glirel_model, device=args.device)
    else:
        print("Using fallback heuristics only (no relation model).", file=sys.stderr)
        relation_extractor = NullRelationExtractor()

    if args.coref == "stanza":
        print("Loading Stanza coreference model...", file=sys.stderr)
        coref = StanzaCorefResolver(args.stanza_dir, device=args.device)
    elif args.coref == "rule":
        coref = RuleCorefResolver()
    else:
        coref = NullCorefResolver()

    print("Loading records...", file=sys.stderr)
    all_records = list(load_jsonl(Path(args.input)))

    # Build user aliases and known-user set for cross-user filtering.
    if args.user_aliases is not None:
        aliases = load_user_aliases(args.user_aliases)
    elif args.auto_aliases:
        aliases = {
            user: list(vals)
            for user, vals in build_user_aliases(all_records).items()
        }
    else:
        aliases = {}

    print("Loading reference word lists...", file=sys.stderr)
    common_words = load_word_set(args.common_words_list)
    job_titles_raw = load_word_set(args.job_titles_list)
    job_titles = {_normalize_for_match(t) for t in job_titles_raw}

    builder = PIIGraphBuilder(
        ner=ner,
        coref=coref,
        relation_extractor=relation_extractor,
        schema=schema,
        user_linker=UserLinker(aliases) if aliases else UserLinker(),
        require_ner_tail=args.require_ner_tail,
        known_users={
            str(rec.get("user") or rec.get("user_id") or rec.get("author"))
            for rec in all_records
            if rec.get("user") or rec.get("user_id") or rec.get("author")
        },
        fallback_proximity_chars=args.fallback_proximity_chars,
        common_words=common_words,
        job_titles=job_titles,
    )

    # Group records by user, then apply user- and record-level caps.
    user_records: dict[str, list[dict]] = defaultdict(list)
    for rec in all_records:
        user = rec.get("user") or rec.get("user_id") or rec.get("author")
        if user is None:
            continue
        user_records[user].append(rec)

    selected_users = list(user_records.keys())
    if args.max_users > 0:
        selected_users = selected_users[: args.max_users]

    rng = random.Random(42)
    records_to_process: list[dict] = []
    for user in selected_users:
        recs = user_records[user]
        if 0 < args.max_records_per_user < len(recs):
            recs = rng.sample(recs, args.max_records_per_user)
        records_to_process.extend(recs)

    graph = PIIGraph(schema=schema)

    iterator = records_to_process
    if not args.dry_run:
        iterator = tqdm(
            records_to_process,
            desc="Extracting PII graph",
            total=len(records_to_process),
        )

    for idx, rec in enumerate(iterator):
        text = rec.get("text", "")
        if args.max_chars > 0 and len(text) > args.max_chars:
            text = text[: args.max_chars]
        rec = {**rec, "text": text}
        graph = builder.process_record(rec, graph)
        _maybe_cleanup_memory(idx + 1)
        if args.dry_run:
            break

    if args.dry_run:
        print("Dry-run complete. Graph stats:", graph.to_dict()["stats"], file=sys.stderr)
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as fh:
        json.dump(graph.to_dict(), fh, indent=2)
    print(f"Wrote graph to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
