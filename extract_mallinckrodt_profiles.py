#!/usr/bin/env python3
"""Extract per-user ID aliases from the Mallinckrodt corpus using NER + regex.

This script produces a lightweight profile file for the cross-association
augmentation experiment.  It detects:

- Names (via GLiNER NER and regex-derived variants)
- Emails (regex + ``From:``/``Sender:`` headers + name-pattern matching)
- Phones (regex)
- Job titles (existing PII graph when available)

Only values observed in the corpus or the supplied PII graph are emitted.

Outputs:
    preprocessed/mallinckrodt_xassoc_profiles.json
    preprocessed/pii_graph_mallinckrodt_xassoc.json
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from gliner import GLiNER
except ImportError as exc:  # pragma: no cover
    raise ImportError("gliner is required for alias extraction") from exc


_DEFAULT_ALLOWED_DOMAINS = {
    "mallinckrodt.com",
    "covidien.com",
    "mnk.com",
    "tycohealthcare.com",
    "tyco.com",
}

_EMAIL_RE = re.compile(r"[\w.-]+@[\w.-]+\.\w+")
_PHONE_RE = re.compile(r"\b\d{3}[-.)]?\s*\d{3}[-.\s]?\d{4}\b")


def _canonical_phone_digits(value: str) -> str:
    """Return only the digits from a phone number string."""
    return "".join(ch for ch in value if ch.isdigit())


def _format_phone(raw_digits: str) -> str:
    """Format 10-digit phone number as XXX-XXX-XXXX."""
    if len(raw_digits) == 10:
        return f"{raw_digits[:3]}-{raw_digits[3:6]}-{raw_digits[6:]}"
    return raw_digits


def _normalize_name(value: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    value = re.sub(r"[^\w\s]", " ", value.lower())
    return re.sub(r"\s+", " ", value).strip()


def _parse_display_name(display_name: str) -> dict[str, str]:
    """Parse a display name into name components.

    Supports ``Last, First Middle (extra)`` formats common in Mallinckrodt.
    """
    parts = re.split(r"[()]", display_name)
    main = parts[0]
    extras = [p.strip(", ") for p in parts[1:] if p.strip(", ")]

    tokens = [t.strip() for t in main.replace(",", " ").split() if t.strip()]
    if not tokens:
        return {"first": "", "middle": "", "last": "", "extras": extras}

    last = tokens[0].lower()
    first = ""
    middle = ""
    if len(tokens) > 1:
        first = tokens[1].lower()
    if len(tokens) > 2 and len(tokens[2]) == 1:
        middle = tokens[2].lower()

    return {"first": first, "middle": middle, "last": last, "extras": extras}


def _email_candidates_from_name(display_name: str) -> set[str]:
    """Generate likely email local parts from a display name."""
    parsed = _parse_display_name(display_name)
    first = parsed["first"]
    middle = parsed["middle"]
    last = parsed["last"]
    extras = parsed["extras"]

    candidates: set[str] = set()
    if first and last:
        candidates.add(f"{first}.{last}")
        candidates.add(f"{first[0]}.{last}")
        if middle:
            candidates.add(f"{first}.{middle}.{last}")
            candidates.add(f"{first[0]}{middle}.{last}")
    for extra in extras:
        extra_tokens = [t.lower() for t in extra.split() if t]
        if extra_tokens:
            candidates.add(f"{extra_tokens[0]}.{last}")
            if first:
                candidates.add(f"{first}.{extra_tokens[0]}")
    return candidates


def _name_variants_from_display_name(display_name: str) -> set[str]:
    """Return a set of name variants derived from the display name."""
    parsed = _parse_display_name(display_name)
    first = parsed["first"]
    middle = parsed["middle"]
    last = parsed["last"]
    extras = parsed["extras"]

    variants: set[str] = set()
    if first and last:
        variants.add(f"{first} {last}")
        variants.add(f"{first[0]}. {last}")
        variants.add(f"{first[0]}{middle}. {last}" if middle else f"{first[0]}. {last}")
        variants.add(f"{first}")
        variants.add(f"{last}")
        variants.add(f"{last}, {first}")
    if middle and first and last:
        variants.add(f"{first} {middle}. {last}")
    for extra in extras:
        extra_tokens = [t.lower() for t in extra.split() if t]
        if extra_tokens and first:
            variants.add(f"{first} {extra_tokens[0]}")
            variants.add(f"{extra_tokens[0]}")
    return {v for v in variants if v}


def _extract_text_emails(text: str, allowed_domains: set[str] | None = None) -> list[str]:
    """Extract clean emails from a text string."""
    text = html.unescape(text)
    found: list[str] = []
    for match in _EMAIL_RE.finditer(text):
        raw = match.group(0)
        start = max(0, match.start() - 100)
        context = text[start : match.end()].lower()
        if "imceaex" in context or "+2e" in context or "_o=" in context or "cn=" in context:
            continue
        lower = raw.lower()
        local, _, domain = lower.partition("@")
        if allowed_domains is not None and domain not in allowed_domains:
            continue
        found.append(lower)
    return found


def _extract_text_phones(text: str) -> list[str]:
    """Extract 10-digit phone numbers from a text string."""
    digits_list: list[str] = []
    for raw in _PHONE_RE.findall(text):
        digits = _canonical_phone_digits(raw)
        if len(digits) == 10:
            digits_list.append(digits)
    return digits_list


def _extract_graph_values(graph: dict[str, Any], relation: str) -> dict[str, set[str]]:
    """Return all graph values for ``relation`` grouped by user."""
    values: dict[str, set[str]] = defaultdict(set)
    labels = {n["id"]: n.get("label", "") for n in graph.get("nodes", [])}
    for edge in graph.get("edges", []):
        if edge.get("relation") != relation:
            continue
        source = edge.get("source", "")
        if not source.startswith("USER:"):
            continue
        user = source[5:]
        value = labels.get(edge.get("target", ""), "").strip()
        if value:
            values[user].add(value)
    return dict(values)


def _select_most_supported(
    records: list[dict[str, Any]],
    users: set[str],
    candidates: dict[str, set[str]],
) -> dict[str, str]:
    """For each user, count how often each candidate value appears and pick best."""
    user_counts: dict[str, Counter[str]] = {user: Counter() for user in users}
    candidate_lower: dict[str, set[str]] = {
        user: {_normalize_name(v) for v in vals} for user, vals in candidates.items()
    }
    for record in records:
        user = record.get("user")
        if user not in users:
            continue
        text = _normalize_name(record.get("text", ""))
        for value in candidate_lower.get(user, set()):
            user_counts[user][value] += text.count(value)

    result: dict[str, str] = {}
    for user, counts in user_counts.items():
        if counts:
            best = counts.most_common(1)[0][0]
            for orig in candidates.get(user, set()):
                if _normalize_name(orig) == best:
                    result[user] = orig
                    break
            else:
                result[user] = best
    return result


def _format_signature_name(display_name: str) -> str:
    """Convert ``Last, First Middle`` to ``First Middle Last``."""
    parsed = _parse_display_name(display_name)
    tokens = []
    if parsed["first"]:
        tokens.append(parsed["first"].capitalize())
    if parsed["middle"]:
        tokens.append(parsed["middle"].upper() + ".")
    if parsed["last"]:
        tokens.append(parsed["last"].capitalize())
    return " ".join(tokens) if tokens else display_name


def _load_job_titles(path: Path) -> list[str]:
    """Load a newline-delimited list of job titles."""
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _detect_user_emails(
    records: list[dict[str, Any]],
    allowed_domains: set[str],
) -> dict[str, str]:
    """Return the most likely email address per user."""
    user_candidates: dict[str, Counter[str]] = defaultdict(Counter)
    user_locals: dict[str, set[str]] = {}

    for record in records:
        user = record.get("user")
        if not user:
            continue
        if user not in user_locals:
            user_locals[user] = _email_candidates_from_name(user)

        text_parts = [
            record.get("text", ""),
            record.get("from", ""),
            record.get("sender", ""),
        ]
        for part in text_parts:
            for email in _extract_text_emails(part, allowed_domains):
                local, _, _ = email.partition("@")
                if user_locals[user] and local not in user_locals[user]:
                    continue
                user_candidates[user][email] += 1

    # Frequency fallback: allow any allowed-domain email that is not another user's
    # primary candidate if no name-pattern match was found.
    for record in records:
        user = record.get("user")
        if not user:
            continue
        if user_candidates.get(user):
            continue
        for part in [record.get("text", ""), record.get("from", ""), record.get("sender", "")]:
            for email in _extract_text_emails(part, allowed_domains):
                user_candidates[user][email] += 1

    # Exclude emails that are the most common address of another user.
    other_emails = {
        counts.most_common(1)[0][0] for counts in user_candidates.values() if counts
    }

    result: dict[str, str] = {}
    for user, counts in user_candidates.items():
        filtered = {e: c for e, c in counts.items() if e not in other_emails}
        if not filtered:
            filtered = dict(counts)
        if filtered:
            result[user] = max(filtered, key=lambda e: (filtered[e], -len(e)))
    return result


def _detect_user_phones(records: list[dict[str, Any]]) -> dict[str, str]:
    """Return the most common 10-digit phone number per user."""
    user_candidates: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        user = record.get("user")
        if not user:
            continue
        for digits in _extract_text_phones(record.get("text", "")):
            user_candidates[user][digits] += 1

    result: dict[str, str] = {}
    for user, counts in user_candidates.items():
        if counts:
            result[user] = _format_phone(counts.most_common(1)[0][0])
    return result


def _detect_user_job_titles(
    records: list[dict[str, Any]],
    users: set[str],
    graph: dict[str, Any],
    job_titles_list: list[str],
    seed: int,
) -> dict[str, str]:
    """Return a job title per user, using the graph when possible."""
    graph_values = _extract_graph_values(graph, "has_job_title")
    graph_title = _select_most_supported(records, users, graph_values)

    result: dict[str, str] = {}
    for user in users:
        if user in graph_title and graph_title[user]:
            result[user] = graph_title[user]
        else:
            result[user] = ""
    return result


def _run_ner(
    model: GLiNER,
    texts: list[str],
    batch_size: int,
    threshold: float,
) -> list[list[dict[str, Any]]]:
    """Run GLiNER PERSON detection over a list of texts in batches."""
    results: list[list[dict[str, Any]]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results.extend(model.batch_predict_entities(batch, ["PERSON"], threshold=threshold))
    return results


def _clean_alias(raw: str) -> str:
    """Normalize a raw alias: strip honorifics, remove parentheticals, collapse spaces."""
    # Remove parenthetical content.
    cleaned = re.sub(r"\s*\([^)]*\)", "", raw)
    # Strip common honorifics.
    cleaned = re.sub(r"^(mr\.?|mrs\.?|ms\.?|dr\.?)\s+", "", cleaned, flags=re.IGNORECASE)
    # Normalize whitespace.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_valid_name_alias(
    alias: str,
    user: str,
    other_last_names: set[str],
    other_first_names: set[str],
    require_last_name: bool,
) -> bool:
    """Return True if ``alias`` is a plausible name for ``user``.

    If ``require_last_name`` is True, the alias must contain the user's last
    name (or an initial+last pattern).  Otherwise first-name-only aliases are
    also accepted when the first name is not shared with another user.
    """
    parsed = _parse_display_name(user)
    first = parsed["first"]
    middle = parsed["middle"]
    last = parsed["last"]

    # Basic hygiene.
    if not alias or len(alias) < 2:
        return False
    if "@" in alias or ";" in alias or "\n" in alias:
        return False

    # Allow "Last, First" forms by converting commas to spaces.
    alias = alias.replace(",", " ")
    norm = _normalize_name(alias)
    tokens = norm.split()
    if not tokens or len(tokens) > 4:
        return False

    # Reject aliases that contain another user's last name unless they also
    # contain this user's last name.
    if any(t in other_last_names for t in tokens) and last not in tokens:
        return False

    # Accept if the alias contains the user's last name.
    if last and last in tokens:
        return True

    # Accept initial + last patterns, e.g. "M. Abbey" or "M Abbey".
    if last:
        initial = first[0] if first else ""
        initial_last_re = re.compile(
            r"^" + re.escape(initial) + r"\.?\s+" + re.escape(last) + r"$"
        )
        if initial_last_re.match(norm):
            return True

    # Strict mode stops here.
    if require_last_name:
        return False

    # Lenient mode: accept first-name-only or first+middle aliases if the first
    # name is not shared with another user.
    return (
        (first and tokens == [first] and first not in other_first_names)
        or (first and middle and tokens == [first, middle] and first not in other_first_names)
    )


def _derive_name_variants(
    aliases: set[str],
    user: str,
) -> set[str]:
    """Add normalized and casing variants of accepted aliases."""
    variants: set[str] = set()
    parsed = _parse_display_name(user)
    first = parsed["first"]
    last = parsed["last"]
    middle = parsed["middle"]

    for alias in aliases:
        variants.add(alias)
        variants.add(_normalize_name(alias))
        # Title-cased version.
        variants.add(alias.title())

    # Ensure canonical variants exist.
    if first and last:
        variants.add(f"{first} {last}")
        variants.add(f"{first.title()} {last.title()}")
        variants.add(f"{last}, {first}")
        variants.add(f"{last.title()}, {first.title()}")
    if first and middle and last:
        variants.add(f"{first} {middle}. {last}")
        variants.add(f"{first.title()} {middle.upper()}. {last.title()}")
    return {v for v in variants if v}


def _collect_name_aliases(
    records: list[dict[str, Any]],
    users: set[str],
    primary_emails: dict[str, str],
    primary_phones: dict[str, str],
    model: GLiNER,
    max_records_per_user: int,
    batch_size: int,
    threshold: float,
    max_chars: int,
) -> dict[str, set[str]]:
    """Collect and filter PERSON-span aliases for each user using GLiNER NER."""
    # Index records by user and build mention sets.
    user_records: dict[str, list[int]] = defaultdict(list)
    mention_records: dict[str, set[int]] = defaultdict(set)
    for idx, record in enumerate(records):
        user = record.get("user")
        if user in users:
            user_records[user].append(idx)
        text = record.get("text", "")
        for user in users:
            email = primary_emails.get(user, "")
            phone_digits = _canonical_phone_digits(primary_phones.get(user, ""))
            if email and email in text.lower():
                mention_records[user].add(idx)
            if phone_digits and phone_digits in re.sub(r"\D", "", text):
                mention_records[user].add(idx)

    # Compute shared first/last names across users for filtering.
    parsed_names = {user: _parse_display_name(user) for user in users}
    first_name_counts: Counter[str] = Counter()
    last_names: set[str] = set()
    for _user, parsed in parsed_names.items():
        if parsed["first"]:
            first_name_counts[parsed["first"]] += 1
        if parsed["last"]:
            last_names.add(parsed["last"])

    shared_first_names = {f for f, c in first_name_counts.items() if c > 1}

    def _prepare_texts(idxs: list[int]) -> list[str]:
        texts = []
        for idx in idxs:
            text = records[idx].get("text", "")
            if max_chars and len(text) > max_chars:
                text = text[:max_chars]
            texts.append(text)
        return texts

    user_aliases: dict[str, set[str]] = {user: set() for user in users}

    for user in users:
        parsed = parsed_names[user]
        other_last_names = last_names - {parsed["last"]}

        own = user_records.get(user, [])[:max_records_per_user]
        mentions = list(mention_records.get(user, set()))[:max_records_per_user]

        own_spans: set[str] = set()
        if own:
            for spans in _run_ner(model, _prepare_texts(own), batch_size, threshold):
                for span in spans:
                    own_spans.add(_clean_alias(span.get("text", "").strip()))

        mention_spans: set[str] = set()
        if mentions:
            for spans in _run_ner(model, _prepare_texts(mentions), batch_size, threshold):
                for span in spans:
                    mention_spans.add(_clean_alias(span.get("text", "").strip()))

        accepted: set[str] = set()
        for span in own_spans:
            if _is_valid_name_alias(span, user, other_last_names, shared_first_names, require_last_name=False):
                accepted.add(span)
        for span in mention_spans:
            if _is_valid_name_alias(span, user, other_last_names, shared_first_names, require_last_name=True):
                accepted.add(span)

        # Always include display-name-derived variants.
        accepted |= _name_variants_from_display_name(user)
        user_aliases[user] = _derive_name_variants(accepted, user)

    return user_aliases


def _build_profiles(
    records: list[dict[str, Any]],
    graph: dict[str, Any],
    model: GLiNER,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build the full profiles dict."""
    users = {r.get("user") for r in records if r.get("user")}

    # Determine group membership.
    splits_per_user: dict[str, set[str]] = defaultdict(set)
    for r in records:
        splits_per_user[r["user"]].add(r.get("split", ""))
    groups = {
        user: "held_in" if "train" in splits else "held_out"
        for user, splits in splits_per_user.items()
    }

    # Detect IDs.
    allowed_domains = set(args.allowed_domains) if args.allowed_domains else _DEFAULT_ALLOWED_DOMAINS
    primary_emails = _detect_user_emails(records, allowed_domains)
    primary_phones = _detect_user_phones(records)
    job_titles = _detect_user_job_titles(
        records, users, graph, _load_job_titles(Path(args.job_titles_list)), args.seed
    )

    # Name aliases via NER.
    name_aliases = _collect_name_aliases(
        records,
        users,
        primary_emails,
        primary_phones,
        model,
        args.max_records_per_user,
        args.batch_size,
        args.entity_threshold,
        args.max_chars,
    )

    profiles: dict[str, Any] = {"seed": args.seed, "users": {}}
    for user in sorted(users):
        display_variants = _name_variants_from_display_name(user)
        aliases = name_aliases.get(user, set()) | display_variants
        aliases = {a for a in aliases if a}

        sig_name = _format_signature_name(user)
        canonical_name = sig_name
        email = primary_emails.get(user, "")
        phone = primary_phones.get(user, "")
        job_title = job_titles.get(user, "")

        profiles["users"][user] = {
            "group": groups.get(user, "held_out"),
            "ids": {
                "name": canonical_name,
                "email": email,
                "phone": phone,
                "job_title": job_title,
            },
            "aliases": {
                "name": sorted(aliases),
                "email": sorted({email, email.split("@", 1)[0]}) if email else [],
                "phone": sorted(
                    {_canonical_phone_digits(phone), _format_phone(_canonical_phone_digits(phone))}
                )
                if phone
                else [],
            },
        }

    return profiles


def _build_graph(profiles: dict[str, Any]) -> dict[str, Any]:
    """Derive a PII graph from profiles for downstream attack compatibility."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()

    type_prefix = {
        "name": "NAME",
        "email": "EMAIL_ADDRESS",
        "phone": "PHONE_NUMBER",
        "job_title": "JOB_TITLE",
    }
    relation_for_field = {
        "name": "has_name",
        "email": "has_email_address",
        "phone": "has_phone_number",
        "job_title": "has_job_title",
    }

    for user, data in profiles["users"].items():
        user_node_id = f"USER:{user}"
        nodes.append({"id": user_node_id, "type": "USER", "label": user})
        for field, prefix in type_prefix.items():
            value = data["ids"].get(field, "")
            if not value:
                continue
            canonical = value.lower()
            node_id = f"{prefix}:{canonical}"
            if node_id not in seen_nodes:
                nodes.append(
                    {
                        "id": node_id,
                        "type": prefix,
                        "label": value,
                        "canonical_value": canonical,
                    }
                )
                seen_nodes.add(node_id)
            edges.append(
                {
                    "source": user_node_id,
                    "target": node_id,
                    "relation": relation_for_field[field],
                    "score": 1.0,
                    "provenance": {"source": "extract_mallinckrodt_profiles.py"},
                }
            )

    return {"nodes": nodes, "edges": edges}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Mallinckrodt ID aliases.")
    parser.add_argument("--input", type=Path, required=True, help="Input Mallinckrodt JSONL.")
    parser.add_argument(
        "--pii-graph",
        type=Path,
        default=Path("preprocessed/pii_graph_mallinckrodt.json"),
        help="Existing PII graph JSON for job_title hints.",
    )
    parser.add_argument(
        "--gliner-model",
        type=Path,
        default=Path("models/ner/gliner_small-v2.5"),
        help="Local GLiNER model directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-records-per-user",
        type=int,
        default=500,
        help="Max records per user to run NER on.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--entity-threshold", type=float, default=0.5)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--allowed-domains",
        nargs="+",
        default=list(_DEFAULT_ALLOWED_DOMAINS),
        help="Allowed email domains.",
    )
    parser.add_argument(
        "--job-titles-list",
        type=str,
        default="configs/pii_graph_job_titles.txt",
        help="Fallback job-title list.",
    )
    args = parser.parse_args()

    print(f"Loading records from {args.input} ...")
    records = [json.loads(line) for line in args.input.open(encoding="utf-8")]
    print(f"Loaded {len(records):,} records.")

    print(f"Loading graph from {args.pii_graph} ...")
    graph = json.loads(args.pii_graph.read_text(encoding="utf-8"))

    print(f"Loading GLiNER model from {args.gliner_model} ...")
    model = GLiNER.from_pretrained(str(args.gliner_model))
    if args.device != "cpu":
        model = model.to(args.device)

    print("Building profiles ...")
    profiles = _build_profiles(records, graph, model, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = args.output_dir / "mallinckrodt_xassoc_profiles.json"
    graph_path = args.output_dir / "pii_graph_mallinckrodt_xassoc.json"

    profiles_path.write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote profiles to {profiles_path}")

    graph_out = _build_graph(profiles)
    graph_path.write_text(json.dumps(graph_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote graph to {graph_path}")

    print("Done.")


if __name__ == "__main__":
    main()
