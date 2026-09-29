"""Enron-specific PII classification, exposure counting, and extraction helpers."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable

from utils.data_utils import load_jsonl

PII_TYPES = ("email", "phone", "fax", "address")

# Phone/fax numbers in Enron signatures are overwhelmingly North-American 10-digit.
# Use ``[-. ]`` (not ``\s``) so the pattern cannot span newlines.
_PHONE_RE = re.compile(
    r"(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}",
    re.IGNORECASE,
)

# Keywords that signal a phone/fax line in a signature.
_PHONE_LABEL_RE = re.compile(
    r"(?:phone|tel|telephone|direct|cell|mobile|office|home)\s*[:.]?\s*",
    re.IGNORECASE,
)
_FAX_LABEL_RE = re.compile(
    r"(?:fax|facsimile)\s*[:.]?\s*",
    re.IGNORECASE,
)

# Very rough address heuristic: a 5-digit ZIP in the last few lines.
_ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")

# Signature delimiter lines.
_SIGNATURE_DELIM_RE = re.compile(
    r"^\s*(?:--|_+|regards|thanks|sincerely|best)(?:\b|\s*$)",
    re.IGNORECASE,
)

_HEADER_PREFIX_RE = re.compile(
    r"^\s*(?:to|from|cc|bcc|subject|date|reply-to|message-id|content-type)\s*:",
    re.IGNORECASE,
)


def normalize_email(value: str) -> str:
    """Lower-case and strip an email address."""
    return value.strip().lower()


def normalize_phone(value: str) -> str:
    """Strip a phone/fax number down to its digits."""
    return re.sub(r"\D", "", value.strip())


def canonical_phone_digits(value: str) -> str:
    """
    Return a 10-digit canonical form for a North-American phone/fax number.

    Strips a leading country-code ``1`` so that ``+1 713-...`` and
    ``713-...`` map to the same digits.
    """
    digits = normalize_phone(value)
    if len(digits) == 11 and digits.startswith("1"):
        return digits[1:]
    return digits


def is_placeholder_phone(value: str | None) -> bool:
    """Return True for obvious placeholder numbers (e.g. 000-000-0000)."""
    if value is None:
        return False
    digits = canonical_phone_digits(value)
    if len(digits) < 10:
        return True
    return all(d == digits[0] for d in digits)


def is_toll_free_phone(value: str | None) -> bool:
    """Return True for toll-free numbers (800/833/844/855/866/877/888)."""
    if value is None:
        return False
    digits = canonical_phone_digits(value)
    return len(digits) >= 10 and digits[:3] in {
        "800", "833", "844", "855", "866", "877", "888",
    }


def _phone_match_variants(value: str) -> set[str]:
    """Digit variants to search for when matching a phone/fax value in text."""
    digits = normalize_phone(value)
    variants = {digits}
    if len(digits) == 11 and digits.startswith("1"):
        variants.add(digits[1:])
    if len(digits) == 10:
        variants.add("1" + digits)
    return variants


def normalize_address(value: str) -> str:
    """Lower-case, collapse whitespace, and remove punctuation for address matching."""
    value = value.lower()
    value = re.sub(r"[^\w\s]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _extract_signature_block(text: str, tail_lines: int = 20) -> str:
    """Return the lower portion of a message where signatures live."""
    lines = text.splitlines()
    # Look for a classic delimiter in the bottom half of the text.
    search_start = max(0, len(lines) // 2)
    for i in range(len(lines) - 1, search_start - 1, -1):
        if _SIGNATURE_DELIM_RE.match(lines[i]):
            return "\n".join(lines[i + 1 :])
    return "\n".join(lines[-tail_lines:])


def _phone_candidates(text: str, label_re: re.Pattern[str]) -> list[str]:
    """Return phone-like strings that are preceded by a label (Phone:, Fax:, etc.)."""
    candidates: list[str] = []
    for line in text.splitlines():
        # If a label appears, take the first phone-like token after it.
        for match in label_re.finditer(line):
            after = line[match.end() :]
            phone_match = _PHONE_RE.search(after)
            if phone_match:
                candidates.append(phone_match.group(0))
    return candidates


def _all_phone_candidates(text: str) -> list[str]:
    """Return all phone-like strings in the text."""
    return [m.group(0) for m in _PHONE_RE.finditer(text)]


def _address_candidates(text: str) -> list[str]:
    """Return candidate address strings from the signature block."""
    sig = _extract_signature_block(text)
    candidates: list[str] = []
    lines = sig.splitlines()
    for i, line in enumerate(lines):
        if _ZIP_RE.search(line):
            # Take the ZIP line plus up to two preceding non-empty lines.
            block_lines = []
            for j in range(max(0, i - 2), i + 1):
                stripped = lines[j].strip()
                if stripped:
                    block_lines.append(stripped)
            if block_lines:
                candidates.append("\n".join(block_lines))
    return candidates


def extract_signature_pii(text: str, pii_type: str) -> list[str]:
    """Extract candidate PII strings of ``pii_type`` from an email text."""
    if pii_type == "phone":
        # Prefer labelled phones; fall back to any phone-like string.
        labelled = _phone_candidates(text, _PHONE_LABEL_RE)
        return labelled or _all_phone_candidates(text)
    if pii_type == "fax":
        return _phone_candidates(text, _FAX_LABEL_RE)
    if pii_type == "address":
        return _address_candidates(text)
    raise ValueError(f"Unsupported PII type for signature extraction: {pii_type}")


def _select_most_frequent(candidates: list[str], min_support: int = 2) -> tuple[str | None, int]:
    """Return the most frequent candidate and its count, filtering by support."""
    if not candidates:
        return None, 0
    counts = Counter(candidates)
    value, count = counts.most_common(1)[0]
    if count >= min_support:
        return value, count
    return None, 0


def extract_user_ground_truth(
    user: str,
    records: list[dict],
    pii_types: Iterable[str],
    min_support: int = 2,
    fallback_to_single: bool = True,
) -> dict[str, dict[str, object]]:
    """
    Extract ground-truth PII values for ``user`` from their own records.

    Returns a dict mapping PII type to a metadata dict with keys:
        - ``value``: the extracted string (or None)
        - ``support``: how many of the user's records contained it
        - ``low_confidence``: True if the candidate fell back to a single occurrence
    """
    pii_types = tuple(pii_types)
    out: dict[str, dict[str, object]] = {}

    if "email" in pii_types:
        out["email"] = {"value": normalize_email(user), "support": None, "low_confidence": False}

    for pii_type in pii_types:
        if pii_type == "email":
            continue
        candidates: list[str] = []
        for rec in records:
            text = rec.get("body") or rec.get("text", "")
            candidates.extend(extract_signature_pii(text, pii_type))

        value, support = _select_most_frequent(candidates, min_support)
        low_confidence = False
        if value is None and fallback_to_single and candidates:
            value, support = Counter(candidates).most_common(1)[0]
            low_confidence = True

        out[pii_type] = {"value": value, "support": support, "low_confidence": low_confidence}

    return out


def build_user_split_map(records: Iterable[dict]) -> dict[str, set[str]]:
    """Map each user to the set of split values they appear in."""
    split_map: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        user = rec.get("user")
        split = rec.get("split")
        if user and split:
            split_map[user].add(split)
    return split_map


def is_held_out(splits: set[str]) -> bool:
    """A user is held-out if none of their records are in train/val/test."""
    return not (splits & {"train", "val", "test"})


def is_held_in(splits: set[str]) -> bool:
    """A user is held-in if any of their records are in train/val/test."""
    return bool(splits & {"train", "val", "test"})


def _text_regions(text: str, header_lines_cap: int = 8, signature_tail: int = 15) -> dict[str, str]:
    """Split a record's text into header / quoted / signature / body regions."""
    lines = text.splitlines()

    # Header = contiguous prefix lines that look like email headers.
    header_end = 0
    for i, line in enumerate(lines[:header_lines_cap]):
        if line.strip() == "" or not _HEADER_PREFIX_RE.match(line):
            break
        header_end = i + 1
    header = "\n".join(lines[:header_end])
    remaining = lines[header_end:]

    # Quoted block = lines starting with '>' and everything after a
    # -----Original Message----- marker.
    quoted_mask = [False] * len(remaining)
    in_quote = False
    for i, line in enumerate(remaining):
        if in_quote or "-----Original Message-----" in line or line.lstrip().startswith(">"):
            quoted_mask[i] = True
            in_quote = True
        # Heuristic: a blank line after the marker sometimes ends the block.
        if in_quote and line.strip() == "" and i > 0 and quoted_mask[i - 1]:
            in_quote = False

    quoted_lines = [ln for ln, mask in zip(remaining, quoted_mask, strict=True) if mask]
    non_quoted = [ln for ln, mask in zip(remaining, quoted_mask, strict=True) if not mask]

    # Signature = bottom of the non-quoted body, either after a delimiter or
    # the last few lines if the body is long enough.
    signature_start = len(non_quoted)
    for i in range(len(non_quoted) - 1, -1, -1):
        if _SIGNATURE_DELIM_RE.match(non_quoted[i]):
            signature_start = i
            break

    if signature_start == len(non_quoted) and len(non_quoted) > signature_tail:
        signature_start = len(non_quoted) - signature_tail

    signature = "\n".join(non_quoted[signature_start:])
    body = "\n".join(non_quoted[:signature_start])
    quoted = "\n".join(quoted_lines)

    return {
        "header": header,
        "body": body,
        "signature": signature,
        "quoted": quoted,
    }


def _value_occurs_in(value: str | None, pii_type: str, text: str) -> bool:
    """Check whether a normalized PII value occurs in ``text``."""
    if value is None:
        return False
    if pii_type == "email":
        return normalize_email(value) in text.lower()
    if pii_type in ("phone", "fax"):
        variants = _phone_match_variants(value)
        text_digits = re.sub(r"\D", "", text)
        return any(v in text_digits for v in variants if len(v) >= 10)
    if pii_type == "address":
        return normalize_address(value) in normalize_address(text)
    # Generic fallback for PII types sourced from the graph (job_title, etc.).
    return value.strip().lower() in text.lower()


def _find_pii_positions(value: str | None, pii_type: str, text: str) -> list[int]:
    """Return character start positions of ``value`` occurrences in ``text``."""
    if value is None:
        return []
    positions: list[int] = []
    if pii_type == "email":
        target = normalize_email(value)
        for match in _EMAIL_RE.finditer(text):
            if normalize_email(match.group(0)) == target:
                positions.append(match.start())
        return positions
    if pii_type in ("phone", "fax"):
        target_digits = canonical_phone_digits(value)
        for match in _PHONE_RE.finditer(text):
            if canonical_phone_digits(match.group(0)) == target_digits:
                positions.append(match.start())
        return positions
    if pii_type == "address":
        target = normalize_address(value)
        # Naive sliding-window search across lines.
        lines = text.splitlines()
        for i in range(len(lines)):
            window = "\n".join(lines[max(0, i - 2) : i + 1])
            if target in normalize_address(window):
                # Approximate position to the start of the window.
                pos = sum(len(line) + 1 for line in lines[: max(0, i - 2)])
                positions.append(pos)
        return positions
    # Generic fallback for PII types sourced from the graph (job_title, etc.).
    target = value.strip().lower()
    start = 0
    while True:
        idx = text.lower().find(target, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1
    return positions


# Very rough email regex for locating occurrences in text.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# Recipient header fields (To / Cc / Bcc). Used for header-only contexts.
_HEADER_FIELD_RE = re.compile(
    r"^\s*(to|cc|bcc)\s*:",
    re.IGNORECASE,
)
# Variant that also matches quoted/forwarded header lines (e.g. "> To:").
_HEADER_FIELD_RE_QUOTED = re.compile(
    r"^[>\s]*(?:to|cc|bcc)\s*:",
    re.IGNORECASE,
)


def _line_start(text: str, pos: int) -> int:
    """Return the character index of the start of the line containing ``pos``."""
    idx = text.rfind("\n", 0, pos)
    return 0 if idx == -1 else idx + 1


def _is_continuation_line(text: str, line_start: int) -> bool:
    """Return True if the line at ``line_start`` is a header continuation line."""
    line = text[line_start:]
    return bool(line) and line[0] in " \t"


def _is_continuation_line_quoted(text: str, line_start: int) -> bool:
    """Return True for quoted (e.g. `>    ...`) or whitespace continuation lines."""
    line = text[line_start:]
    if line.startswith(">") and _HEADER_FIELD_RE_QUOTED.match(line) is None:
        return True
    return bool(line) and line[0] in " \t"


def _header_field_start(
    text: str,
    pos: int,
    header_re: re.Pattern[str] = _HEADER_FIELD_RE,
    is_continuation: object = _is_continuation_line,
) -> int | None:
    """Return the start index of the To/Cc/Bcc field containing ``pos``.

    Args:
        text: Full record text.
        pos: Character position inside the field.
        header_re: Regex matching the start of a header field line.
        is_continuation: Callable returning True for continuation lines.

    Returns:
        Start index of the header field, or ``None`` if ``pos`` is not inside
        a matching header field.
    """
    line_start = _line_start(text, pos)
    # Walk backwards over continuation lines until we find the field start.
    while line_start > 0:
        if header_re.match(text[line_start:]):
            return line_start
        if is_continuation(text, line_start):
            prev_newline = text.rfind("\n", 0, line_start - 1)
            line_start = 0 if prev_newline == -1 else prev_newline + 1
            continue
        return None
    if header_re.match(text[line_start:]):
        return line_start
    return None


def _signature_block_start(text: str, pos: int, tail_lines: int = 15) -> int | None:
    """Return the start index of the signature block containing ``pos``.

    The signature block starts after a classic delimiter line (``--``,
    ``Regards``, etc.) or, if none is found, is taken as the last
    ``tail_lines`` lines of the text.

    Returns:
        Character start index of the signature block, or ``None`` if ``pos``
        is not inside it.
    """
    lines = text.splitlines()
    pos_line = text[:pos].count("\n")
    if pos_line >= len(lines):
        return None

    search_start = max(0, len(lines) // 2)
    delimiter_line: int | None = None
    for i in range(len(lines) - 1, search_start - 1, -1):
        if _SIGNATURE_DELIM_RE.match(lines[i]):
            delimiter_line = i
            break

    if delimiter_line is not None:
        if pos_line <= delimiter_line:
            return None
        sig_start_line = delimiter_line + 1
    else:
        sig_start_line = max(0, len(lines) - tail_lines)
        if pos_line < sig_start_line:
            return None

    char_idx = 0
    for i in range(sig_start_line):
        char_idx += len(lines[i]) + 1
    return char_idx


def find_exposure_contexts(
    records: Iterable[dict],
    target_user: str,
    value: str | None,
    raw_value: str | None,
    pii_type: str,
    tokenizer,
    max_samples: int = 100,
    max_context_tokens: int = 256,
    mask_token: str = "",
    skip_own_records: bool = True,
    context_mode: str = "any",
) -> list[str]:
    """Find prefix contexts where ``value`` occurs in other users' train records.

    For each occurrence, the function returns a string containing the
    ``max_context_tokens`` tokens immediately before the value, optionally
    followed by ``mask_token``. These contexts are used for cloze-style
    extraction scoring.

    Args:
        records: Iterable of records (typically the train split).
        target_user: The canonical user identifier; records belonging to this
            user are skipped to preserve the cross-user leakage setting.
        value: Normalized PII value.
        raw_value: Raw PII string as it appears in the user's own records. If
            ``None``, ``value`` is used.
        pii_type: One of the supported PII types.
        tokenizer: HuggingFace tokenizer used to count context tokens.
        max_samples: Maximum number of contexts to return.
        max_context_tokens: Maximum number of tokens to keep in the prefix.
        mask_token: Token appended at the end of each context to mark the blank.
            If empty, no separator is added.
        skip_own_records: If True, skip records authored by ``target_user``.
        context_mode: ``"any"`` uses any occurrence (header, body, signature,
            quoted). ``"header"`` uses only main-header To/Cc/Bcc fields.
            ``"structured"`` uses headers, quoted/forwarded headers, and
            signature blocks.

    Returns:
        A list of context strings, each optionally ending with ``mask_token``.
    """
    if value is None:
        return []
    if context_mode not in ("any", "header", "structured"):
        raise ValueError(
            f"context_mode must be 'any', 'header', or 'structured', got {context_mode!r}"
        )

    search_value = raw_value if raw_value is not None else value
    contexts: list[str] = []
    for rec in records:
        if skip_own_records and rec.get("user") == target_user:
            continue
        text = rec.get("text", "")
        if not _value_occurs_in(value, pii_type, text):
            continue

        # Prefer literal matches of the raw value; fall back to canonical search.
        positions: list[int] = []
        if search_value and search_value in text:
            start = 0
            while True:
                idx = text.find(search_value, start)
                if idx == -1:
                    break
                positions.append(idx)
                start = idx + 1
        if not positions:
            positions = _find_pii_positions(value, pii_type, text)

        for pos in positions:
            if context_mode == "any":
                prefix_text = text[:pos]
            elif context_mode == "header":
                field_start = _header_field_start(text, pos)
                if field_start is None:
                    continue
                prefix_text = text[field_start:pos]
            else:  # structured
                field_start = _header_field_start(
                    text,
                    pos,
                    header_re=_HEADER_FIELD_RE_QUOTED,
                    is_continuation=_is_continuation_line_quoted,
                )
                if field_start is not None:
                    prefix_text = text[field_start:pos]
                else:
                    sig_start = _signature_block_start(text, pos)
                    if sig_start is None:
                        continue
                    prefix_text = text[sig_start:pos]

            # Keep the last max_context_tokens tokens of the prefix.
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            if len(prefix_ids) > max_context_tokens:
                prefix_ids = prefix_ids[-max_context_tokens:]
            truncated_prefix = tokenizer.decode(prefix_ids, skip_special_tokens=True)
            contexts.append(truncated_prefix + mask_token if mask_token else truncated_prefix)
            if len(contexts) >= max_samples:
                return contexts

    return contexts


def compute_exposure(
    value: str | None,
    pii_type: str,
    train_records: Iterable[dict],
) -> dict[str, int]:
    """
    Count how many train records contain ``value`` and where.

    Returns counts for ``total``, ``header``, ``body``, ``signature``, ``quoted``.
    """
    counts = {"total": 0, "header": 0, "body": 0, "signature": 0, "quoted": 0}
    if value is None:
        return counts

    # train_records may be a generator; materialize once so we can show progress
    # and iterate multiple times for region checks without re-consuming it.
    records = list(train_records)

    # Precompute lower-cased and digit-only versions of each region to avoid
    # repeating the same regex/substring work for every PII value.
    text_regions = [(rec.get("text", ""), _text_regions(rec.get("text", ""))) for rec in records]
    region_caches: list[dict[str, str]] = []
    for text, regions in text_regions:
        cache: dict[str, str] = {"text_lower": text.lower()}
        if pii_type in ("phone", "fax"):
            cache["text_digits"] = re.sub(r"\D", "", text)
            for region_name, region_text in regions.items():
                cache[f"{region_name}_digits"] = re.sub(r"\D", "", region_text)
        else:
            for region_name, region_text in regions.items():
                cache[f"{region_name}_lower"] = region_text.lower()
        region_caches.append(cache)

    if pii_type in ("phone", "fax"):
        variants = {v for v in _phone_match_variants(value) if len(v) >= 10}
        if not variants:
            return counts
        for (_text, _regions), cache in zip(text_regions, region_caches, strict=True):
            if any(v in cache["text_digits"] for v in variants):
                counts["total"] += 1
                if any(v in cache["header_digits"] for v in variants):
                    counts["header"] += 1
                if any(v in cache["quoted_digits"] for v in variants):
                    counts["quoted"] += 1
                if any(v in cache["signature_digits"] for v in variants):
                    counts["signature"] += 1
                if any(v in cache["body_digits"] for v in variants):
                    counts["body"] += 1
    elif pii_type == "email":
        target = normalize_email(value)
        for (_text, _regions), cache in zip(text_regions, region_caches, strict=True):
            if target in cache["text_lower"]:
                counts["total"] += 1
                if target in cache["header_lower"]:
                    counts["header"] += 1
                if target in cache["quoted_lower"]:
                    counts["quoted"] += 1
                if target in cache["signature_lower"]:
                    counts["signature"] += 1
                if target in cache["body_lower"]:
                    counts["body"] += 1
    else:
        target = value.strip().lower()
        if not target:
            return counts
        for (_text, _regions), cache in zip(text_regions, region_caches, strict=True):
            if target in cache["text_lower"]:
                counts["total"] += 1
                if target in cache["header_lower"]:
                    counts["header"] += 1
                if target in cache["quoted_lower"]:
                    counts["quoted"] += 1
                if target in cache["signature_lower"]:
                    counts["signature"] += 1
                if target in cache["body_lower"]:
                    counts["body"] += 1

    return counts


def build_distractor_pool(
    values: Iterable[str | None],
    pii_type: str,
    target_value: str | None = None,
    same_domain: bool = True,
) -> list[str]:
    """
    Build a pool of distractor values for a PII type.

    For ``email``, distractors are restricted to the same domain as the target
    when ``same_domain`` is True, to prevent the model from winning on format
    guessing alone.
    """
    pool = sorted({v for v in values if v and v != target_value})
    if pii_type == "email" and same_domain and target_value and "@" in target_value:
        domain = target_value.split("@", 1)[1]
        pool = [v for v in pool if v.endswith(f"@{domain}")]
    return pool


def filter_shared_and_placeholder_pii(
    all_ground_truth: dict[str, dict[str, dict[str, object]]],
) -> dict[str, dict[str, dict[str, object]]]:
    """
    Remove phone/fax values that are placeholders or shared by multiple users.

    Modifies the ground-truth dicts in place: matching values are replaced by
    ``None`` and an ``excluded_reason`` key is added (``"placeholder"`` or
    ``"shared"``).
    """
    # Count canonical phone/fax digits across users.
    phone_counts: Counter[str] = Counter()
    for gt in all_ground_truth.values():
        for pii_type in ("phone", "fax"):
            meta = gt.get(pii_type)
            if not meta or meta.get("value") is None:
                continue
            digits = canonical_phone_digits(meta["value"])
            if len(digits) >= 10:
                phone_counts[digits] += 1

    for gt in all_ground_truth.values():
        for pii_type in ("phone", "fax"):
            meta = gt.get(pii_type)
            if not meta or meta.get("value") is None:
                continue
            value = meta["value"]
            if is_placeholder_phone(value):
                meta["raw_value"] = value
                meta["value"] = None
                meta["excluded_reason"] = "placeholder"
                continue
            if is_toll_free_phone(value):
                meta["raw_value"] = value
                meta["value"] = None
                meta["excluded_reason"] = "toll_free"
                continue
            digits = canonical_phone_digits(value)
            if len(digits) >= 10 and phone_counts[digits] > 1:
                meta["raw_value"] = value
                meta["value"] = None
                meta["excluded_reason"] = "shared"

    return all_ground_truth


def load_records(data_path: str) -> list[dict]:
    """Load a JSONL dataset."""
    return list(load_jsonl(data_path))
