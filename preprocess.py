from __future__ import annotations

import argparse
import csv
import email
import functools
import hashlib
import json
import os
import platform
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from email import policy as email_policy
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from config import PreprocessConfig

_PREPROCESS_CFG = PreprocessConfig()


# ---------------------------------------------------------------------------
# Windows trailing-dot workaround
# ---------------------------------------------------------------------------
def _windows_extended_path(path: Path) -> Path:
    """Return a Windows extended-length path to bypass trailing-dot stripping."""
    if platform.system() != "Windows":
        return path
    abs_path = path.resolve()
    # Only prefix if not already prefixed and the path has a trailing dot component
    prefix = "\\\\?\\"
    if str(abs_path).startswith(prefix):
        return abs_path
    return Path(prefix + str(abs_path))


def safe_read_bytes(path: Path) -> bytes:
    """Read file bytes, working around Windows trailing-dot filenames."""
    return _windows_extended_path(path).read_bytes()


def safe_path_exists(path: Path) -> bool:
    """Check if a path exists, working around Windows trailing-dot filenames."""
    return _windows_extended_path(path).exists()


def safe_path_is_file(path: Path) -> bool:
    """Check if a path is a file, working around Windows trailing-dot filenames."""
    return _windows_extended_path(path).is_file()

# ---------------------------------------------------------------------------
# Regexes for body cleaning (shared)
# ---------------------------------------------------------------------------
_FWD_HDR_RE = re.compile(r"-{3,}\s*Forwarded\b.*?(?=\n\n|\Z)", re.I | re.S)
_REPLY_HDR_RE = re.compile(r"^On .{5,120} wrote:\s*$", re.M)
_QUOTE_LINE_RE = re.compile(r"^>.*$", re.M)
_STRAY_HDR_RE = re.compile(
    r"^(To|Cc|Bcc|Subject|From|Date|Message-ID|Content-Type|"
    r"Mime-Version|X-[\w-]+)\s*:.*$",
    re.M | re.I,
)
_MULTI_NL_RE = re.compile(r"\n{3,}")
_MULTI_SP_RE = re.compile(r"[ \t]{2,}")


# ---------------------------------------------------------------------------
# IDL / Mallinckrodt CSV loader
# ---------------------------------------------------------------------------
_IDL_HEADER_NAMES = ["Subject", "From", "Date", "To", "Cc"]
_IDL_HEADER_BODY_DELIM_RE = re.compile(r"\s{4,}|\n\n")
_IDL_SUBJECT_TRUNC_RE = re.compile(r"\s{2,}From:\s", re.I)


def _clean_idl_subject(raw_subject: str) -> str:
    """
    Truncate OCR-extracted subjects that swallowed forwarded message text.

    Some OCR texts have no proper header ``From:`` after ``Subject:``; the
    next ``From:`` occurrence is inside the body/signature. If the extracted
    subject is very long, truncate at the first signature-like ``From:`` so
    the subject stays short and the body stays in the body.
    """
    raw_subject = raw_subject.strip()
    if len(raw_subject) <= 200:
        return raw_subject

    match = _IDL_SUBJECT_TRUNC_RE.search(raw_subject)
    if match:
        return raw_subject[: match.start()].strip()

    return raw_subject


def _parse_idl_ocr_headers(ocr_text: str) -> dict[str, str]:
    """
    Parse the embedded email header block from UCSF IDL OCR text.

    The OCR text starts with a header line such as::

        ?Subject: FW: PREA manuscript information  From: "Decker, John F" ...

    This function extracts ``Subject``, ``From``, ``Date``, ``To`` and
    ``Cc`` values and returns the remaining text as ``body``.
    """
    text = ocr_text.lstrip("\ufeff").lstrip("?")

    positions: dict[str, int] = {}
    for name in _IDL_HEADER_NAMES:
        needle = f"{name}:"
        idx = text.find(needle)
        if idx != -1:
            positions[name] = idx

    if "Subject" not in positions or "From" not in positions:
        return {}

    ordered = sorted(positions.items(), key=lambda x: x[1])
    result: dict[str, str] = {}
    for i, (name, pos) in enumerate(ordered):
        start = pos + len(name) + 1
        if i + 1 < len(ordered):
            end = ordered[i + 1][1]
            value = text[start:end].strip()
        else:
            rest = text[start:]
            match = _IDL_HEADER_BODY_DELIM_RE.search(rest)
            if match:
                value = rest[: match.start()].strip()
                result["body"] = rest[match.end() :].strip()
            else:
                value = rest.strip()
                result["body"] = ""
        result[name.lower()] = value

    if "subject" in result:
        result["subject"] = _clean_idl_subject(result["subject"])

    return result


def _extract_emails_from_text(text: str) -> list[str]:
    """Extract all email addresses from a string, lowercased and deduplicated."""
    if not text:
        return []
    found = re.findall(r"[\w.+%-]+@[\w.-]+\.[a-z]{2,}", text, re.I)
    seen: set[str] = set()
    out: list[str] = []
    for addr in found:
        a = addr.lower()
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def _match_sender_to_custodian(
    sender: str,
    eligible_custodians: set[str],
) -> str | None:
    """Return the eligible custodian whose name best matches ``sender``.

    Matching prefers last-name variants and falls back to first-name variants
    only when no last-name match exists.  Returns ``None`` when no custodian
    matches.
    """
    sender_tokens = _extract_sender_name_tokens(sender)
    if not sender_tokens:
        return None

    best: str | None = None
    best_score = 0
    for custodian in eligible_custodians:
        parts = custodian.split(",", 1)
        last_name_part = parts[0].strip() if parts else ""
        first_name_part = parts[1].strip() if len(parts) > 1 else ""
        last_variants = _name_variants(last_name_part)
        first_variants = _name_variants(first_name_part.split()[0] if first_name_part else "")

        # Score: last-name match is worth 2, first-name match is worth 1.
        score = 0
        if last_variants and any(variant in token for variant in last_variants for token in sender_tokens):
            score += 2
        if first_variants and any(variant in token for variant in first_variants for token in sender_tokens):
            score += 1

        if score > best_score:
            best_score = score
            best = custodian

    # Require both last-name and first-name matches for a confident assignment;
    # a last-name-only match is ambiguous (e.g. multiple people share a surname).
    return best if best_score >= 3 else None


def load_idl_records(
    data_dir: Path,
    max_users: int | None = None,
    min_examples: int = 100,
    min_body_words: int = 5,
    max_examples_per_user: int | None = None,
    max_files: int | None = None,
    custodian_whitelist: set[str] | None = None,
    user_resolver: str = "identity",
) -> list[TextRecord]:
    """
    Load Mallinckrodt litigation emails from UCSF IDL metadata CSVs.

    Each CSV is pipe-delimited and contains an ``ocr_text`` column with the
    full document OCR. Rows whose ``type`` contains ``Email`` are treated as
    email records.

    For ``user_resolver='identity'`` the ``custodian`` column is used as the
    privacy unit and only records from the selected custodians' mailboxes are
    loaded.

    For the IDL canonical-sender resolver (``user_resolver='idl-kandpal'``)
    the privacy unit is the canonical sender. The loader scans all selected
    files, matches each email's ``author`` / ``From`` header to the eligible
    custodian names, and assigns the record to the matching custodian. This
    captures emails sent by the custodian even when they appear in another
    custodian's mailbox.

    Headers (``Subject``, ``From``, ``Date``, ``To``, ``Cc``) are extracted from
    the OCR text. The ``From`` value is dropped from the formatted model input
    downstream to avoid trivializing user-inference attacks.

    Args:
        data_dir: Directory containing the ``Mallinckrodt_Litigation_Documents*.csv``
            metadata files.
        max_users: If set, keep only the top-N custodians by email count.
        min_examples: Minimum emails per custodian to retain.
        min_body_words: Minimum word count for the cleaned body (0 to disable).
        max_examples_per_user: Optional per-user cap for quick experiments.
        max_files: Optional cap on the number of CSV files to scan (quick tests).
        custodian_whitelist: Optional set of custodian names to restrict loading.
        user_resolver: Which resolver will be used downstream. Affects how
            ``user`` is assigned during loading.

    Returns:
        A list of ``TextRecord`` objects, one per email.
    """
    import pandas as pd

    cross_mailbox_mode = user_resolver == "idl-kandpal"

    csv_files = sorted(data_dir.glob("Mallinckrodt_Litigation_Documents*.csv"))
    if not csv_files:
        sys.exit(f"ERROR: no Mallinckrodt CSV files found under '{data_dir}'.")

    if max_files:
        csv_files = csv_files[:max_files]
        print(f"  Loading IDL Mallinckrodt from first {len(csv_files)} CSV file(s) (max_files={max_files}) ...", flush=True)
    else:
        print(f"  Loading IDL Mallinckrodt from {len(csv_files)} CSV file(s) ...", flush=True)

    # First pass: count emails per custodian and record which files they appear in.
    custodian_counts: Counter = Counter()
    custodian_files: dict[str, set[Path]] = defaultdict(set)
    file_counts: Counter[Path] = Counter()
    for f in tqdm(csv_files, desc="Scanning custodian counts", unit="file", file=sys.stdout, ascii=True):
        df = pd.read_csv(
            f,
            sep="|",
            usecols=["type", "custodian"],
            dtype=str,
            low_memory=False,
            engine="c",
        )
        email_mask = df["type"].fillna("").str.contains("Email", case=False, na=False)
        custodians = df.loc[email_mask, "custodian"].dropna().str.strip()
        for custodian in custodians:
            custodian_counts[custodian] += 1
            custodian_files[custodian].add(f)
            file_counts[f] += 1

    eligible = [
        custodian
        for custodian, count in custodian_counts.most_common()
        if count >= min_examples and (custodian_whitelist is None or custodian in custodian_whitelist)
    ]
    if max_users:
        eligible = eligible[:max_users]
    eligible_set = set(eligible)

    print(f"  {len(eligible)} custodian(s) meet >= {min_examples} emails", flush=True)
    if max_users:
        print(f"  Limited to top {max_users} custodian(s)", flush=True)

    # Second pass: read the files most likely to contain relevant records.
    if cross_mailbox_mode:
        # For sender-based resolution we may need every file; emails sent by a
        # selected custodian can live in any custodian's mailbox.
        files_to_read = sorted(csv_files, key=lambda f: file_counts[f], reverse=True)
        print(f"  Scanning all {len(files_to_read)} file(s) for emails sent by selected custodians", flush=True)
    else:
        files_to_read = set()
        for custodian in eligible_set:
            files_to_read.update(custodian_files.get(custodian, set()))
        files_to_read = sorted(files_to_read, key=lambda f: file_counts[f], reverse=True)
        print(f"  {len(files_to_read)} file(s) contain selected custodian(s)", flush=True)

    records: list[TextRecord] = []
    skipped = 0
    per_user_counts: Counter = Counter()

    _idl_load_cols = {
        "type",
        "custodian",
        "ocr_text",
        "author",
        "recipient",
        "copied",
        "subject",
        "document_date",
    }
    csv.field_size_limit(1_000_000_000)
    target_per_user = max_examples_per_user
    for f in tqdm(files_to_read, desc="Loading selected records", unit="file", file=sys.stdout, ascii=True):
        if target_per_user and all(per_user_counts[c] >= target_per_user for c in eligible_set):
            break
        with f.open("r", encoding="utf-8", errors="replace") as fin:
            reader = csv.DictReader(fin, delimiter="|")
            for row in reader:
                if not row:
                    continue
                row_type = str(row.get("type", "") or "").strip()
                if "email" not in row_type.lower():
                    continue

                custodian = str(row.get("custodian", "") or "").strip()

                # Use metadata columns as fallback when OCR header parsing fails.
                author = str(row.get("author", "") or "").strip()
                recipient = str(row.get("recipient", "") or "").strip()
                copied = str(row.get("copied", "") or "").strip()
                subject_col = str(row.get("subject", "") or "").strip()
                date_col = str(row.get("document_date", "") or "").strip()

                ocr_text = str(row.get("ocr_text", "") or "").strip()
                if not ocr_text:
                    skipped += 1
                    continue

                sender = author or ""
                if not sender:
                    sender = _extract_emails_from_text(ocr_text)[:1]
                    sender = sender[0] if sender else ""

                if cross_mailbox_mode:
                    matched_custodian = _match_sender_to_custodian(sender, eligible_set)
                    if not matched_custodian:
                        continue
                    user_key = matched_custodian
                else:
                    if custodian not in eligible_set:
                        continue
                    user_key = custodian

                if max_examples_per_user and per_user_counts[user_key] >= max_examples_per_user:
                    continue

                parsed = _parse_idl_ocr_headers(ocr_text)
                body = parsed.get("body", ocr_text)
                if min_body_words > 0 and len(body.split()) < min_body_words:
                    skipped += 1
                    continue

                # Prefer parsed From when available; author is already the
                # metadata fallback.
                sender = parsed.get("from", sender)

                to = _extract_emails_from_text(parsed.get("to", recipient))
                cc = _extract_emails_from_text(parsed.get("cc", copied))
                # Prefer the metadata subject when available; OCR subject can
                # accidentally swallow forwarded message text.
                subject = subject_col or parsed.get("subject", "")
                date = parsed.get("date", date_col)

                raw_headers = "\n".join(
                    f"{k}: {v}"
                    for k, v in [
                        ("Subject", subject),
                        ("From", sender),
                        ("Date", date),
                        ("To", ", ".join(to)),
                        ("Cc", ", ".join(cc)),
                    ]
                    if v
                )

                records.append(
                    TextRecord(
                        user=user_key,
                        sender=sender,
                        to=to,
                        cc=cc,
                        subject=subject,
                        body=body,
                        date=date,
                        raw_headers=raw_headers,
                        raw_path=f"{f}:{custodian}",
                        is_forward=False,
                    )
                )
                per_user_counts[user_key] += 1

    print(f"  Loaded {len(records):,} records (skipped {skipped:,})", flush=True)
    return records


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class TextRecord:
    """Structured representation of a text record (email, synthetic post, etc.)."""

    user: str = ""
    sender: str = ""
    to: list[str] = field(default_factory=list)
    cc: list[str] = field(default_factory=list)
    bcc: list[str] = field(default_factory=list)
    subject: str = ""
    body: str = ""
    content_type: str = ""
    date: str = ""
    raw_headers: str = ""
    raw_path: str = ""
    is_forward: bool = False
    profile_fields: dict[str, str] = field(default_factory=dict, repr=False)
    prepend_profile_fields: bool = True


# ---------------------------------------------------------------------------
# Core parsing
# ---------------------------------------------------------------------------
def decode_payload(part: email.message.Message) -> str:
    raw = part.get_payload(decode=True)
    if raw is None:
        return str(part.get_payload() or "")
    charset = part.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


def extract_body(msg: email.message.Message) -> str:
    body_parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body_parts.append(decode_payload(part))
    else:
        body_parts.append(decode_payload(msg))
    return "\n".join(body_parts)


def clean_body(raw: str) -> str:
    text = _FWD_HDR_RE.sub("", raw)
    text = _REPLY_HDR_RE.sub("", text)
    text = _QUOTE_LINE_RE.sub("", text)
    text = _STRAY_HDR_RE.sub("", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    text = _MULTI_SP_RE.sub(" ", text)
    return text.strip()


def _name_to_email(name: str) -> str:
    """Convert a display name like 'Kenneth Lay' or 'Lay, Kenneth' to an email-like id."""
    name = name.strip().strip('"').strip()
    name = name.split("/")[0].strip()
    if "," in name:
        parts = [p.strip() for p in name.split(",")]
        if len(parts) == 2:
            name = f"{parts[1]} {parts[0]}"
    words = name.split()
    return ".".join(words).lower()


def _extract_sender(msg: email.message.Message) -> str:
    """Extract sender from From:, falling back to X-From: for Lotus Notes placeholders."""
    from_hdr: str = msg.get("From", "")
    sender_match = re.search(r"[\w.+%-]+@[\w.-]+\.[a-z]{2,}", from_hdr, re.I)
    sender = sender_match.group(0).lower() if sender_match else ""

    if not sender or sender == "no.address@enron.com":
        xfrom_hdr: str = msg.get("X-From", "")
        if xfrom_hdr:
            xfrom_match = re.search(r"[\w.+%-]+@[\w.-]+\.[a-z]{2,}", xfrom_hdr, re.I)
            if xfrom_match:
                sender = xfrom_match.group(0).lower()
            else:
                # Strip Notes CN paths and @ENRON suffixes, then use display name.
                name = re.sub(r"</O=ENRON[^>]*>", "", xfrom_hdr, flags=re.I)
                name = re.sub(r"@ENRON", "", name, flags=re.I)
                quote_match = re.search(r'"([^"]+)"', name)
                if quote_match:
                    name = quote_match.group(1)
                sender = _name_to_email(name)
                if "@" not in sender:
                    sender = f"{sender}@enron.com"

    return sender


def parse_email_file(
    path: Path,
    min_body_words: int = 5,
    clean_body: bool = True,
) -> TextRecord | None:
    try:
        raw = safe_read_bytes(path)
        msg = email.message_from_bytes(raw, policy=email_policy.compat32)
    except Exception:
        return None

    # Sender
    sender = _extract_sender(msg)

    # Recipients
    def _addrs(hdr: str) -> list[str]:
        vals = msg.get_all(hdr, [])
        addrs = []
        for v in vals:
            addrs.extend(re.findall(r"[\w.+%-]+@[\w.-]+\.[a-z]{2,}", str(v), re.I))
        return [a.lower() for a in addrs]

    to_addrs = _addrs("To")
    cc_addrs = _addrs("Cc")
    bcc_addrs = _addrs("Bcc")

    body_raw = extract_body(msg)
    body_text = clean_body(body_raw) if clean_body else body_raw

    # Forward detection (always run on the raw body so the flag is stable)
    is_forward = bool(_FWD_HDR_RE.search(body_raw))

    if min_body_words > 0 and len(body_text.split()) < min_body_words:
        return None

    raw_headers = "\n".join(f"{k}: {v}" for k, v in msg.items())

    return TextRecord(
        sender=sender,
        to=to_addrs,
        cc=cc_addrs,
        bcc=bcc_addrs,
        subject=str(msg.get("Subject", "")),
        body=body_text,
        date=str(msg.get("Date", "")),
        raw_headers=raw_headers,
        raw_path=str(path),
        is_forward=is_forward,
    )


# ---------------------------------------------------------------------------
# CC / BCC expansion
# ---------------------------------------------------------------------------
def expand_ccbcc(records: list[TextRecord]) -> list[TextRecord]:
    """Create one record per CC/BCC recipient (Chua et al. style)."""
    out: list[TextRecord] = []
    for r in records:
        all_recips = r.to + r.cc + r.bcc
        if not all_recips:
            all_recips = [""]
        for recip in all_recips:
            copy = TextRecord(
                user=r.user,
                sender=r.sender,
                to=[recip] if recip else [],
                cc=[],
                bcc=[],
                subject=r.subject,
                body=r.body,
                date=r.date,
                raw_path=r.raw_path,
                is_forward=r.is_forward,
            )
            out.append(copy)
    return out


# ---------------------------------------------------------------------------
# User resolution strategies
# ---------------------------------------------------------------------------
def _top_senders(records: list[TextRecord], n: int = 5) -> list[tuple[str, int]]:
    """Return the most frequent sender addresses in a list of records."""
    return Counter(r.sender for r in records).most_common(n)


def resolve_user_kandpal(
    records: list[TextRecord],
    maildir: Path,
    keep_all: bool = False,
    collapse_aliases: bool = False,
    **_kwargs,
) -> dict[str, list[TextRecord]]:
    """
    Canonical-sender resolver (paper strategy):
      - Group by maildir folder name (e.g. 'beck-s')
      - Infer last name ('beck')
      - Keep candidate senders whose email contains the last name
      - Pick the most frequent candidate as the canonical sender
      - Retain only emails from that canonical sender.
    """
    by_folder: dict[str, list[TextRecord]] = defaultdict(list)
    for r in records:
        try:
            rel = Path(r.raw_path).relative_to(maildir)
            folder = rel.parts[0]
        except ValueError:
            continue
        by_folder[folder].append(r)

    user_records: dict[str, list[TextRecord]] = defaultdict(list)
    dropped_folders: list[str] = []
    folder_summaries: list[tuple[str, str, int, int, list[tuple[str, int]]]] = []

    for folder, f_records in by_folder.items():
        parts = folder.rsplit("-", 1)
        if len(parts) != 2:
            continue
        last_name = parts[0].lower()

        candidates = [r.sender for r in f_records if last_name in r.sender.lower()]
        if not candidates:
            dropped_folders.append(folder)
            continue
        canonical = Counter(candidates).most_common(1)[0][0]

        kept = 0
        for r in f_records:
            is_canonical = r.sender.lower() == canonical.lower()
            is_alias = collapse_aliases and (last_name in r.sender.lower())
            if keep_all or is_canonical or is_alias:
                r.user = canonical.lower()
                user_records[r.user].append(r)
                kept += 1

        folder_summaries.append((folder, canonical, kept, len(f_records), _top_senders(f_records)))

    # Detailed debug logging
    print("\n  Canonical-sender resolution summary:")
    print(f"    Total folders scanned: {len(by_folder)}")
    print(f"    Folders resolved: {len(folder_summaries)}")
    print(f"    Folders dropped (no candidate sender): {len(dropped_folders)}")
    if dropped_folders:
        print(f"    Dropped folder names: {', '.join(sorted(dropped_folders))}")

    print("\n  Per-folder canonical sender assignment (sorted by kept count):")
    folder_summaries.sort(key=lambda x: x[2])
    for folder, canonical, kept, total, top5 in folder_summaries:
        warn = "  *** LOW COUNT ***" if kept < 150 else ""
        print(f"    {folder:<22} -> {canonical:<38} kept {kept:>5}/{total:<6}{warn}")
        for sender, count in top5:
            marker = " <-- canonical" if sender.lower() == canonical.lower() else ""
            print(f"                              {count:>5}  {sender}{marker}")

    low_count_users = sorted(
        [(u, len(recs)) for u, recs in user_records.items() if len(recs) < 150],
        key=lambda x: x[1],
    )
    print(f"\n  Users likely to be dropped by >=150 filter: {len(low_count_users)}")
    for user, count in low_count_users:
        print(f"    {count:>5}  {user}")

    return dict(user_records)


def resolve_user_custom(
    records: list[TextRecord],
    map_file: Path | None = None,
    regex_pattern: str | None = None,
    **_kwargs,
) -> dict[str, list[TextRecord]]:
    """
    Custom resolution via CSV mapping (email -> user) or regex.
    """
    mapping: dict[str, str] = {}
    if map_file and map_file.exists():
        with map_file.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    mapping[row[0].strip().lower()] = row[1].strip()

    regex = re.compile(regex_pattern, re.I) if regex_pattern else None
    user_records: dict[str, list[TextRecord]] = defaultdict(list)
    for r in records:
        user = mapping.get(r.sender.lower())
        if not user and regex:
            m = regex.search(r.sender)
            user = m.group(0) if m else None
        if not user:
            user = r.sender.lower()
        r.user = user
        user_records[user].append(r)
    return dict(user_records)


def resolve_user_identity(
    records: list[TextRecord],
    **_kwargs,
) -> dict[str, list[TextRecord]]:
    """
    Identity resolution: each record's `user` field is already the privacy unit.
    """
    user_records: dict[str, list[TextRecord]] = defaultdict(list)
    for r in records:
        if r.user:
            user_records[r.user].append(r)
    return dict(user_records)


def _extract_sender_email(sender: str) -> str:
    """Extract the email address from a sender string, lowercased."""
    match = re.search(r"[\w.+%-]+@[\w.-]+\.[a-z]{2,}", sender, re.I)
    return match.group(0).lower() if match else sender.lower().strip()


def _extract_sender_name_tokens(sender: str) -> set[str]:
    """Return a set of searchable tokens from a sender string."""
    tokens: set[str] = set()
    # Add email local part
    email = _extract_sender_email(sender)
    if "@" in email:
        tokens.add(email.split("@")[0])
    # Add display name words
    display = re.sub(r"[\w.+%-]+@[\w.-]+\.[a-z]{2,}", "", sender)
    display = re.sub(r"[^\w\s'-]", " ", display)
    for word in display.split():
        word = re.sub(r"[^\w']", "", word).lower().strip("'")
        if word:
            tokens.add(word)
    return tokens


def _name_variants(name: str) -> list[str]:
    """
    Generate searchable name variants from a display name fragment.

    Handles hyphenated names (e.g. 'Hejduk-Clark' -> 'hejdukclark', 'hejduk',
    'clark') and apostrophes (e.g. "O'Neill" -> 'oneill').
    """
    name = name.strip().lower()
    name = re.sub(r"[^\w'-]", "", name)
    variants = {name}

    # Hyphenated variants
    if "-" in name:
        parts = [p.strip("'") for p in name.split("-") if p.strip("'")]
        variants.add("".join(parts))
        variants.update(parts)

    # Apostrophe variants
    if "'" in name:
        variants.add(name.replace("'", ""))

    return [v for v in variants if v]


def resolve_user_idl_kandpal(
    records: list[TextRecord],
    **_kwargs,
) -> dict[str, list[TextRecord]]:
    """
    Canonical-sender resolution for IDL/Mallinckrodt metadata.

    The upstream ``load_idl_records`` already performs cross-mailbox matching
    when ``user_resolver='idl-kandpal'``: it matches each email's ``author`` /
    ``From`` header to an eligible custodian and assigns ``user`` to that
    custodian.  This resolver therefore just groups the records by that user,
    drops empty groups, and prints a diagnostic summary.

    This mirrors the Enron canonical-sender approach: privacy unit = canonical
    sender, and training data is restricted to emails actually sent by that
    user, regardless of whose mailbox they were stored in.
    """
    user_records: dict[str, list[TextRecord]] = defaultdict(list)
    for r in records:
        if r.user:
            user_records[r.user].append(r)

    # Summarize the most common senders per user to aid debugging.
    print("\n  IDL canonical-sender resolution summary:")
    print(f"    Users resolved: {len(user_records)}")
    print(f"    Total records: {sum(len(v) for v in user_records.values())}")

    print("\n  Per-user record distribution:")
    summary = sorted(
        ((u, len(recs), Counter(r.sender for r in recs).most_common(1)[0]) for u, recs in user_records.items()),
        key=lambda x: x[1],
    )
    for user, count, (top_sender, top_n) in summary:
        warn = "  *** LOW COUNT ***" if count < 100 else ""
        print(f"    {user:<35} records={count:>6}  top_sender={top_sender!r:<45} ({top_n}){warn}")

    return dict(user_records)


# ---------------------------------------------------------------------------
# Deduplication strategies
# ---------------------------------------------------------------------------
def dedup_exact(user_records: dict[str, list[TextRecord]]) -> dict[str, list[TextRecord]]:
    """Remove exact body duplicates (per user or globally depending on usage)."""
    out: dict[str, list[TextRecord]] = {}
    for user, records in user_records.items():
        seen: set[str] = set()
        kept: list[TextRecord] = []
        for r in records:
            h = hashlib.md5(r.body.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                kept.append(r)
        out[user] = kept
    return out


def _word_ngram_hashes(text: str, n: int) -> list[int]:
    """Return integer hashes of word-level n-grams (faster set ops than strings)."""
    words = text.split()
    if len(words) < n:
        return []
    return [hash(" ".join(words[i : i + n])) for i in range(len(words) - n + 1)]


def _find_duplicate_ngrams(records: list[TextRecord], n: int, desc: str) -> set[int]:
    """
    Duplicate n-gram detection that clusters exact-duplicate bodies first.
    This is semantically equivalent to processing every record individually but
    much faster when many exact duplicates exist (e.g. company-wide emails).
    """
    body_clusters: dict[str, list[TextRecord]] = defaultdict(list)
    for r in records:
        body_clusters[r.body].append(r)

    ngram_counts: dict[int, int] = defaultdict(int)

    for body, cluster in tqdm(
        body_clusters.items(),
        desc=desc,
        unit="body",
        file=sys.stdout,
        ascii=True,
        miniters=1,
        mininterval=0.5,
    ):
        weight = len(cluster)
        for g in _word_ngram_hashes(body, n):
            ngram_counts[g] += weight

    return {g for g, count in ngram_counts.items() if count > 1}


def _filter_by_ngrams(records: list[TextRecord], duplicates: set[int], n: int) -> list[TextRecord]:
    """Keep records whose n-grams do not intersect the duplicate set."""
    kept = []
    for r in records:
        grams = _word_ngram_hashes(r.body, n)
        if not any(g in duplicates for g in grams):
            kept.append(r)
    return kept


def dedup_ngram(
    user_records: dict[str, list[TextRecord]],
    n: int = 20,
    scope: Literal["user", "global"] = "user",
) -> dict[str, list[TextRecord]]:
    """
    Chua et al.: keep only records whose maximum n-gram repetition count is 1.

    Uses a single-pass set-based algorithm:
      1. Identify all n-grams that appear more than once.
      2. Discard any record that contains at least one duplicate n-gram.
    """
    out: dict[str, list[TextRecord]] = {}

    if scope == "global":
        all_records = [r for recs in user_records.values() for r in recs]
        print(f"  [ngram dedup] Finding duplicate {n}-grams across {len(all_records):,} records ...", flush=True)
        duplicates = _find_duplicate_ngrams(all_records, n, desc="Finding duplicate n-grams")
        print(f"  [ngram dedup] Found {len(duplicates):,} duplicate n-grams", flush=True)

        for user, records in tqdm(
            user_records.items(),
            desc="Filtering records",
            unit="user",
            file=sys.stdout,
            ascii=True,
        ):
            out[user] = _filter_by_ngrams(records, duplicates, n)
    else:
        for user, records in tqdm(
            user_records.items(),
            desc="Deduplicating users",
            unit="user",
            file=sys.stdout,
            ascii=True,
        ):
            duplicates = _find_duplicate_ngrams(records, n, desc=f"User {user[:30]:30s}")
            out[user] = _filter_by_ngrams(records, duplicates, n)

    return out


# ---------------------------------------------------------------------------
# Filtering strategies
# ---------------------------------------------------------------------------
def filter_kandpal(
    user_records: dict[str, list[TextRecord]],
    min_examples: int = 150,
    **_kwargs,
) -> dict[str, list[TextRecord]]:
    out = {u: recs for u, recs in user_records.items() if len(recs) >= min_examples}
    print(f"Users after >={min_examples} filter: {len(out):,}")
    return out


def filter_custom(
    user_records: dict[str, list[TextRecord]],
    min_examples: int = 1,
    remove_forwards: bool = False,
    **_kwargs,
) -> dict[str, list[TextRecord]]:
    out = {}
    for u, recs in user_records.items():
        kept = [r for r in recs if not (remove_forwards and r.is_forward)]
        if len(kept) >= min_examples:
            out[u] = kept
    print(f"Users after custom filter: {len(out):,}")
    return out


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------
def format_text_body_only(record: TextRecord) -> str:
    return record.body


def format_text_headers_body(record: TextRecord) -> str:
    """Canonical headers + body. The sender (`From:`) is intentionally omitted
    from the model input because it directly identifies the user and would
    trivialize user-inference attacks.
    """
    parts = []
    if record.date:
        parts.append(f"Date: {record.date}")
    if record.to:
        parts.append(f"To: {', '.join(record.to)}")
    if record.cc:
        parts.append(f"Cc: {', '.join(record.cc)}")
    if record.subject:
        parts.append(f"Subject: {record.subject}")
    parts.append(record.body)
    return "\n".join(parts)


def format_text_raw_headers_body(record: TextRecord) -> str:
    """Full raw header block + body, with the sender (`From:`) removed.

    Keeping `From:` in the model input would directly reveal the user and
    trivialize user-inference attacks.
    """
    filtered_headers = "\n".join(
        line
        for line in record.raw_headers.splitlines()
        if not line.lower().startswith("from:")
    )
    return f"{filtered_headers}\n\n{record.body}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def record_to_dict(record: TextRecord) -> dict:
    return {
        "sender": record.sender,
        "from": record.sender,
        "to": record.to,
        "cc": record.cc,
        "bcc": record.bcc,
        "subject": record.subject,
        "body": record.body,
        "content_type": record.content_type,
        "date": record.date,
        "is_forward": record.is_forward,
        "profile_fields": record.profile_fields,
        "prepend_profile_fields": record.prepend_profile_fields,
        "text": record.body,
    }


def write_jsonl(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Print stats
    users = len({r["user"] for r in rows})
    print(f"\nOutput -> {output_path}")
    print(f"  Users : {users:,}")
    if rows and "split" in rows[0]:
        splits = Counter(r["split"] for r in rows)
        for split, count in sorted(splits.items()):
            print(f"  {split:<10}: {count:,}")
    print(f"  Total : {len(rows):,}")


def log_user_distribution(user_records: dict[str, list[TextRecord]]) -> None:
    """Print a detailed summary of per-user record counts before filtering."""
    counts = sorted(
        [(user, len(recs)) for user, recs in user_records.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    n_users = len(counts)
    if n_users == 0:
        print("  No users to summarize.")
        return

    values = [c for _, c in counts]
    total = sum(values)
    median = values[n_users // 2] if n_users % 2 else (values[n_users // 2 - 1] + values[n_users // 2]) / 2

    print(f"\n  Per-user record distribution ({n_users:,} users, {total:,} records):")
    print(f"    min    : {values[-1]:,}")
    print(f"    max    : {values[0]:,}")
    print(f"    mean   : {total / n_users:.1f}")
    print(f"    median : {median:.1f}")

    thresholds = [150, 100, 50, 25, 10]
    for t in thresholds:
        below = sum(1 for c in values if c < t)
        print(f"    users with < {t:>3} records: {below:>3} ({100 * below / n_users:.1f}%)")

    print("\n  Top 15 users by record count:")
    for user, c in counts[:15]:
        print(f"    {c:>6}  {user}")

    print("\n  Bottom 15 users by record count:")
    for user, c in counts[-15:]:
        print(f"    {c:>6}  {user}")


def _collect_email_paths(maildir: Path, skip_folders: set[str] | None = None) -> list[Path]:
    """Collect all candidate email file paths under maildir with a progress bar."""
    skip_folders = {f.lower() for f in (skip_folders or set())}
    paths: list[Path] = []
    user_dirs = [d for d in maildir.iterdir() if d.is_dir()]

    for user_dir in tqdm(user_dirs, desc="Scanning user folders", unit="user", file=sys.stdout, ascii=True):
        for folder in user_dir.iterdir():
            if not folder.is_dir():
                continue
            if folder.name.lower() in skip_folders:
                continue
            for email_file in folder.iterdir():
                if safe_path_is_file(email_file):
                    paths.append(email_file)

    return paths


def collect_raw_records(
    maildir: Path,
    max_workers: int | None = None,
    skip_folders: set[str] | None = None,
    min_body_words: int = 5,
    clean_body: bool = True,
) -> list[TextRecord]:
    """Parse all email files under maildir in parallel with a progress bar."""
    print("  Collecting email file paths ...", flush=True)
    email_paths = _collect_email_paths(maildir, skip_folders=skip_folders)
    total_files = len(email_paths)
    print(f"  Found {total_files:,} candidate files to parse", flush=True)

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    chunksize = max(1, total_files // (max_workers * 32))

    records: list[TextRecord] = []
    skipped = 0

    parse_fn = functools.partial(
        parse_email_file, min_body_words=min_body_words, clean_body=clean_body
    )

    print(f"  Parsing with {max_workers} workers (chunksize={chunksize}) ...", flush=True)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for rec in tqdm(
            executor.map(parse_fn, email_paths, chunksize=chunksize),
            total=total_files,
            desc="Parsing emails",
            unit="file",
            file=sys.stdout,
            ascii=True,
            miniters=1,
            mininterval=0.5,
            position=0,
            leave=True,
            disable=False,
        ):
            if rec is None:
                skipped += 1
            else:
                records.append(rec)

    print(f"Scanned {total_files:,} files  |  skipped: {skipped:,}")
    return records


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    "kandpal_strict": {
        "source": "enron",
        "user_resolver": "kandpal",
        "dedup": "exact",
        "cc_bcc_mode": "ignore",
        "filter": "kandpal",
        "text_format": "headers_body",
        "min_examples": 150,
        "min_body_words": 5,
        "remove_forwards": False,
        "ngram_size": 20,
        "ngram_scope": "user",
        "skip_folders": [],
        "clean_body": False,
    },
    "kandpal_raw_nolimit": {
        "source": "enron",
        "user_resolver": "kandpal",
        "dedup": "exact",
        "cc_bcc_mode": "ignore",
        "filter": "kandpal",
        "text_format": "raw_headers_body",
        "min_examples": 150,
        "min_body_words": 5,
        "remove_forwards": False,
        "ngram_size": 20,
        "ngram_scope": "user",
        "skip_folders": [],
        "clean_body": False,
    },
    "mallinckrodt": {
        "source": "idl",
        "data_dir": "datasets/Mallinckrodt_Litigation_Documents",
        "user_resolver": "idl-kandpal",
        "dedup": "exact",
        "cc_bcc_mode": "ignore",
        "filter": "custom",
        "text_format": "headers_body",
        "min_examples": 100,
        "min_body_words": 5,
        "remove_forwards": False,
        "clean_body": False,
    },
}

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(cfg: dict, maildir: Path, output: Path) -> None:
    source = cfg.get("source", "enron")

    # 1. Parse / load
    if source == "idl":
        print("\n[1/6] Loading IDL Mallinckrodt records ...")
        data_dir = Path(cfg.get("data_dir", "datasets/Mallinckrodt_Litigation_Documents"))
        records = load_idl_records(
            data_dir,
            max_users=cfg.get("max_users"),
            min_examples=cfg.get("min_examples", 100),
            min_body_words=cfg.get("min_body_words", _PREPROCESS_CFG.min_body_words),
            max_examples_per_user=cfg.get("max_examples_per_user"),
            max_files=cfg.get("max_files"),
            user_resolver=cfg.get("user_resolver", "identity"),
        )
    else:
        print("\n[1/6] Parsing raw emails ...")
        skip_folders = cfg.get("skip_folders")
        if skip_folders:
            print(f"  Skipping folders: {', '.join(sorted(skip_folders))}")
        else:
            print("  Skipping no folders")
        records = collect_raw_records(
            maildir,
            skip_folders=set(skip_folders) if skip_folders else None,
            min_body_words=cfg.get("min_body_words", _PREPROCESS_CFG.min_body_words),
            clean_body=cfg.get("clean_body", True),
        )

    # 2. Expand CC/BCC if requested (email-only)
    cc_bcc_mode = cfg.get("cc_bcc_mode", "ignore")
    if cc_bcc_mode == "expand":
        print("[2/6] Expanding CC/BCC recipients ...")
        records = expand_ccbcc(records)
    else:
        print("[2/6] Keeping single record per original sample ...")

    # 3. Resolve users
    resolver_name = cfg.get("user_resolver", "chua")
    print(f"[3/6] Resolving users ({resolver_name}) ...")
    resolver_map: dict[str, Callable] = {
        "kandpal": resolve_user_kandpal,
        "idl-kandpal": resolve_user_idl_kandpal,
        "custom": resolve_user_custom,
        "identity": resolve_user_identity,
    }
    resolver = resolver_map[resolver_name]
    user_records = resolver(
        records,
        maildir=maildir,
        map_file=cfg.get("user_map"),
        regex_pattern=cfg.get("user_regex"),
        keep_all=cfg.get("keep_all", False),
        collapse_aliases=cfg.get("collapse_aliases", False),
    )
    print(f"Unique users before filtering: {len(user_records):,}")

    # 4. Deduplicate
    dedup_name = cfg.get("dedup", "exact")
    print(f"[4/6] Deduplicating ({dedup_name}) ...")
    if dedup_name == "exact":
        user_records = dedup_exact(user_records)
    elif dedup_name == "ngram":
        user_records = dedup_ngram(
            user_records,
            n=cfg.get("ngram_size", 20),
            scope=cfg.get("ngram_scope", "user"),
        )
    else:
        print("  (skipping deduplication)")

    # 5. Filter
    log_user_distribution(user_records)
    filter_name = cfg.get("filter", "custom")
    print(f"[5/6] Filtering ({filter_name}) ...")
    filter_map: dict[str, Callable] = {
        "kandpal": filter_kandpal,
        "custom": filter_custom,
    }
    user_records = filter_map[filter_name](
        user_records,
        min_examples=cfg.get("min_examples", 1),
        remove_forwards=cfg.get("remove_forwards", False),
    )

    # 6. Format text and write unsplit records
    fmt_name = cfg.get("text_format", "body_only")
    print(f"[6/6] Formatting text ({fmt_name}) ...")
    fmt_map = {
        "body_only": format_text_body_only,
        "headers_body": format_text_headers_body,
        "raw_headers_body": format_text_raw_headers_body,
    }
    fmt_fn = fmt_map.get(fmt_name, format_text_body_only)

    rows: list[dict] = []
    for user, recs in user_records.items():
        for r in recs:
            row = record_to_dict(r)
            row["user"] = user
            row["text"] = fmt_fn(r)
            rows.append(row)

    write_jsonl(rows, output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess text data (Enron emails or Mallinckrodt IDL) for user-inference auditing."
    )
    p.add_argument("--maildir", type=Path, default=Path("datasets/enron_raw/maildir"),
                   help="Directory containing raw email files (Enron source).")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Directory containing downloaded IDL metadata CSVs (overrides preset default).")
    p.add_argument("--output", type=Path, default=None, help="Output JSONL path (default: derived from --preset).")
    p.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="kandpal_strict",
        help="Use a built-in paper preset (default: kandpal_strict).",
    )
    p.add_argument(
        "--source",
        choices=["enron", "idl"],
        default=None,
        help="Input data source (default: inferred from preset).",
    )

    # Strategy overrides
    p.add_argument("--user-resolver", choices=["kandpal", "idl-kandpal", "custom", "identity"], default=None)
    p.add_argument("--user-map", type=Path, default=None, help="CSV file: email,user_id for custom resolver.")
    p.add_argument("--user-regex", type=str, default=None, help="Regex to extract user ID from sender for custom resolver.")

    p.add_argument("--dedup", choices=["exact", "ngram", "none"], default=None)
    p.add_argument("--ngram-size", type=int, default=None)
    p.add_argument("--ngram-scope", choices=["user", "global"], default=None)

    p.add_argument("--cc-bcc-mode", choices=["ignore", "expand"], default=None)

    p.add_argument("--filter", choices=["kandpal", "custom"], default=None)
    p.add_argument("--min-examples", type=int, default=None)
    p.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="For IDL sources, limit preprocessing to the top-N custodians (useful for quick tests).",
    )
    p.add_argument(
        "--max-examples-per-user",
        type=int,
        default=None,
        help="For IDL sources, cap records per custodian (useful for quick tests).",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="For IDL sources, limit preprocessing to the first N CSV files (useful for quick tests).",
    )
    p.add_argument("--min-body-words", type=int, default=None, help="Minimum number of words in the cleaned email body (0 to disable).")
    p.add_argument("--remove-forwards", action="store_true", default=None)

    p.add_argument(
        "--text-format",
        choices=["body_only", "headers_body", "raw_headers_body"],
        default=None,
    )
    p.add_argument(
        "--clean-body",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Clean email bodies (remove forwards, quoted replies, stray headers). "
             "The default depends on the selected preset.",
    )
    p.add_argument(
        "--skip-folders",
        type=str,
        default=None,
        help="Comma-separated list of folder names to skip during scanning (overrides preset).",
    )
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    cfg: dict = {"seed": args.seed}

    if args.preset:
        preset = PRESETS[args.preset].copy()
        cfg.update(preset)
        print(f"Loaded preset: {args.preset}")

    # CLI overrides
    overrides = [
        ("source", args.source),
        ("data_dir", args.data_dir),
        ("user_resolver", args.user_resolver),
        ("user_map", args.user_map),
        ("user_regex", args.user_regex),
        ("dedup", args.dedup),
        ("ngram_size", args.ngram_size),
        ("ngram_scope", args.ngram_scope),
        ("cc_bcc_mode", args.cc_bcc_mode),
        ("filter", args.filter),
        ("min_examples", args.min_examples),
        ("max_users", args.max_users),
        ("max_examples_per_user", args.max_examples_per_user),
        ("max_files", args.max_files),
        ("min_body_words", args.min_body_words),
        ("remove_forwards", args.remove_forwards),
        ("text_format", args.text_format),
        ("clean_body", args.clean_body),
        ("seed", args.seed),
    ]
    for key, val in overrides:
        if val is not None:
            cfg[key] = val

    if args.skip_folders is not None:
        cfg["skip_folders"] = [f.strip() for f in args.skip_folders.split(",") if f.strip()]

    if not cfg.get("preset") and not cfg.get("user_resolver"):
        # Default behaviour close to the original script
        cfg.setdefault("source", "enron")
        cfg.setdefault("user_resolver", "chua")
        cfg.setdefault("dedup", "exact")
        cfg.setdefault("cc_bcc_mode", "ignore")
        cfg.setdefault("filter", "custom")
        cfg.setdefault("text_format", "body_only")
        cfg.setdefault("min_examples", 28)
        cfg.setdefault("remove_forwards", False)
        cfg.setdefault("skip_folders", [])
        cfg.setdefault("clean_body", True)

    return cfg


def _default_output_for_preset(preset: str) -> Path:
    """Return the default unsplit output path for a built-in preset."""
    mapping = {
        "kandpal_strict": "preprocessed/enron_unsplit.jsonl",
        "kandpal_raw_nolimit": "preprocessed/enron_raw_nolimit_unsplit.jsonl",
        "mallinckrodt": "preprocessed/mallinckrodt_unsplit.jsonl",
    }
    return Path(mapping.get(preset, "preprocessed/enron_processed_unsplit.jsonl"))


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    source = cfg.get("source", "enron")

    if source == "idl":
        data_dir = Path(cfg.get("data_dir", "datasets/Mallinckrodt_Litigation_Documents"))
        if not data_dir.exists():
            sys.exit(f"ERROR: IDL data directory not found at '{data_dir}'.")
    else:
        if not args.maildir.exists():
            sys.exit(f"ERROR: maildir not found at '{args.maildir}'.")

    output = args.output or _default_output_for_preset(args.preset)
    run_pipeline(cfg, args.maildir, output)


if __name__ == "__main__":
    main()
