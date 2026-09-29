"""User-level PII graph extraction from email-style records.

The pipeline combines GLiNER for zero-shot named-entity recognition, GLiREL for
zero-shot relation extraction, and Stanza for coreference resolution.  Its main
product is a set of (user, relation, PII-value) triples that can be used to
audit or attack fine-tuned language models.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import stanza
import torch
from gliner import GLiNER
from glirel import GLiREL
from tqdm import tqdm

from utils.enron_pii_utils import (
    _phone_match_variants,
    canonical_phone_digits,
    is_placeholder_phone,
    is_toll_free_phone,
    normalize_address,
    normalize_email,
)

DEFAULT_SCHEMA_PATH = Path(__file__).parent.parent / "configs" / "pii_graph_schema.json"

#: Natural-language relation queries used by GLiREL.  The keys are schema
#: relation labels; the values are the strings passed to the model.
DEFAULT_RELATION_QUERIES: dict[str, str] = {
    "has_email_address": "email address",
    "has_phone_number": "phone number",
    "has_age": "age",
    "born_on_date": "date of birth",
    "lives_in_city": "city of residence",
    "lives_in_country": "country of residence",
    "country_of_origin": "nationality",
    "works_at_organization": "employer",
    "has_job_title": "job title",
    "has_home_address": "home address",
}

def resolve_device(device: str) -> str:
    """Resolve a device string to a concrete torch device.

    Supports ``auto`` (CUDA if available, otherwise CPU), ``cuda``, and
    ``cpu``. Unknown strings are passed through so that downstream libraries
    can raise their own errors.
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device in {"cpu", "cuda"}:
        return device
    return device


#: Pronouns that refer to the author of a record.
_FIRST_PERSON_PRONOUNS = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
}


#: Values that are too generic to be useful PII and are common false positives.
_GENERIC_PII_BLOCKLIST: dict[str, set[str]] = {
    "EMAIL_ADDRESS": {
        "enron.com",
        "ect.enron.com",
        "enron.net",
        "aol.com",
        "yahoo.com",
        "hotmail.com",
        "gmail.com",
        "none",
    },
    "PHONE_NUMBER": {
        "",
        "none",
        "n/a",
        "na",
    },
    "DATE": {"mon", "tue", "wed", "thu", "fri", "sat", "sun"},
}


def load_word_set(path: str | os.PathLike[str]) -> set[str]:
    """Load a newline-delimited word list, skipping blank lines and comments."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Word list not found: {path}\n"
            "Download or provide the file and see README.md Setup step 5 for sources "
            "and licensing notes."
        )
    with path.open("r", encoding="utf-8") as fh:
        return {
            line.strip().lower()
            for line in fh
            if line.strip() and not line.strip().startswith("#")
        }


def _normalize_for_match(value: str) -> str:
    """Lower-case, remove non-alphanumeric chars, and collapse whitespace."""
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return " ".join(normalized.split())


def _normalize_text_for_nlp(text: str) -> str:
    """Normalize OCR and metadata text before NER/coref.

    Replaces non-breaking spaces, zero-width spaces, directional marks, and
    the byte-order mark with a normal space, then collapses repeated
    whitespace. This prevents Stanza coref from hitting "blank word" errors
    on malformed input.
    """
    text = re.sub(r"[\xa0\u200b-\u200f\ufeff]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _looks_like_email(value: str) -> bool:
    """Return True if ``value`` has a local part and a dotted domain."""
    parts = value.split("@", 1)
    if len(parts) != 2:
        return False
    local, domain = parts
    return bool(local.strip()) and "." in domain and len(domain.split(".")[-1]) >= 2


def is_valid_pii_value(
    entity: EntitySpan,
    known_users: set[str] | None = None,
    current_user: str | None = None,
    common_words: set[str] | None = None,
    job_titles: set[str] | None = None,
) -> bool:
    """Return True if ``entity`` looks like a real, attributable PII value.

    This filters obvious false positives (domain-only emails, toll-free
    prefixes, placeholder dates), cross-user email collisions, generic
    organization nouns, and non-job-title strings.
    """
    label = entity.label.upper()
    raw = entity.text.strip()
    value = canonical_value(entity)

    if not raw or len(raw) < 2:
        return False

    if label == "EMAIL_ADDRESS":
        if not _looks_like_email(value):
            return False
        # Only keep the user's own email address; forwarded/mentioned emails
        # of other people are too common in email threads.
        if current_user is not None:
            own_email = normalize_email(str(current_user))
            if value != own_email:
                return False
        return value not in _GENERIC_PII_BLOCKLIST.get("EMAIL_ADDRESS", set())

    if label == "PHONE_NUMBER":
        # GLiNER often mislabels timestamps such as "09:41:00 -0700" as phone
        # numbers. Any colon in the raw text is a cheap and reliable signal.
        if ":" in raw:
            return False
        digits = canonical_phone_digits(raw)
        if not digits or len(digits) < 7:
            return False
        if is_placeholder_phone(raw) or is_toll_free_phone(raw):
            return False
        return value.lower() not in _GENERIC_PII_BLOCKLIST.get("PHONE_NUMBER", set())

    if label == "DATE":
        # Dates are only kept when GLiREL predicts a relation; the fallback is
        # too noisy because every email header contains a date.
        return False

    if label == "ORGANIZATION":
        # Drop very short or punctuation-only values.
        alpha = re.sub(r"[^a-zA-Z]", "", raw)
        if len(alpha) < 2:
            return False
        # Drop values made entirely of common English words (e.g. "trading",
        # "east desk"), which are usually mislabels rather than real employers.
        if common_words:
            tokens = [t for t in re.findall(r"[a-z]+", value) if len(t) > 1]
            if tokens and all(t in common_words for t in tokens):
                return False
        return True

    if label == "JOB_TITLE":
        # Drop very short or punctuation-only values.
        alpha = re.sub(r"[^a-zA-Z]", "", raw)
        if len(alpha) < 2:
            return False
        if job_titles:
            normalized = _normalize_for_match(value)
            if normalized not in job_titles:
                return False
        return True

    if label in {"CITY", "COUNTRY", "NATIONALITY", "HOME_ADDRESS"}:
        # Drop very short or punctuation-only values.
        alpha = re.sub(r"[^a-zA-Z]", "", raw)
        return len(alpha) >= 2

    if label == "AGE":
        try:
            age = int(re.sub(r"[^0-9]", "", raw))
        except ValueError:
            return False
        return 0 < age < 130

    if label == "NAME":
        # Canonical names are derived from verified user aliases, so they are
        # valid by construction.
        return True

    return True


@dataclass(frozen=True)
class EntitySpan:
    """A named entity mention extracted from raw text."""

    label: str
    text: str
    start: int
    end: int
    score: float = 0.0
    chunk_offset: int = 0


@dataclass(frozen=True)
class CorefMention:
    """A single mention inside a coreference cluster."""

    start: int
    end: int
    text: str


@dataclass(frozen=True)
class CorefCluster:
    """A coreference cluster linking multiple mentions to the same referent."""

    cluster_id: int
    mentions: tuple[CorefMention, ...]
    representative: CorefMention | None = None


@dataclass(frozen=True)
class RelationPrediction:
    """A raw relation triple returned by a relation-extraction model."""

    head_start: int
    head_end: int
    head_text: str
    tail_start: int
    tail_end: int
    tail_text: str
    relation: str
    score: float


class PIIGraphSchema:
    """Wrapper around the configurable entity/relation schema JSON."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        path = Path(path or DEFAULT_SCHEMA_PATH)
        with path.open("r", encoding="utf-8") as fh:
            self._raw: dict[str, Any] = json.load(fh)

        self.name: str = self._raw.get("schema_name", "")
        self.entity_labels: list[str] = list(self._raw.get("entity_labels", {}).keys())
        self.relation_labels: list[str] = list(self._raw.get("relation_labels", {}).keys())
        self._relations: dict[str, dict[str, Any]] = self._raw.get("relation_labels", {})
        self.thresholds: dict[str, float] = self._raw.get("default_thresholds", {})
        self._relation_queries: dict[str, str] = {
            **DEFAULT_RELATION_QUERIES,
            **self._raw.get("relation_queries", {}),
        }

    def allowed_tails(self, relation: str) -> list[str]:
        """Return the tail entity labels allowed for ``relation``."""
        return list(self._relations.get(relation, {}).get("allowed_tail", []))

    def allowed_heads(self, relation: str) -> list[str]:
        """Return the head entity labels allowed for ``relation``."""
        return list(self._relations.get(relation, {}).get("allowed_head", ["PERSON"]))

    def relation_query(self, relation: str) -> str:
        """Return the natural-language query string for a schema relation."""
        return self._relation_queries.get(relation, relation.replace("_", " "))

    def to_dict(self) -> dict[str, Any]:
        """Return the raw schema dictionary."""
        return dict(self._raw)


class NERExtractor(ABC):
    """Abstract named-entity recogniser."""

    @abstractmethod
    def extract(
        self,
        text: str,
        schema: PIIGraphSchema,
        threshold: float | None = None,
    ) -> list[EntitySpan]:
        """Return all entity spans found in ``text``."""


class CorefResolver(ABC):
    """Abstract coreference resolver."""

    @abstractmethod
    def resolve(self, text: str) -> list[CorefCluster]:
        """Return coreference clusters found in ``text``."""


class RelationExtractor(ABC):
    """Abstract relation extractor."""

    @abstractmethod
    def extract(
        self,
        text: str,
        schema: PIIGraphSchema,
        threshold: float | None = None,
        entities: list[EntitySpan] | None = None,
    ) -> list[RelationPrediction]:
        """Return relation predictions for ``text``.

        Args:
            entities: Optional entity spans that restrict candidate head/tail
                pairs for models (like GLiREL) that require them.
        """


class NullRelationExtractor(RelationExtractor):
    """No-op relation extractor; rely purely on fallback heuristics."""

    def extract(
        self,
        text: str,
        schema: PIIGraphSchema,
        threshold: float | None = None,
        entities: list[EntitySpan] | None = None,
    ) -> list[RelationPrediction]:
        return []


class GLiNERNER(NERExtractor):
    """GLiNER-based zero-shot NER."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        device = resolve_device(device)
        self.model = GLiNER.from_pretrained(model_path)
        self.device = device
        if device != "cpu" and hasattr(self.model, "to"):
            self.model.to(device)

    def extract(
        self,
        text: str,
        schema: PIIGraphSchema,
        threshold: float | None = None,
    ) -> list[EntitySpan]:
        if threshold is None:
            threshold = schema.thresholds.get("gliner_entity", 0.5)

        spans: list[EntitySpan] = []
        for chunk, offset in chunk_text(text):
            predictions = self.model.predict_entities(chunk, schema.entity_labels, threshold)
            for pred in predictions:
                spans.append(
                    EntitySpan(
                        label=pred["label"],
                        text=pred["text"],
                        start=pred["start"] + offset,
                        end=pred["end"] + offset,
                        score=float(pred.get("score", 0.0)),
                        chunk_offset=offset,
                    )
                )
        return spans


class GLiRELExtractor(RelationExtractor):
    """GLiREL-based zero-shot relation extraction.

    GLiREL is end-to-end: it predicts both entity spans and relation labels.
    This wrapper maps the model's natural-language labels back to the schema's
    relation labels and reports character-based positions.
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        device = resolve_device(device)
        self.model = GLiREL._from_pretrained(
            model_id=model_path,
            revision=None,
            cache_dir=None,
            force_download=False,
            proxies=None,
            resume_download=False,
            local_files_only=True,
            token=None,
            map_location=device if device != "cpu" else "cpu",
            strict=False,
        )
        self.device = device

    def extract(
        self,
        text: str,
        schema: PIIGraphSchema,
        threshold: float | None = None,
        entities: list[EntitySpan] | None = None,
    ) -> list[RelationPrediction]:
        if threshold is None:
            threshold = schema.thresholds.get("glirel_relation", 0.5)

        query_to_relation = {
            schema.relation_query(rel): rel for rel in schema.relation_labels
        }
        labels = list(query_to_relation.keys())
        if not labels:
            return []

        entities = entities or []
        predictions: list[RelationPrediction] = []
        for chunk, offset in chunk_text(text):
            chunk_entities = [e for e in entities if e.start >= offset and e.end <= offset + len(chunk)]
            ner = [(e.start - offset, e.end - offset, e.label) for e in chunk_entities]
            if len(ner) < 2:
                # GLiREL's collator requires at least two candidate entities.
                continue
            raw = self.model.predict_relations(
                chunk,
                labels,
                threshold=threshold,
                ner=ner,
            )
            for pred in raw:
                relation = query_to_relation.get(pred.get("label"))
                if relation is None:
                    continue
                predictions.append(
                    RelationPrediction(
                        head_start=int(pred["head_pos"][0]) + offset,
                        head_end=int(pred["head_pos"][1]) + offset,
                        head_text=pred["head_text"],
                        tail_start=int(pred["tail_pos"][0]) + offset,
                        tail_end=int(pred["tail_pos"][1]) + offset,
                        tail_text=pred["tail_text"],
                        relation=relation,
                        score=float(pred.get("score", 0.0)),
                    )
                )
        return predictions


class StanzaCorefResolver(CorefResolver):
    """Stanza neural coreference resolver."""

    def __init__(self, model_dir: str, device: str = "cpu") -> None:
        os.environ["STANZA_RESOURCES_DIR"] = model_dir
        device = resolve_device(device)
        self.nlp = stanza.Pipeline(
            "en",
            dir=model_dir,
            processors="tokenize,coref",
            use_gpu=(device != "cpu"),
            download_method=stanza.DownloadMethod.REUSE_RESOURCES,
        )

    def resolve(self, text: str) -> list[CorefCluster]:
        try:
            doc = self.nlp(text)
        except Exception as exc:
            warnings.warn(
                f"Stanza coref failed on text of length {len(text)}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return []
        clusters: list[CorefCluster] = []
        if not hasattr(doc, "coref"):
            return clusters

        for idx, chain in enumerate(doc.coref):
            mentions: list[CorefMention] = []
            representative: CorefMention | None = None
            rep_index = getattr(chain, "representative_index", 0)
            for m_idx, mention in enumerate(chain.mentions):
                sent = doc.sentences[mention.sentence]
                words = sent.words[mention.start_word : mention.end_word]
                if not words:
                    continue
                start = words[0].start_char
                end = words[-1].end_char
                cm = CorefMention(start=start, end=end, text=text[start:end])
                mentions.append(cm)
                if m_idx == rep_index:
                    representative = cm
            if mentions:
                clusters.append(
                    CorefCluster(
                        cluster_id=idx,
                        mentions=tuple(mentions),
                        representative=representative,
                    )
                )
        return clusters


class RuleCorefResolver(CorefResolver):
    """Lightweight rule-based coreference resolver.

    Returns singleton clusters for first-person pronouns so that downstream
    linking can still resolve them to the record author.
    """

    _PRONOUN_RE = re.compile(
        r"\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b",
        re.IGNORECASE,
    )

    def resolve(self, text: str) -> list[CorefCluster]:
        clusters: list[CorefCluster] = []
        for match in self._PRONOUN_RE.finditer(text):
            cm = CorefMention(
                start=match.start(),
                end=match.end(),
                text=match.group(0),
            )
            clusters.append(
                CorefCluster(
                    cluster_id=len(clusters),
                    mentions=(cm,),
                    representative=cm,
                )
            )
        return clusters


class NullCorefResolver(CorefResolver):
    """No-op coreference resolver."""

    def resolve(self, text: str) -> list[CorefCluster]:
        return []


@dataclass
class UserLinkResult:
    """Result of trying to attribute an entity mention to a user."""

    user_id: str | None = None
    canonical_name: str | None = None
    via: str | None = None  # e.g. "pronoun", "name_match", "coref"


class UserLinker:
    """Link PERSON mentions to user identities."""

    def __init__(self, aliases: dict[str, list[str]] | None = None) -> None:
        """Create a linker.

        Args:
            aliases: Optional pre-computed user aliases.  If omitted, aliases
                are derived from each record's ``user`` field on the fly.
        """
        self._aliases = aliases or {}

    @staticmethod
    def _name_parts(user_id: str) -> set[str]:
        """Extract lower-case name tokens from a user identifier."""
        user_id = str(user_id).strip().lower()
        if "@" in user_id:
            user_id = user_id.split("@", 1)[0]
        parts = re.split(r"[._\-\s]+", user_id)
        return {p for p in parts if len(p) > 1}

    def aliases_for(self, user_id: str, record: dict[str, Any] | None = None) -> set[str]:
        """Return the alias set for ``user_id``."""
        if user_id in self._aliases:
            return set(self._aliases[user_id])

        aliases: set[str] = set()
        aliases.update(self._name_parts(user_id))

        if record is not None:
            for key in ("name", "display_name", "author_name", "sender"):
                value = record.get(key)
                if value:
                    aliases.add(str(value).strip().lower())
                    aliases.update(self._name_parts(str(value)))

            # Parse a From: header if one slipped through preprocessing.
            text = record.get("text", "")
            from_match = re.search(
                r"^\s*From:\s*(.+?)(?:\s+\[mailto:|\s+<|$)",
                text,
                re.IGNORECASE | re.MULTILINE,
            )
            if from_match:
                name = from_match.group(1).strip()
                aliases.add(name.lower())
                aliases.update(self._name_parts(name))

        return aliases

    def link(
        self,
        entity: EntitySpan,
        user_id: str,
        clusters: list[CorefCluster],
        record: dict[str, Any] | None = None,
    ) -> UserLinkResult:
        """Return the user identity this entity refers to, if any."""
        if entity.label != "PERSON":
            return UserLinkResult()

        text_lower = entity.text.strip().lower()

        # First-person pronouns always refer to the author.
        if text_lower in _FIRST_PERSON_PRONOUNS:
            return UserLinkResult(user_id=user_id, canonical_name=text_lower, via="pronoun")

        # Coref-resolved name.
        for cluster in clusters:
            if any(
                entity.start < mention.end and mention.start < entity.end
                for mention in cluster.mentions
            ):
                rep = cluster.representative
                if rep is not None:
                    rep_lower = rep.text.strip().lower()
                    if self._matches_aliases(rep_lower, user_id, record):
                        return UserLinkResult(
                            user_id=user_id,
                            canonical_name=rep.text.strip(),
                            via="coref",
                        )

        # Direct name match against the user's aliases.
        if self._matches_aliases(text_lower, user_id, record):
            return UserLinkResult(
                user_id=user_id,
                canonical_name=entity.text.strip(),
                via="name_match",
            )

        return UserLinkResult()

    def _matches_aliases(
        self,
        text: str,
        user_id: str,
        record: dict[str, Any] | None = None,
    ) -> bool:
        aliases = self.aliases_for(user_id, record)
        if text in aliases:
            return True
        text_parts = self._name_parts(text)
        # Require at least two overlapping name parts to reduce false positives.
        return len(text_parts) >= 2 and len(text_parts & aliases) >= 2


def build_user_aliases(records: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    """Build a global alias map from record metadata.

    Aliases are derived from the ``user``/``user_id``/``author`` identifier and
    from any ``name``/``display_name``/``author_name``/``sender`` fields. This
    is intentionally conservative: it does *not* parse ``From:`` headers inside
    the text, because forwarded emails would pollute one user with another's
    aliases.
    """
    aliases: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        user = rec.get("user") or rec.get("user_id") or rec.get("author")
        if not user:
            continue
        user_str = str(user)
        aliases[user_str].add(user_str.strip().lower())
        aliases[user_str].update(UserLinker._name_parts(user_str))
        for key in ("name", "display_name", "author_name", "sender"):
            value = rec.get(key)
            if value:
                v = str(value).strip().lower()
                aliases[user_str].add(v)
                aliases[user_str].update(UserLinker._name_parts(v))
    return aliases


def canonical_value(entity: EntitySpan) -> str:
    """Return a canonical string value for an entity mention."""
    raw = entity.text.strip()
    label = entity.label.upper()
    if label == "EMAIL_ADDRESS":
        return normalize_email(raw)
    if label in {"PHONE_NUMBER"}:
        digits = canonical_phone_digits(raw)
        return digits if digits else raw.lower()
    if label == "HOME_ADDRESS":
        return normalize_address(raw)
    if label in {"ORGANIZATION", "NAME"}:
        return re.sub(r"\s+", " ", raw.lower())
    return raw.lower()


def node_id_for_entity(entity: EntitySpan) -> str:
    """Stable node id for a PII value node."""
    return f"{entity.label}:{canonical_value(entity)}"


def user_node_id(user_id: str) -> str:
    return f"USER:{user_id}"


@dataclass
class PIIGraph:
    """A simple user-level PII graph."""

    schema: PIIGraphSchema
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[dict[str, Any]] = field(default_factory=list)
    _edge_index: dict[tuple[str, str, str], int] = field(
        default_factory=dict, init=False, repr=False
    )

    def add_user(self, user_id: str) -> str:
        nid = user_node_id(user_id)
        if nid not in self.nodes:
            self.nodes[nid] = {
                "id": nid,
                "type": "USER",
                "label": user_id,
            }
        return nid

    def add_entity(self, entity: EntitySpan) -> str:
        nid = node_id_for_entity(entity)
        if nid not in self.nodes:
            self.nodes[nid] = {
                "id": nid,
                "type": entity.label,
                "label": entity.text.strip(),
                "canonical_value": canonical_value(entity),
            }
        return nid

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        score: float,
        provenance: dict[str, Any],
    ) -> None:
        """Add an edge, keeping only the highest score for duplicate triples."""
        key = (source_id, target_id, relation)
        existing_idx = self._edge_index.get(key)
        if existing_idx is None:
            self.edges.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "relation": relation,
                    "score": score,
                    "provenance": provenance,
                }
            )
            self._edge_index[key] = len(self.edges) - 1
        elif score > self.edges[existing_idx]["score"]:
            self.edges[existing_idx]["score"] = score
            self.edges[existing_idx]["provenance"] = provenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema.to_dict(),
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
            "stats": {
                "num_nodes": len(self.nodes),
                "num_edges": len(self.edges),
                "num_users": sum(1 for n in self.nodes.values() if n["type"] == "USER"),
            },
        }


class PIIGraphBuilder:
    """Orchestrate NER, coreference, relation extraction, and user linking."""

    def __init__(
        self,
        ner: NERExtractor,
        coref: CorefResolver,
        relation_extractor: RelationExtractor,
        schema: PIIGraphSchema,
        user_linker: UserLinker | None = None,
        require_ner_tail: bool = True,
        known_users: set[str] | None = None,
        fallback_proximity_chars: int = 300,
        common_words: set[str] | None = None,
        job_titles: set[str] | None = None,
    ) -> None:
        self.ner = ner
        self.coref = coref
        self.relation_extractor = relation_extractor
        self.schema = schema
        self.user_linker = user_linker or UserLinker()
        self.require_ner_tail = require_ner_tail
        self.known_users = known_users or set()
        self.fallback_proximity_chars = fallback_proximity_chars
        self.common_words = common_words or set()
        self.job_titles = job_titles or set()

    def process_record(
        self,
        record: dict[str, Any],
        graph: PIIGraph | None = None,
    ) -> PIIGraph:
        """Extract PII triples from ``record`` and add them to ``graph``."""
        if graph is None:
            graph = PIIGraph(schema=self.schema)

        user_id = record.get("user") or record.get("user_id") or record.get("author")
        if not user_id:
            return graph

        text = record.get("text", "")
        if not text.strip():
            return graph
        text = _normalize_text_for_nlp(text)

        record_id = record.get("record_id", record.get("id"))
        provenance_base: dict[str, Any] = {
            "user_id": user_id,
            "record_id": record_id,
        }

        entities = self.ner.extract(text, self.schema)
        try:
            clusters = self.coref.resolve(text)
        except Exception as exc:
            snippet = text[:200].replace("\n", " ")
            warnings.warn(
                f"Coref failed for user={user_id} record={record_id}: {exc} "
                f"(snippet: {snippet!r})",
                RuntimeWarning,
                stacklevel=2,
            )
            clusters = []
        relations = self.relation_extractor.extract(text, self.schema, entities=entities)

        # Map each PERSON entity to a user, if possible.
        entity_to_user: dict[tuple[int, int], UserLinkResult] = {}
        for entity in entities:
            if entity.label == "PERSON":
                entity_to_user[(entity.start, entity.end)] = self.user_linker.link(
                    entity, user_id, clusters, record
                )

        # Add has_name edges for every PERSON mention successfully linked to
        # the record author. The canonical name value is the user ID string,
        # which avoids OCR variants like "Kate" vs "Muhlenkamp, Kate".
        seen_name_edges: set[tuple[str, str]] = set()
        for link in entity_to_user.values():
            if not link.user_id:
                continue
            user_nid = graph.add_user(link.user_id)
            name_entity = EntitySpan(
                label="NAME",
                text=link.user_id,
                start=0,
                end=0,
                score=1.0,
            )
            name_nid = graph.add_entity(name_entity)
            key = (user_nid, name_nid)
            if key in seen_name_edges:
                continue
            seen_name_edges.add(key)
            graph.add_edge(
                source_id=user_nid,
                target_id=name_nid,
                relation="has_name",
                score=1.0,
                provenance={**provenance_base, "name_source": "user_link"},
            )

        # Resolve relation predictions to graph edges.
        for rel in relations:
            link = self._resolve_head(rel, entities, entity_to_user)
            if link is None:
                continue

            tail_entity = self._resolve_tail(rel, entities)
            if tail_entity is None and self.require_ner_tail:
                continue

            if tail_entity is None:
                # Infer tail label from the relation schema.
                allowed_tails = self.schema.allowed_tails(rel.relation)
                tail_label = allowed_tails[0] if allowed_tails else "UNKNOWN"
                tail_entity = EntitySpan(
                    label=tail_label,
                    text=rel.tail_text,
                    start=rel.tail_start,
                    end=rel.tail_end,
                    score=rel.score,
                )

            user_nid = graph.add_user(user_id)
            tail_nid = graph.add_entity(tail_entity)
            graph.add_edge(
                source_id=user_nid,
                target_id=tail_nid,
                relation=rel.relation,
                score=rel.score,
                provenance={**provenance_base, "head_text": rel.head_text},
            )

        # Also attach high-confidence PII mentions that did not receive a GLiREL
        # relation but are strongly associated with the author (e.g. a signature
        # line "Phone: ...").
        for entity in entities:
            if entity.label == "PERSON":
                continue
            relation_guess = self._relation_for_entity(entity)
            if relation_guess is None:
                continue
            if not is_valid_pii_value(
                entity,
                known_users=self.known_users,
                current_user=user_id,
                common_words=self.common_words,
                job_titles=self.job_titles,
            ):
                continue
            # Only attach if a user-linked PERSON mention is close in the text.
            distance = self._nearest_user_mention_distance(
                entity, entities, entity_to_user
            )
            if distance is None or distance > self.fallback_proximity_chars:
                continue
            user_nid = graph.add_user(user_id)
            tail_nid = graph.add_entity(entity)
            graph.add_edge(
                source_id=user_nid,
                target_id=tail_nid,
                relation=relation_guess,
                score=entity.score,
                provenance={**provenance_base, "fallback": True},
            )

        return graph

    def _resolve_head(
        self,
        rel: RelationPrediction,
        entities: list[EntitySpan],
        entity_to_user: dict[tuple[int, int], UserLinkResult],
    ) -> UserLinkResult | None:
        """Return the user linked to a relation head span, if any."""
        for entity in entities:
            if entity.label != "PERSON":
                continue
            if rel.head_start < entity.end and entity.start < rel.head_end:
                link = entity_to_user.get((entity.start, entity.end))
                if link and link.user_id:
                    return link
        # Check first-person pronouns in the raw head text.
        head_text = rel.head_text.strip().lower()
        if head_text in _FIRST_PERSON_PRONOUNS:
            # We need the user_id; return a placeholder and fill it in caller.
            return UserLinkResult(user_id="__SELF__", via="pronoun")
        return None

    def _resolve_tail(
        self,
        rel: RelationPrediction,
        entities: list[EntitySpan],
    ) -> EntitySpan | None:
        """Return the best GLiNER entity overlapping a relation tail span."""
        allowed = set(self.schema.allowed_tails(rel.relation))
        best: EntitySpan | None = None
        best_score = -1.0
        for entity in entities:
            if (
                entity.label in allowed
                and rel.tail_start < entity.end
                and entity.start < rel.tail_end
                and entity.score > best_score
            ):
                best = entity
                best_score = entity.score
        return best

    @staticmethod
    def _relation_for_entity(entity: EntitySpan) -> str | None:
        """Simple keyword-based relation fallback for stand-alone PII mentions."""
        label = entity.label.upper()
        if label == "EMAIL_ADDRESS":
            return "has_email_address"
        if label == "PHONE_NUMBER":
            return "has_phone_number"
        # HOME_ADDRESS is intentionally excluded from the fallback: GLiNER labels
        # office room numbers, city codes, and usernames as addresses on this data.
        if label == "AGE":
            return "has_age"
        # DATE is intentionally excluded: email headers are full of dates, so
        # falling back to born_on_date creates mostly noise.
        if label == "JOB_TITLE":
            return "has_job_title"
        if label == "ORGANIZATION":
            return "works_at_organization"
        if label in {"CITY", "COUNTRY", "NATIONALITY"}:
            return "lives_in_city" if label == "CITY" else "country_of_origin"
        return None

    @staticmethod
    def _nearest_user_mention_distance(
        entity: EntitySpan,
        entities: list[EntitySpan],
        entity_to_user: dict[tuple[int, int], UserLinkResult],
    ) -> int | None:
        """Return the smallest char distance from ``entity`` to a user-linked PERSON.

        Distance is measured from the nearest end of one span to the nearest
        end of the other. Returns ``None`` if no user-linked PERSON exists.
        """
        best: int | None = None
        for e in entities:
            if e.label != "PERSON":
                continue
            link = entity_to_user.get((e.start, e.end))
            if not link or not link.user_id:
                continue
            # distance between the two spans
            distance = max(0, max(entity.start, e.start) - min(entity.end, e.end))
            if best is None or distance < best:
                best = distance
        return best


# ---------------------------------------------------------------------------
# Graph -> attack ground truth
# ---------------------------------------------------------------------------

_GRAPH_RELATION_TO_PII_TYPE: dict[str, str] = {
    "has_email_address": "email",
    "has_phone_number": "phone",
    "has_home_address": "address",
    "has_job_title": "job_title",
    "works_at_organization": "organization",
    "lives_in_city": "city",
    "country_of_origin": "country",
    "has_age": "age",
    "has_name": "name",
}

GRAPH_PII_TYPES: set[str] = set(_GRAPH_RELATION_TO_PII_TYPE.values())


def build_ground_truth_from_graph(
    graph_path: str | os.PathLike[str],
    users: list[str],
    pii_types: list[str],
    user_records: dict[str, list[dict]],
    min_support: int = 2,
) -> dict[str, dict[str, dict[str, object]]]:
    """Build a per-user ground-truth dict from a PII graph.

    For each user and requested PII type, all graph values of the corresponding
    relation are collected, their support in the user's own records is counted,
    and the most-supported value is selected. This mirrors the signature-based
    extraction but uses graph candidates instead.
    """
    with Path(graph_path).open("r", encoding="utf-8") as f:
        graph = json.load(f)

    labels = {node["id"]: node["label"] for node in graph["nodes"]}

    # Group edges by user and PII type.
    candidates: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for edge in graph["edges"]:
        relation = edge.get("relation")
        pii_type = _GRAPH_RELATION_TO_PII_TYPE.get(relation)
        if pii_type is None or pii_type not in pii_types:
            continue
        source = edge.get("source", "")
        if not source.startswith("USER:"):
            continue
        user = source[5:]
        value = labels.get(edge.get("target"), "").strip()
        if not value:
            continue
        # Clean up whitespace in the raw label.
        value = re.sub(r"\s+", " ", value)
        candidates[user][pii_type].append((value, float(edge.get("score", 0.0))))

    all_ground_truth: dict[str, dict[str, dict[str, object]]] = {}
    for user in tqdm(users, desc="Building ground truth from graph", unit="user"):
        gt: dict[str, dict[str, object]] = {}
        records = user_records.get(user, [])
        # Precompute lower-cased and digit-only versions of each record text
        # once per user, so candidate support counting is O(1) per check.
        record_texts = [rec.get("text", "") for rec in records]
        record_lowers = [t.lower() for t in record_texts]
        record_digits = [re.sub(r"\D", "", t) for t in record_texts]

        for pii_type in pii_types:
            cand_list = candidates[user].get(pii_type, [])
            if not cand_list:
                continue

            # Count own-record support for each distinct value.
            support: dict[str, int] = defaultdict(int)
            if pii_type in ("phone", "fax"):
                for value, _ in cand_list:
                    variants = {v for v in _phone_match_variants(value) if len(v) >= 10}
                    if not variants:
                        continue
                    for rec_digits in record_digits:
                        if any(v in rec_digits for v in variants):
                            support[value] += 1
            elif pii_type == "email":
                for value, _ in cand_list:
                    target = normalize_email(value)
                    for rec_text in record_texts:
                        if target in rec_text.lower():
                            support[value] += 1
            else:
                # name, job_title, organization, city, country, age, address
                for value, _ in cand_list:
                    target = value.strip().lower()
                    if not target:
                        continue
                    for rec_lower in record_lowers:
                        if target in rec_lower:
                            support[value] += 1

            if not support:
                continue

            # Pick the value with the highest support; break ties by GLiNER score.
            value_scores: dict[str, float] = {}
            for value, score in cand_list:
                value_scores[value] = max(value_scores.get(value, 0.0), score)

            sorted_values = sorted(
                support.keys(),
                key=lambda v: (-support[v], -value_scores.get(v, 0.0)),
            )
            best_value = sorted_values[0]
            best_support = support[best_value]
            low_confidence = best_support < min_support

            gt[pii_type] = {
                "value": best_value,
                "support": best_support,
                "low_confidence": low_confidence,
            }
        all_ground_truth[user] = gt

    return all_ground_truth


# ---------------------------------------------------------------------------
# Text chunking helpers
# ---------------------------------------------------------------------------

_MAX_CHUNK_CHARS = 1500


def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Split ``text`` into sentence spans, returning ``(start, end, sentence)``."""
    pattern = re.compile(r"[^.!?]+[.!?]+[\s]*")
    matches = list(pattern.finditer(text))
    if not matches:
        return [(0, len(text), text)]

    spans: list[tuple[int, int, str]] = []
    prev_end = 0
    for match in matches:
        if match.start() > prev_end:
            spans.append((prev_end, match.start(), text[prev_end : match.start()]))
        spans.append((match.start(), match.end(), match.group()))
        prev_end = match.end()

    if prev_end < len(text):
        spans.append((prev_end, len(text), text[prev_end:]))

    # If no actual sentence punctuation was found, keep the whole text.
    if len(spans) == 1:
        spans = [(0, len(text), text)]
    return spans


def chunk_text(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> Iterable[tuple[str, int]]:
    # NOTE: this function is named ``chunk_text`` (used by callers).  The loop
    # variable below is intentionally named ``chunk`` to avoid shadowing.
    """Yield ``(chunk_text, char_offset)`` chunks that fit in model context.

    The chunker tries to keep sentence boundaries intact.  If a single sentence
    exceeds ``max_chars`` it is still emitted as its own chunk.
    """
    sentences = _split_sentences(text)
    current_chunks: list[str] = []
    current_length = 0
    current_offset: int = sentences[0][0] if sentences else 0

    for start, end, sentence in sentences:
        sentence_len = len(sentence)
        if sentence_len > max_chars:
            # Flush any buffered sentences first.
            if current_chunks:
                yield "".join(current_chunks), current_offset
                current_chunks = []
                current_length = 0
            yield sentence, start
            current_offset = end
            continue

        if current_length + sentence_len > max_chars and current_chunks:
            yield "".join(current_chunks), current_offset
            current_chunks = [sentence]
            current_length = sentence_len
            current_offset = start
        else:
            current_chunks.append(sentence)
            current_length += sentence_len

    if current_chunks:
        yield "".join(current_chunks), current_offset
