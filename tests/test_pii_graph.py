"""Tests for the user-level PII graph pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
import torch

from utils import pii_graph_utils
from utils.pii_graph_utils import (
    CorefCluster,
    CorefMention,
    EntitySpan,
    GLiNERNER,
    GLiRELExtractor,
    NullRelationExtractor,
    PIIGraph,
    PIIGraphBuilder,
    PIIGraphSchema,
    RelationPrediction,
    RuleCorefResolver,
    StanzaCorefResolver,
    UserLinker,
    build_ground_truth_from_graph,
    build_user_aliases,
    canonical_value,
    chunk_text,
    is_valid_pii_value,
    load_word_set,
    node_id_for_entity,
    resolve_device,
    user_node_id,
)


class _MockNER:
    """Stub NER extractor that returns configured entities."""

    def __init__(self, entities: list[EntitySpan]) -> None:
        self.entities = entities

    def extract(
        self,
        text: str,
        schema: PIIGraphSchema,
        threshold: float | None = None,
    ) -> list[EntitySpan]:
        return list(self.entities)


class _MockCoref:
    """Stub coref resolver that returns configured clusters."""

    def __init__(self, clusters: list[CorefCluster]) -> None:
        self.clusters = clusters

    def resolve(self, text: str) -> list[CorefCluster]:
        return list(self.clusters)


class _MockRE:
    """Stub relation extractor that returns configured predictions."""

    def __init__(self, relations: list[RelationPrediction]) -> None:
        self.relations = relations

    def extract(
        self,
        text: str,
        schema: PIIGraphSchema,
        threshold: float | None = None,
        entities: list[EntitySpan] | None = None,
    ) -> list[RelationPrediction]:
        return list(self.relations)


def test_schema_loads_default() -> None:
    schema = PIIGraphSchema()
    assert schema.name == "default-email-pii"
    assert "PERSON" in schema.entity_labels
    assert "has_phone_number" in schema.relation_labels
    assert schema.allowed_tails("has_phone_number") == ["PHONE_NUMBER"]


def test_schema_loads_custom(tmp_path: Path) -> None:
    path = tmp_path / "schema.json"
    path.write_text(
        json.dumps(
            {
                "schema_name": "custom",
                "entity_labels": {"VEHICLE": "A car"},
                "relation_labels": {
                    "has_car": {
                        "description": "Owns a car",
                        "allowed_head": ["PERSON"],
                        "allowed_tail": ["VEHICLE"],
                    }
                },
                "default_thresholds": {},
            }
        )
    )
    schema = PIIGraphSchema(path)
    assert schema.entity_labels == ["VEHICLE"]
    assert schema.relation_query("has_car") == "has car"


def test_chunk_text_respects_offsets() -> None:
    text = "Hello. World. Foo bar."
    chunks = list(chunk_text(text, max_chars=12))
    # Sentence "Hello. " (7 chars) + "World. " (7 chars) = 14 > 12, so first
    # chunk is just "Hello. " and the next chunk starts at offset 7.
    assert chunks[0] == ("Hello. ", 0)
    assert chunks[1][1] == 7
    # Check that offsets are consistent with slicing the original text.
    for piece, offset in chunks:
        assert text[offset : offset + len(piece)] == piece


def test_user_linker_pronouns() -> None:
    linker = UserLinker()
    rec = {"user": "john.smith"}
    entity = EntitySpan(label="PERSON", text="My", start=0, end=2, score=1.0)
    link = linker.link(entity, "john.smith", [], rec)
    assert link.user_id == "john.smith"
    assert link.via == "pronoun"


def test_user_linker_name_match() -> None:
    linker = UserLinker()
    rec = {"user": "john.smith"}
    entity = EntitySpan(label="PERSON", text="John Smith", start=0, end=10, score=1.0)
    link = linker.link(entity, "john.smith", [], rec)
    assert link.user_id == "john.smith"
    assert link.via == "name_match"


def test_user_linker_coref_match() -> None:
    linker = UserLinker()
    rec = {"user": "john.smith"}
    entity = EntitySpan(label="PERSON", text="He", start=15, end=17, score=1.0)
    cluster = CorefCluster(
        cluster_id=0,
        mentions=(
            CorefMention(start=0, end=10, text="John Smith"),
            CorefMention(start=15, end=17, text="He"),
        ),
        representative=CorefMention(start=0, end=10, text="John Smith"),
    )
    link = linker.link(entity, "john.smith", [cluster], rec)
    assert link.user_id == "john.smith"
    assert link.via == "coref"


def test_graph_builder_adds_user_pii_edges() -> None:
    schema = PIIGraphSchema()
    entities = [
        EntitySpan(label="PERSON", text="John Smith", start=0, end=10, score=1.0),
        EntitySpan(label="PHONE_NUMBER", text="555-123-4567", start=20, end=32, score=1.0),
        EntitySpan(label="EMAIL_ADDRESS", text="js@enron.com", start=40, end=52, score=1.0),
    ]
    relations = [
        RelationPrediction(
            head_start=0,
            head_end=10,
            head_text="John Smith",
            tail_start=20,
            tail_end=32,
            tail_text="555-123-4567",
            relation="has_phone_number",
            score=0.9,
        ),
    ]
    builder = PIIGraphBuilder(
        ner=_MockNER(entities),
        coref=_MockCoref([]),
        relation_extractor=_MockRE(relations),
        schema=schema,
    )
    rec = {"user": "john.smith", "text": "John Smith ...", "record_id": "r1"}
    graph = builder.process_record(rec)

    assert user_node_id("john.smith") in graph.nodes
    assert node_id_for_entity(entities[1]) in graph.nodes
    assert any(e["relation"] == "has_phone_number" for e in graph.edges)


def test_graph_builder_fallback_edges_when_no_relations() -> None:
    schema = PIIGraphSchema()
    entities = [
        EntitySpan(label="PERSON", text="John Smith", start=0, end=10, score=1.0),
        EntitySpan(label="AGE", text="34", start=20, end=22, score=1.0),
    ]
    builder = PIIGraphBuilder(
        ner=_MockNER(entities),
        coref=_MockCoref([]),
        relation_extractor=_MockRE([]),
        schema=schema,
    )
    rec = {"user": "john.smith", "text": "John Smith is aged 34.", "record_id": "r1"}
    graph = builder.process_record(rec)

    assert any(e["relation"] == "has_age" for e in graph.edges)


def test_graph_builder_skips_non_user_person() -> None:
    schema = PIIGraphSchema()
    entities = [
        EntitySpan(label="PERSON", text="Mary", start=0, end=4, score=1.0),
        EntitySpan(label="PHONE_NUMBER", text="555-000-0000", start=10, end=22, score=1.0),
    ]
    relations = [
        RelationPrediction(
            head_start=0,
            head_end=4,
            head_text="Mary",
            tail_start=10,
            tail_end=22,
            tail_text="555-000-0000",
            relation="has_phone_number",
            score=0.9,
        ),
    ]
    builder = PIIGraphBuilder(
        ner=_MockNER(entities),
        coref=_MockCoref([]),
        relation_extractor=_MockRE(relations),
        schema=schema,
    )
    rec = {"user": "john.smith", "text": "Mary called 555-000-0000.", "record_id": "r1"}
    graph = builder.process_record(rec)

    # Mary does not match john.smith and is not a first-person pronoun, so the
    # phone number should not be attributed to the user.
    assert not any(e["relation"] == "has_phone_number" for e in graph.edges)


def test_rule_coref_resolves_pronouns() -> None:
    resolver = RuleCorefResolver()
    clusters = resolver.resolve("I called. He answered.")
    assert len(clusters) == 1
    assert clusters[0].mentions[0].text == "I"


def test_graph_deduplicates_edges_by_score() -> None:
    schema = PIIGraphSchema()
    graph = PIIGraph(schema=schema)
    graph.add_user("u1")
    graph.add_user("u2")
    graph.add_edge(
        "USER:u1", "USER:u2", "knows", 0.5, {"record_id": "first"}
    )
    graph.add_edge(
        "USER:u1", "USER:u2", "knows", 0.3, {"record_id": "second"}
    )
    graph.add_edge(
        "USER:u1", "USER:u2", "knows", 0.7, {"record_id": "third"}
    )
    assert len(graph.edges) == 1
    assert graph.edges[0]["score"] == 0.7
    assert graph.edges[0]["provenance"]["record_id"] == "third"


def test_null_relation_extractor_returns_empty() -> None:
    extractor = NullRelationExtractor()
    assert extractor.extract("any text", PIIGraphSchema()) == []


def test_phone_timestamp_is_filtered_out() -> None:
    timestamp = EntitySpan(
        label="PHONE_NUMBER", text="09:41:00 -0700", start=0, end=15, score=0.9
    )
    real_phone = EntitySpan(
        label="PHONE_NUMBER", text="713-853-1803", start=0, end=12, score=0.9
    )
    assert is_valid_pii_value(timestamp) is False
    assert is_valid_pii_value(real_phone) is True


def test_organization_canonical_value_collapse_whitespace() -> None:
    e1 = EntitySpan(label="ORGANIZATION", text="DJ\n Calif ISO", start=0, end=13, score=0.9)
    e2 = EntitySpan(label="ORGANIZATION", text="DJ\nCalif ISO", start=0, end=12, score=0.9)
    assert canonical_value(e1) == canonical_value(e2)


def test_home_address_fallback_is_not_used() -> None:
    builder = PIIGraphBuilder(
        ner=_MockNER([]),
        coref=_MockCoref([]),
        relation_extractor=_MockRE([]),
        schema=PIIGraphSchema(),
    )
    assert builder._relation_for_entity(
        EntitySpan(label="HOME_ADDRESS", text="123 Main St", start=0, end=11, score=0.9)
    ) is None


def test_node_ids_are_stable() -> None:
    entity = EntitySpan(label="EMAIL_ADDRESS", text="John@Enron.COM", start=0, end=14, score=1.0)
    assert node_id_for_entity(entity) == "EMAIL_ADDRESS:john@enron.com"
    assert user_node_id("user A") == "USER:user A"


@pytest.mark.slow
@mock.patch.object(pii_graph_utils, "GLiNER")
def test_gliner_ner_wraps_model(mock_gliner_class: mock.MagicMock) -> None:
    """Smoke test that the GLiNER wrapper builds the expected entity spans."""
    mock_model = mock.MagicMock()
    mock_model.predict_entities.return_value = [
        {"text": "Alice", "label": "PERSON", "start": 0, "end": 5, "score": 0.99}
    ]
    mock_gliner_class.from_pretrained.return_value = mock_model

    ner = GLiNERNER("dummy_path")
    schema = PIIGraphSchema()
    spans = ner.extract("Alice is here.", schema)
    assert len(spans) == 1
    assert spans[0].text == "Alice"
    assert spans[0].label == "PERSON"


@pytest.mark.slow
@mock.patch.object(pii_graph_utils, "GLiREL")
def test_glirel_extractor_wraps_model(mock_glirel_class: mock.MagicMock) -> None:
    """Smoke test that the GLiREL wrapper builds the expected relation predictions."""
    mock_model = mock.MagicMock()
    mock_model._from_pretrained.return_value = mock_model
    mock_model.predict_relations.return_value = [
        {
            "head_pos": [0, 5],
            "head_text": "Alice",
            "tail_pos": [10, 20],
            "tail_text": "555-0000",
            "label": "phone number",
            "score": 0.85,
        }
    ]
    mock_glirel_class._from_pretrained = mock_model._from_pretrained
    extractor = GLiRELExtractor("dummy_path")
    schema = PIIGraphSchema()
    entities = [
        EntitySpan(label="PERSON", text="Alice", start=0, end=5, score=0.99),
        EntitySpan(label="PHONE_NUMBER", text="555-0000", start=10, end=20, score=0.99),
    ]
    preds = extractor.extract("Alice called 555-0000.", schema, entities=entities)
    assert len(preds) == 1
    assert preds[0].relation == "has_phone_number"


def test_stanza_coref_skips_when_no_coref_attribute() -> None:
    """Ensure the resolver is resilient to Stanza returning empty coref output."""
    mock_pipeline = mock.MagicMock()
    with mock.patch.object(pii_graph_utils.stanza, "Pipeline", return_value=mock_pipeline):
        resolver = StanzaCorefResolver("dummy_path")
    mock_doc = mock.MagicMock()
    del mock_doc.coref
    mock_pipeline.return_value = mock_doc
    assert resolver.resolve("test") == []


def test_extract_pii_graph_cli_dry_run(tmp_path: Path) -> None:
    """Run the CLI in dry-run mode with mocked heavy components."""
    import extract_pii_graph

    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.json"
    input_path.write_text(
        json.dumps(
            {
                "user": "u1",
                "text": "Alice is aged 30.",
                "record_id": "r1",
            }
        )
        + "\n"
    )

    with mock.patch.object(
        extract_pii_graph, "GLiNERNER", return_value=_MockNER([])
    ), mock.patch.object(
        extract_pii_graph, "GLiRELExtractor", return_value=_MockRE([])
    ), mock.patch.object(
        extract_pii_graph, "StanzaCorefResolver", return_value=_MockCoref([])
    ), mock.patch.object(
        extract_pii_graph, "RuleCorefResolver", return_value=_MockCoref([])
    ), mock.patch.object(
        extract_pii_graph, "NullCorefResolver", return_value=_MockCoref([])
    ):
        args = [
            "extract_pii_graph.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--dry-run",
        ]
        with mock.patch("sys.argv", args):
            extract_pii_graph.main()
    assert not output_path.exists()


def test_extract_pii_graph_cli_caps_records_per_user(tmp_path: Path) -> None:
    """Ensure --max-records-per-user limits records processed per user."""
    import extract_pii_graph

    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.json"
    with input_path.open("w") as fh:
        for i in range(5):
            fh.write(
                json.dumps(
                    {
                        "user": "u1",
                        "text": f"Alice is aged {30 + i}.",
                        "record_id": f"r{i}",
                    }
                )
                + "\n"
            )

    processed: list[dict] = []

    class _RecordingBuilder:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def process_record(self, rec: dict, graph: object = None) -> object:
            processed.append(rec)
            return graph

    with mock.patch.object(
        extract_pii_graph, "GLiNERNER", return_value=_MockNER([])
    ), mock.patch.object(
        extract_pii_graph, "GLiRELExtractor", return_value=_MockRE([])
    ), mock.patch.object(
        extract_pii_graph, "StanzaCorefResolver", return_value=_MockCoref([])
    ), mock.patch.object(
        extract_pii_graph, "RuleCorefResolver", return_value=_MockCoref([])
    ), mock.patch.object(
        extract_pii_graph, "NullCorefResolver", return_value=_MockCoref([])
    ), mock.patch.object(
        extract_pii_graph, "PIIGraphBuilder", _RecordingBuilder
    ):
        args = [
            "extract_pii_graph.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--max-users",
            "1",
            "--max-records-per-user",
            "3",
        ]
        with mock.patch("sys.argv", args):
            extract_pii_graph.main()

    assert len(processed) == 3
    assert all(rec["user"] == "u1" for rec in processed)
    assert output_path.exists()


def test_pii_graph_test_preset_exists() -> None:
    """The PII-graph test preset has the expected limited configuration."""
    from jobs.presets import FINETUNE_PRESETS

    preset = FINETUNE_PRESETS["neo125-enron-strict-pii-graph-test"]
    assert preset.dataset_name == "enron"
    assert preset.pii_graph.max_users == 50
    assert preset.pii_graph.max_records_per_user == 50
    assert preset.pii_graph.coref == "stanza"
    assert preset.pii_graph.relation_extractor == "none"
    assert preset.pii_graph.max_chars == 2000
    assert preset.pii_graph.slurm is not None
    assert preset.pii_graph.slurm.partition == "gpu-1xA100"
    assert preset.pii_graph.slurm.mem_per_cpu == 3900
    assert preset.pii_graph.slurm.gpus_per_node is None


def test_pii_graph_full_preset_exists() -> None:
    """The full PII-graph preset uses no caps and a 12-hour time limit."""
    from jobs.presets import FINETUNE_PRESETS

    preset = FINETUNE_PRESETS["neo125-enron-strict-pii-graph"]
    assert preset.dataset_name == "enron"
    assert preset.pii_graph.max_users is None
    assert preset.pii_graph.max_records_per_user is None
    assert preset.pii_graph.coref == "stanza"
    assert preset.pii_graph.relation_extractor == "none"
    assert preset.pii_graph.max_chars == 1350
    assert preset.pii_graph.slurm is not None
    assert preset.pii_graph.slurm.partition == "gpu-1xA100"
    assert preset.pii_graph.slurm.time == "0-12:00:00"
    assert preset.pii_graph.slurm.mem_per_cpu == 3900


def test_resolve_device_auto_uses_cuda_when_available() -> None:
    """auto resolves to cuda iff torch.cuda.is_available() is True."""
    with mock.patch.object(torch.cuda, "is_available", return_value=True):
        assert resolve_device("auto") == "cuda"
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        assert resolve_device("auto") == "cpu"


def test_resolve_device_passes_through_cpu_and_cuda() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_passes_through_unknown_strings() -> None:
    """Unknown device strings are passed through for downstream errors."""
    assert resolve_device("mps") == "mps"


@pytest.mark.slow
@mock.patch.object(pii_graph_utils, "GLiNER")
def test_gliner_ner_resolves_auto_device(mock_gliner_class: mock.MagicMock) -> None:
    """GLiNERNER resolves 'auto' before calling model.to(...)."""
    mock_model = mock.MagicMock()
    mock_gliner_class.from_pretrained.return_value = mock_model

    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        ner = GLiNERNER("dummy_path", device="auto")

    assert ner.device == "cpu"
    mock_model.to.assert_not_called()


@pytest.mark.slow
@mock.patch.object(pii_graph_utils, "GLiNER")
def test_gliner_ner_moves_model_to_cuda_when_available(
    mock_gliner_class: mock.MagicMock,
) -> None:
    mock_model = mock.MagicMock()
    mock_gliner_class.from_pretrained.return_value = mock_model

    with mock.patch.object(torch.cuda, "is_available", return_value=True):
        ner = GLiNERNER("dummy_path", device="auto")

    assert ner.device == "cuda"
    mock_model.to.assert_called_once_with("cuda")


def test_is_valid_pii_value_accepts_real_pii() -> None:
    assert is_valid_pii_value(
        EntitySpan(label="EMAIL_ADDRESS", text="john.smith@enron.com", start=0, end=22, score=1.0),
        current_user="john.smith@enron.com",
    )
    assert is_valid_pii_value(
        EntitySpan(label="PHONE_NUMBER", text="713-555-1234", start=0, end=12, score=1.0)
    )
    assert is_valid_pii_value(
        EntitySpan(label="AGE", text="34", start=0, end=2, score=1.0)
    )


def test_is_valid_pii_value_rejects_garbage() -> None:
    assert not is_valid_pii_value(
        EntitySpan(label="EMAIL_ADDRESS", text="enron.com", start=0, end=9, score=1.0),
        current_user="john.smith@enron.com",
    )
    assert not is_valid_pii_value(
        EntitySpan(label="PHONE_NUMBER", text="0800", start=0, end=4, score=1.0)
    )
    assert not is_valid_pii_value(
        EntitySpan(label="PHONE_NUMBER", text="PST", start=0, end=3, score=1.0)
    )
    assert not is_valid_pii_value(
        EntitySpan(label="DATE", text="Mon", start=0, end=3, score=1.0)
    )
    assert not is_valid_pii_value(
        EntitySpan(label="DATE", text="26 Mar 2001", start=0, end=11, score=1.0)
    )
    assert not is_valid_pii_value(
        EntitySpan(label="AGE", text="abc", start=0, end=3, score=1.0)
    )


def test_is_valid_pii_value_keeps_only_own_email() -> None:
    current_user = "john.smith@enron.com"
    assert is_valid_pii_value(
        EntitySpan(label="EMAIL_ADDRESS", text="john.smith@enron.com", start=0, end=22, score=1.0),
        current_user=current_user,
    )
    # Another user's email is not attributed, even if the user is not in the dataset.
    assert not is_valid_pii_value(
        EntitySpan(label="EMAIL_ADDRESS", text="mary.jones@enron.com", start=0, end=22, score=1.0),
        current_user=current_user,
    )


def test_build_user_aliases_uses_metadata_fields() -> None:
    records = [
        {
            "user": "john.smith",
            "name": "John Smith",
            "display_name": "Johnny Smith",
            "text": "...",
        },
        {
            "user": "john.smith",
            "sender": "J. Smith",
            "text": "...",
        },
    ]
    aliases = build_user_aliases(records)
    assert "john.smith" in aliases["john.smith"]
    assert "john" in aliases["john.smith"]
    assert "smith" in aliases["john.smith"]
    assert "john smith" in aliases["john.smith"]
    assert "johnny smith" in aliases["john.smith"]
    assert "j. smith" in aliases["john.smith"]


def test_builder_fallback_respects_proximity_threshold() -> None:
    schema = PIIGraphSchema()
    entities = [
        # User mention at the very start.
        EntitySpan(label="PERSON", text="John Smith", start=0, end=10, score=1.0),
        # Phone far away (1000 chars after the user mention).
        EntitySpan(label="PHONE_NUMBER", text="713-555-1234", start=1010, end=1022, score=1.0),
        # Phone close to the user mention.
        EntitySpan(label="PHONE_NUMBER", text="713-555-9999", start=20, end=32, score=1.0),
    ]
    builder = PIIGraphBuilder(
        ner=_MockNER(entities),
        coref=_MockCoref([]),
        relation_extractor=_MockRE([]),
        schema=schema,
        fallback_proximity_chars=50,
    )
    rec = {"user": "john.smith", "text": "John Smith ... 713-555-9999 ... 713-555-1234", "record_id": "r1"}
    graph = builder.process_record(rec)

    relations = [e["relation"] for e in graph.edges]
    assert relations.count("has_phone_number") == 1
    target_ids = {e["target"] for e in graph.edges if e["relation"] == "has_phone_number"}
    assert "PHONE_NUMBER:7135559999" in target_ids
    assert "PHONE_NUMBER:7135551234" not in target_ids


def test_builder_fallback_filters_invalid_values() -> None:
    schema = PIIGraphSchema()
    entities = [
        EntitySpan(label="PERSON", text="John Smith", start=0, end=10, score=1.0),
        EntitySpan(label="EMAIL_ADDRESS", text="enron.com", start=20, end=29, score=1.0),
        EntitySpan(label="EMAIL_ADDRESS", text="john.smith@enron.com", start=31, end=52, score=1.0),
    ]
    builder = PIIGraphBuilder(
        ner=_MockNER(entities),
        coref=_MockCoref([]),
        relation_extractor=_MockRE([]),
        schema=schema,
        fallback_proximity_chars=300,
        known_users={"john.smith@enron.com"},
    )
    rec = {"user": "john.smith@enron.com", "text": "John Smith ...", "record_id": "r1"}
    graph = builder.process_record(rec)

    target_ids = {e["target"] for e in graph.edges if e["relation"] == "has_email_address"}
    assert "EMAIL_ADDRESS:john.smith@enron.com" in target_ids
    assert "EMAIL_ADDRESS:enron.com" not in target_ids


def test_is_valid_pii_value_filters_generic_organizations() -> None:
    common_words = {"trading", "marketing", "desk", "gas", "power", "energy", "east"}
    assert is_valid_pii_value(
        EntitySpan(label="ORGANIZATION", text="Enron", start=0, end=5, score=1.0),
        common_words=common_words,
    )
    assert not is_valid_pii_value(
        EntitySpan(label="ORGANIZATION", text="trading", start=0, end=7, score=1.0),
        common_words=common_words,
    )
    assert not is_valid_pii_value(
        EntitySpan(label="ORGANIZATION", text="east desk", start=0, end=9, score=1.0),
        common_words=common_words,
    )


def test_is_valid_pii_value_filters_unknown_job_titles() -> None:
    job_titles = {"director", "manager", "vice president", "options manager"}
    assert is_valid_pii_value(
        EntitySpan(label="JOB_TITLE", text="Director", start=0, end=8, score=1.0),
        job_titles=job_titles,
    )
    assert is_valid_pii_value(
        EntitySpan(label="JOB_TITLE", text="Options Manager", start=0, end=15, score=1.0),
        job_titles=job_titles,
    )
    assert not is_valid_pii_value(
        EntitySpan(label="JOB_TITLE", text="current gas desk list", start=0, end=19, score=1.0),
        job_titles=job_titles,
    )
    assert not is_valid_pii_value(
        EntitySpan(label="JOB_TITLE", text="FYI", start=0, end=3, score=1.0),
        job_titles=job_titles,
    )


def test_build_ground_truth_from_graph_selects_most_supported_value(tmp_path: Path) -> None:
    graph = {
        "nodes": [
            {"id": "USER:alice", "type": "USER", "label": "alice"},
            {"id": "JOB_TITLE:Director", "type": "JOB_TITLE", "label": "Director"},
            {"id": "JOB_TITLE:Manager", "type": "JOB_TITLE", "label": "Manager"},
        ],
        "edges": [
            {
                "source": "USER:alice",
                "target": "JOB_TITLE:Director",
                "relation": "has_job_title",
                "score": 0.9,
            },
            {
                "source": "USER:alice",
                "target": "JOB_TITLE:Manager",
                "relation": "has_job_title",
                "score": 0.95,
            },
        ],
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph))

    user_records = {
        "alice": [
            {"text": "Alice is a Manager here."},
            {"text": "Manager of the team."},
            {"text": "Director? No, Manager."},
        ]
    }
    gt = build_ground_truth_from_graph(
        graph_path,
        ["alice"],
        ["job_title"],
        user_records,
        min_support=2,
    )
    assert gt["alice"]["job_title"]["value"] == "Manager"
    assert gt["alice"]["job_title"]["support"] == 3
    assert gt["alice"]["job_title"]["low_confidence"] is False


def test_load_word_set_skips_comments_and_blanks(tmp_path: Path) -> None:
    path = tmp_path / "words.txt"
    path.write_text("# header\nApple\n\nbanana\n")
    words = load_word_set(path)
    assert words == {"apple", "banana"}
