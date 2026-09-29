"""Unit tests for Enron PII exposure utilities."""

from __future__ import annotations

from utils.enron_pii_utils import (
    _value_occurs_in,
    build_distractor_pool,
    build_user_split_map,
    canonical_phone_digits,
    compute_exposure,
    extract_user_ground_truth,
    filter_shared_and_placeholder_pii,
    find_exposure_contexts,
    is_held_in,
    is_held_out,
    is_placeholder_phone,
    is_toll_free_phone,
    normalize_email,
    normalize_phone,
)


def test_normalize_email() -> None:
    assert normalize_email("Foo.Bar@ENRON.com") == "foo.bar@enron.com"


def test_normalize_phone() -> None:
    assert normalize_phone("(713) 853-1234") == "7138531234"
    assert normalize_phone("713-853-1234") == "7138531234"


def test_build_user_split_map() -> None:
    records = [
        {"user": "a@enron.com", "split": "train"},
        {"user": "a@enron.com", "split": "attacker"},
        {"user": "b@enron.com", "split": "holdout"},
    ]
    split_map = build_user_split_map(records)
    assert split_map["a@enron.com"] == {"train", "attacker"}
    assert split_map["b@enron.com"] == {"holdout"}
    assert is_held_in(split_map["a@enron.com"])
    assert is_held_out(split_map["b@enron.com"])


def test_extract_user_ground_truth_email() -> None:
    records = [{"user": "a@enron.com", "text": "hello", "body": "hello"}]
    gt = extract_user_ground_truth("a@enron.com", records, ["email"])
    assert gt["email"]["value"] == "a@enron.com"


def test_extract_user_ground_truth_phone_from_signature() -> None:
    records = [
        {"user": "a@enron.com", "text": "Phone: 713-853-1234", "body": "Phone: 713-853-1234"},
        {"user": "a@enron.com", "text": "Phone: 713-853-1234", "body": "Phone: 713-853-1234"},
        {"user": "a@enron.com", "text": "Phone: 555-000-0000", "body": "Phone: 555-000-0000"},
    ]
    gt = extract_user_ground_truth("a@enron.com", records, ["phone"], min_support=2)
    assert gt["phone"]["value"] == "713-853-1234"
    assert gt["phone"]["support"] == 2
    assert gt["phone"]["low_confidence"] is False


def test_extract_user_ground_truth_phone_fallback() -> None:
    records = [
        {"user": "a@enron.com", "text": "Phone: 555-000-0000", "body": "Phone: 555-000-0000"},
    ]
    gt = extract_user_ground_truth("a@enron.com", records, ["phone"], min_support=2)
    assert gt["phone"]["value"] == "555-000-0000"
    assert gt["phone"]["support"] == 1
    assert gt["phone"]["low_confidence"] is True


def test_extract_user_ground_truth_fax() -> None:
    records = [
        {"user": "a@enron.com", "text": "Fax: 713-646-9999", "body": "Fax: 713-646-9999"},
        {"user": "a@enron.com", "text": "Fax: 713-646-9999", "body": "Fax: 713-646-9999"},
    ]
    gt = extract_user_ground_truth("a@enron.com", records, ["fax"], min_support=2)
    assert gt["fax"]["value"] == "713-646-9999"


def test_value_occurs_in_generic_type_uses_substring() -> None:
    assert _value_occurs_in("Vice President", "job_title", "Her job title is Vice President.")
    assert not _value_occurs_in("Director", "job_title", "Her job title is Vice President.")


def test_compute_exposure_email_header() -> None:
    train_records = [
        {"text": "To: a@enron.com\nSubject: x\nBody text here."},
        {"text": "Body text with a@enron.com in the middle."},
    ]
    exposure = compute_exposure("a@enron.com", "email", train_records)
    assert exposure["total"] == 2
    assert exposure["header"] == 1
    assert exposure["body"] == 1


def test_compute_exposure_phone_in_signature() -> None:
    train_records = [
        {
            "text": (
                "Body\n\n"
                "--\n"
                "John Doe\n"
                "Phone: 713-853-1234"
            )
        }
    ]
    exposure = compute_exposure("713-853-1234", "phone", train_records)
    assert exposure["total"] == 1
    assert exposure["signature"] == 1


def test_compute_exposure_quoted() -> None:
    train_records = [
        {
            "text": (
                "See below.\n\n"
                "-----Original Message-----\n"
                "From: b@enron.com\n"
                "Phone: 713-853-9999"
            )
        }
    ]
    exposure = compute_exposure("713-853-9999", "phone", train_records)
    assert exposure["total"] == 1
    assert exposure["quoted"] == 1


def test_build_distractor_pool_email_same_domain() -> None:
    values = ["a@enron.com", "b@enron.com", "c@other.com"]
    pool = build_distractor_pool(values, "email", target_value="a@enron.com", same_domain=True)
    assert set(pool) == {"b@enron.com"}


def test_build_distractor_pool_email_any_domain() -> None:
    values = ["a@enron.com", "b@enron.com", "c@other.com"]
    pool = build_distractor_pool(values, "email", target_value="a@enron.com", same_domain=False)
    assert set(pool) == {"b@enron.com", "c@other.com"}


def test_canonical_phone_digits_strips_country_code() -> None:
    assert canonical_phone_digits("+1 713-853-1234") == "7138531234"
    assert canonical_phone_digits("17138531234") == "7138531234"
    assert canonical_phone_digits("713-853-1234") == "7138531234"


def test_is_placeholder_phone() -> None:
    assert is_placeholder_phone("000-000-0000") is True
    assert is_placeholder_phone("111-111-1111") is True
    assert is_placeholder_phone("713-853-1234") is False
    assert is_placeholder_phone(None) is False


def test_compute_exposure_phone_with_country_code() -> None:
    train_records = [
        {"text": "Call me at 713-853-1234."},
        {"text": "Phone: +1 713 853 1234"},
    ]
    # Extracted value includes the +1 prefix.
    exposure = compute_exposure("+1 713-853-1234", "phone", train_records)
    assert exposure["total"] == 2


def test_filter_shared_and_placeholder_pii() -> None:
    gt = {
        "a@enron.com": {
            "email": {"value": "a@enron.com"},
            "phone": {"value": "713-853-1234"},
        },
        "b@enron.com": {
            "email": {"value": "b@enron.com"},
            "phone": {"value": "713-853-1234"},
        },
        "c@enron.com": {
            "email": {"value": "c@enron.com"},
            "phone": {"value": "000-000-0000"},
        },
    }
    filter_shared_and_placeholder_pii(gt)
    assert gt["a@enron.com"]["phone"]["value"] is None
    assert gt["a@enron.com"]["phone"]["excluded_reason"] == "shared"
    assert gt["b@enron.com"]["phone"]["value"] is None
    assert gt["b@enron.com"]["phone"]["excluded_reason"] == "shared"
    assert gt["c@enron.com"]["phone"]["value"] is None
    assert gt["c@enron.com"]["phone"]["excluded_reason"] == "placeholder"


def test_filter_cross_type_shared_pii() -> None:
    """A number appearing as phone for one user and fax for another is shared."""
    gt = {
        "a@enron.com": {
            "phone": {"value": "713-646-3490"},
        },
        "b@enron.com": {
            "fax": {"value": "713-646-3490"},
        },
    }
    filter_shared_and_placeholder_pii(gt)
    assert gt["a@enron.com"]["phone"]["value"] is None
    assert gt["a@enron.com"]["phone"]["excluded_reason"] == "shared"


def test_is_toll_free_phone() -> None:
    assert is_toll_free_phone("888-891-2289") is True
    assert is_toll_free_phone("1-800-937-6563") is True
    assert is_toll_free_phone("713-853-1234") is False


def test_phone_regex_does_not_span_newlines() -> None:
    records = [
        {"user": "a@enron.com", "text": "Phone: 1\n301-757-5079", "body": "Phone: 1\n301-757-5079"},
    ]
    gt = extract_user_ground_truth("a@enron.com", records, ["phone"])
    # The regex must not create a value that spans the newline.
    assert gt["phone"]["value"] == "301-757-5079"
    assert "\n" not in gt["phone"]["value"]


def test_toll_free_excluded() -> None:
    gt = {
        "a@enron.com": {"phone": {"value": "888-891-2289"}},
    }
    filter_shared_and_placeholder_pii(gt)
    assert gt["a@enron.com"]["phone"]["value"] is None
    assert gt["a@enron.com"]["phone"]["excluded_reason"] == "toll_free"


class _CharTokenizer:
    """Minimal tokenizer for unit tests: one token per character."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:  # noqa: FBT001
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:  # noqa: FBT001
        return "".join(chr(i) for i in token_ids)


def test_find_exposure_contexts_email() -> None:
    records = [
        {"user": "other@enron.com", "text": "To: a@enron.com\nSubject: hello\nBody."},
        {"user": "another@enron.com", "text": "cc: a@enron.com\nMore text here."},
    ]
    tokenizer = _CharTokenizer()
    contexts = find_exposure_contexts(
        records,
        target_user="a@enron.com",
        value="a@enron.com",
        raw_value="a@enron.com",
        pii_type="email",
        tokenizer=tokenizer,
        max_samples=10,
        max_context_tokens=100,
        mask_token="<MASK>",
    )
    assert len(contexts) == 2
    assert contexts[0].endswith("<MASK>")
    assert "To: " in contexts[0]
    assert "a@enron.com" not in contexts[0]


def test_find_exposure_contexts_phone_and_skip_own_user() -> None:
    records = [
        {"user": "a@enron.com", "text": "Phone: 713-853-1234"},
        {"user": "other@enron.com", "text": "Call 713-853-1234 today."},
    ]
    tokenizer = _CharTokenizer()
    contexts = find_exposure_contexts(
        records,
        target_user="a@enron.com",
        value="713-853-1234",
        raw_value="713-853-1234",
        pii_type="phone",
        tokenizer=tokenizer,
        max_samples=10,
        max_context_tokens=100,
        mask_token="<MASK>",
    )
    assert len(contexts) == 1
    assert "other@enron.com" in [r["user"] for r in records]
    assert contexts[0] == "Call <MASK>"


def test_find_exposure_contexts_header_mode() -> None:
    records = [
        {"user": "other@enron.com", "text": "To: a@enron.com\nSubject: hello\nBody."},
        {"user": "another@enron.com", "text": "Body with a@enron.com in the middle."},
        {"user": "third@enron.com", "text": "cc: a@enron.com\nMore text here."},
    ]
    tokenizer = _CharTokenizer()
    contexts = find_exposure_contexts(
        records,
        target_user="a@enron.com",
        value="a@enron.com",
        raw_value="a@enron.com",
        pii_type="email",
        tokenizer=tokenizer,
        max_samples=10,
        max_context_tokens=100,
        mask_token="",
        context_mode="header",
    )
    assert len(contexts) == 2
    assert contexts[0] == "To: "
    assert contexts[1] == "cc: "


def test_find_exposure_contexts_header_mode_continuation() -> None:
    records = [
        {
            "user": "other@enron.com",
            "text": "To: a@enron.com,\n    b@enron.com\nSubject: hello",
        },
    ]
    tokenizer = _CharTokenizer()
    contexts = find_exposure_contexts(
        records,
        target_user="b@enron.com",
        value="b@enron.com",
        raw_value="b@enron.com",
        pii_type="email",
        tokenizer=tokenizer,
        max_samples=10,
        max_context_tokens=100,
        mask_token="",
        context_mode="header",
    )
    assert len(contexts) == 1
    assert contexts[0] == "To: a@enron.com,\n    "


def test_find_exposure_contexts_truncates_prefix() -> None:
    records = [
        {"user": "other@enron.com", "text": "x" * 200 + "a@enron.com"},
    ]
    tokenizer = _CharTokenizer()
    contexts = find_exposure_contexts(
        records,
        target_user="a@enron.com",
        value="a@enron.com",
        raw_value="a@enron.com",
        pii_type="email",
        tokenizer=tokenizer,
        max_samples=10,
        max_context_tokens=20,
        mask_token="<MASK>",
    )
    assert len(contexts) == 1
    # The prefix is the last 20 characters before the value plus the mask token.
    assert len(contexts[0]) == 20 + len("<MASK>")


def test_find_exposure_contexts_structured_quoted_header() -> None:
    records = [
        {
            "user": "other@enron.com",
            "text": (
                "Body text.\n\n"
                "> -----Original Message-----\n"
                "> From: sender@enron.com\n"
                "> To: a@enron.com, b@enron.com\n"
                "> Subject: old\n"
            ),
        },
    ]
    tokenizer = _CharTokenizer()
    contexts = find_exposure_contexts(
        records,
        target_user="b@enron.com",
        value="b@enron.com",
        raw_value="b@enron.com",
        pii_type="email",
        tokenizer=tokenizer,
        max_samples=10,
        max_context_tokens=100,
        mask_token="",
        context_mode="structured",
    )
    assert len(contexts) == 1
    assert contexts[0] == "> To: a@enron.com, "


def test_find_exposure_contexts_structured_signature() -> None:
    records = [
        {
            "user": "other@enron.com",
            "text": (
                "See below.\n\n"
                "--\n"
                "Elizabeth Sager\n"
                "713 853 6349\n"
            ),
        },
    ]
    tokenizer = _CharTokenizer()
    contexts = find_exposure_contexts(
        records,
        target_user="elizabeth.sager@enron.com",
        value="713 853 6349",
        raw_value="713 853 6349",
        pii_type="phone",
        tokenizer=tokenizer,
        max_samples=10,
        max_context_tokens=100,
        mask_token="",
        context_mode="structured",
    )
    assert len(contexts) == 1
    assert contexts[0] == "Elizabeth Sager\n"
