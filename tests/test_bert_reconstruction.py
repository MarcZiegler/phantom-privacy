"""Tests for utils/bert_reconstruction.py."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from utils import bert_reconstruction as bert


class _FakeTokenizer:
    """Tiny tokenizer that maps single characters to integer ids."""

    def __init__(self) -> None:
        self.mask_token_id = 100
        self.pad_token_id = 0
        self.sep_token_id = 102
        self.cls_token_id = 101
        self.model_max_length = 512
        self.vocab = {c: i + 1 for i, c in enumerate("abcd")}
        self.vocab["[MASK]"] = self.mask_token_id
        self.vocab["[PAD]"] = self.pad_token_id
        self.vocab["[SEP]"] = self.sep_token_id
        self.vocab["[CLS]"] = self.cls_token_id

    def _encode(self, text: str) -> list[int]:
        return [self.vocab.get(c, 99) for c in text]

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs) -> list[int]:  # noqa: ARG002
        ids = []
        if add_special_tokens:
            ids.append(self.cls_token_id)
        ids.extend(self._encode(text))
        if add_special_tokens:
            ids.append(self.sep_token_id)
        return ids

    def __call__(self, texts, add_special_tokens: bool = True, return_tensors: str = "pt", padding: bool = False, max_length: int | None = None, truncation: bool = False):  # noqa: ARG002
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
        max_len = max(len(ids) for ids in encoded)
        input_ids = []
        attention_mask = []
        for ids in encoded:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class _FakeMaskedLM:
    """Predicts token_id = previous_token_id + 1 at each masked position."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.device = "cpu"

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size)
        # For every position, predict previous token + 1.
        prev = torch.zeros(batch_size, seq_len, dtype=torch.long)
        prev[:, 1:] = input_ids[:, :-1]
        predicted = (prev + 1) % self.vocab_size
        # Set a high logit for the predicted token.
        for b in range(batch_size):
            for pos in range(seq_len):
                logits[b, pos, predicted[b, pos].item()] = 10.0
        return SimpleNamespace(logits=logits)

    def to(self, device):
        return self

    def eval(self):
        pass


def test_score_candidates_prefers_context_consistent_candidate() -> None:
    tokenizer = _FakeTokenizer()
    model = _FakeMaskedLM(vocab_size=200)

    # Under the fake model, "abc" is self-consistent (each token predicted from
    # the previous one), while "abd" ends in an unexpected token.
    candidates = ["abc", "abd"]
    results = bert.score_candidates_with_mlm(
        prefix="",
        candidates=candidates,
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )

    assert len(results) == 2
    pll_abc, toks_abc = results[0]
    pll_abd, toks_abd = results[1]
    assert math.isfinite(pll_abc)
    assert math.isfinite(pll_abd)
    assert pll_abc < pll_abd
    assert toks_abc == 3
    assert toks_abd == 3


def test_score_candidates_returns_infinity_for_empty_candidate() -> None:
    tokenizer = _FakeTokenizer()
    model = _FakeMaskedLM(vocab_size=200)

    results = bert.score_candidates_with_mlm(
        prefix="",
        candidates=[""],
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )
    assert len(results) == 1
    pll, toks = results[0]
    assert not math.isfinite(pll) or toks == 0


def test_load_bert_model_with_fake_pretrained(monkeypatch) -> None:
    calls = {}

    def fake_from_pretrained(path):
        calls["path"] = path
        return SimpleNamespace(
            to=lambda device: SimpleNamespace(eval=lambda: None),
        )

    class FakeTokenizer:
        mask_token_id = 100
        pad_token_id = 0
        pad_token = "[PAD]"
        eos_token = "[PAD]"

        @classmethod
        def from_pretrained(cls, path):
            calls["tokenizer_path"] = path
            return cls()

    monkeypatch.setattr(
        bert.AutoModelForMaskedLM, "from_pretrained", staticmethod(fake_from_pretrained)
    )
    monkeypatch.setattr(bert.AutoTokenizer, "from_pretrained", FakeTokenizer.from_pretrained)

    model, tokenizer = bert.load_bert_model("models/base/roberta-base", torch.device("cpu"))
    assert calls["path"] == "models/base/roberta-base"
    assert calls["tokenizer_path"] == "models/base/roberta-base"
    assert tokenizer.pad_token == "[PAD]"
