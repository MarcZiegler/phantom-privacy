"""BERT masked-language-model scoring for PII reconstruction.

This module scores a set of candidate values for a masked prompt using the
pseudo-log-likelihood (PLL) of a masked LM.  It is designed to plug into the
existing PII extraction attack: it uses the same prefix and candidate list and
produces a score that can be compared directly with the likelihood-ratio score.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from transformers import AutoModelForMaskedLM, AutoTokenizer


def load_bert_model(
    model_name_or_path: str,
    device: torch.device,
) -> tuple[AutoModelForMaskedLM, AutoTokenizer]:
    """Load a masked LM and its tokenizer.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        device: Torch device to place the model on.

    Returns:
        ``(model, tokenizer)`` tuple.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _candidate_spans(
    prefix_ids: list[int],
    full_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    sep_token_id: int | None,
) -> tuple[list[int], list[int]]:
    """Return (start, end) token spans of the candidate suffix for each example.

    The span is computed as the longest common prefix between the tokenized
    prefix and the full ``prefix + candidate`` sequence.  The trailing special
    token (usually ``[SEP]``) is excluded from the span.
    """
    batch_size, seq_len = full_ids.shape
    prefix_len = len(prefix_ids)
    starts: list[int] = []
    ends: list[int] = []
    for i in range(batch_size):
        full_i = full_ids[i].tolist()
        # Longest common prefix with the prefix-only tokenization.
        k = 0
        limit = min(prefix_len, len(full_i))
        while k < limit and full_i[k] == prefix_ids[k]:
            k += 1
        # Exclude trailing separator token.
        end = seq_len
        if sep_token_id is not None:
            last_idx = int(attention_mask[i].nonzero(as_tuple=True)[0][-1].item())
            if full_i[last_idx] == sep_token_id:
                end = last_idx
        starts.append(k)
        ends.append(max(k, end))
    return starts, ends


def score_candidates_with_mlm(
    prefix: str,
    candidates: list[str],
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_seq_length: int | None = None,
) -> list[tuple[float, int]]:
    """Return average pseudo-log-likelihood and token count per candidate.

    For each candidate ``c`` the score is the average per-token negative
    log-likelihood of the candidate tokens in the sequence ``prefix + c`` under
    the masked LM, where each candidate token is masked one at a time.  Lower
    PLL means the model prefers the candidate.

    Args:
        prefix: Prompt prefix ending before the candidate value.
        candidates: Candidate strings to score.
        model: Masked language model.
        tokenizer: Tokenizer compatible with ``model``.
        device: Torch device.
        max_seq_length: Maximum sequence length; defaults to the model's max.

    Returns:
        A list of ``(avg_pll, num_candidate_tokens)`` tuples.  ``avg_pll`` is
        infinity if the candidate could not be scored.
    """
    if not candidates:
        return []

    if max_seq_length is None:
        max_seq_length = int(tokenizer.model_max_length)
    else:
        max_seq_length = min(int(max_seq_length), int(tokenizer.model_max_length))

    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    sep_token_id = getattr(tokenizer, "sep_token_id", None)
    if mask_token_id is None or pad_token_id is None:
        raise ValueError("Tokenizer must have mask_token_id and pad_token_id")

    model.eval()

    # Tokenize the shared prefix once.
    prefix_enc = tokenizer(
        prefix,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    )
    prefix_ids = prefix_enc["input_ids"][0].tolist()

    # Tokenize each prefix + candidate.
    full_texts = [prefix + c for c in candidates]
    full_enc = tokenizer(
        full_texts,
        add_special_tokens=True,
        return_tensors="pt",
        padding=True,
        max_length=max_seq_length,
        truncation=True,
    )
    full_ids = full_enc["input_ids"].to(device)
    attention_mask = full_enc["attention_mask"].to(device)
    batch_size, seq_len = full_ids.shape

    starts, ends = _candidate_spans(prefix_ids, full_ids, attention_mask, sep_token_id)
    span_lengths = [max(0, ends[i] - starts[i]) for i in range(batch_size)]
    max_span_length = max(span_lengths) if span_lengths else 0

    total_nll = torch.zeros(batch_size, device=device)
    counts = torch.zeros(batch_size, device=device)

    for offset in range(max_span_length):
        input_ids = full_ids.clone()
        labels = torch.full_like(full_ids, -100)
        valid = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i in range(batch_size):
            if offset < span_lengths[i]:
                pos = starts[i] + offset
                input_ids[i, pos] = mask_token_id
                labels[i, pos] = full_ids[i, pos]
                valid[i] = True

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        loss = functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(batch_size, seq_len)

        per_seq = loss.sum(dim=1)
        total_nll += per_seq * valid.float()
        counts += valid.float()

    avg_pll = total_nll / counts.clamp(min=1)

    results: list[tuple[float, int]] = []
    for i in range(batch_size):
        if counts[i].item() == 0:
            results.append((float("inf"), 0))
        else:
            results.append((float(avg_pll[i].item()), int(span_lengths[i])))
    return results
