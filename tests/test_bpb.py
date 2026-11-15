"""Bits per byte: bytes per token measured over every token of each source span, and
pooled as total bits over total bytes (each source at its own bytes/token)."""

import math

import numpy as np
import pytest
import torch

import src.data.dataset as dataset
import src.data.tokenizer as tokenizer
from src.eval import bits_per_byte, bytes_per_token_estimate, evaluate
from tests.conftest import make_config


class TwoByteForOnes:
    """Token 1 decodes to a 2-byte character, every other token to one byte."""

    def decode(self, ids):
        return "".join("é" if i == 1 else "a" for i in ids)


@pytest.fixture
def fake_split(monkeypatch):
    data = np.array([1] * 6 + [2] * 10, dtype=np.uint16)  # source A: 2 B/token, source B: 1 B/token
    monkeypatch.setattr(tokenizer, "resolve_data_tokenizer", lambda data_dir: TwoByteForOnes())
    monkeypatch.setattr(dataset, "load_tokens", lambda path: data)
    return {"A": [0, 6], "B": [6, 16]}


def test_bytes_per_token_counts_every_token_of_each_span(fake_split):
    assert bytes_per_token_estimate("unused", "unused.bin", fake_split) == {"A": 2.0, "B": 1.0}
    assert bytes_per_token_estimate("unused", "unused.bin") == pytest.approx(16 / 16 + 6 / 16)


def test_bits_per_byte_pools_bits_over_bytes():
    sums = {"A": (10.0, 4), "B": (30.0, 8)}  # nats, tokens
    pooled, per = bits_per_byte(sums, {"A": 2.0, "B": 4.0})
    assert per["A"] == pytest.approx(10 / 4 / math.log(2) / 2.0)
    assert per["B"] == pytest.approx(30 / 8 / math.log(2) / 4.0)
    assert pooled == pytest.approx((40 / math.log(2)) / (4 * 2.0 + 8 * 4.0))
    # one float for every source reduces to the old formula
    assert bits_per_byte(sums, 3.0)[0] == pytest.approx(40 / 12 / math.log(2) / 3.0)
    assert bits_per_byte(sums, {"A": 2.0}) == (None, {})  # a source without a ratio: no number
    assert bits_per_byte(sums, None) == (None, {})


def test_evaluate_uses_each_sources_ratio():
    torch.manual_seed(0)
    from src.model import Transformer

    model = Transformer(make_config())
    x = torch.randint(0, 64, (2, 16))
    batches = [(x, x, ["A", "B"])]
    r = evaluate(model, batches, {"A": 2.0, "B": 4.0}, source_names=["A", "B"])
    nll = {n: r.per_source[n]["nll"] for n in "AB"}
    assert r.per_source["A"]["bits_per_byte"] == pytest.approx(nll["A"] / math.log(2) / 2.0)
    assert r.bits_per_byte == pytest.approx((nll["A"] + nll["B"]) / math.log(2) / (2.0 + 4.0))
    # unlabelled batches with a float ratio keep working
    assert evaluate(model, [(x, x)], 2.0).bits_per_byte == pytest.approx(r.nll / math.log(2) / 2.0)
