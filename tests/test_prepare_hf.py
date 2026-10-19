"""Guard corpus split isolation and the HF-to-memmap storage handshake offline."""

import json

import numpy as np
import pytest
from tokenizers import Tokenizer, models

from src.data.dataset import eval_batches, get_batch, load_tokens
from src.data.prepare_hf import document_split, prepare_rows
from src.data.tokenizer import TokenizerAdapter


@pytest.fixture
def corpus(tmp_path):
    texts = [f"document_{i}" for i in range(300)]
    vocab = {"<|endoftext|>": 0, "[UNK]": 1}
    vocab.update({text: i + 2 for i, text in enumerate(texts)})
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    path = tmp_path / "tokenizer.json"
    backend.save(str(path))
    tok = TokenizerAdapter(backend, "custom", str(path))
    settings = dict(
        dataset="offline-fixture", text_column="text", seed=42,
        train_tokens=29, val_tokens=17, test_tokens=13,
        val_fraction=0.2, test_fraction=0.2,
    )
    # Repeated text must never be reassigned after another split fills up.
    rows = [{"text": text} for text in texts for _ in range(3)]
    return tok, settings, rows


def test_packing_is_disjoint_reproducible_and_trainable(tmp_path, corpus):
    import torch

    tok, settings, rows = corpus
    outputs = [tmp_path / "first", tmp_path / "second"]
    for out in outputs:
        prepare_rows(rows, tok, out, settings, model_vocab=tok.vocab_size, seq_len=4)
    meta = json.loads((outputs[0] / "meta.json").read_text())
    offsets = json.loads((outputs[0] / "domain_offsets.json").read_text())
    payloads = {}
    for split in ("train", "val", "test"):
        name = f"{split}.bin"
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
        data = load_tokens(str(outputs[0] / name))
        assert data.dtype == np.uint16
        assert len(data) == settings[f"{split}_tokens"] == meta["tokens"][split]
        assert data[-1] == tok.eot_id
        assert offsets[split]["offline-fixture"] == [0, len(data)]
        payloads[split] = set(data.tolist()) - {tok.eot_id}
        for token in payloads[split]:
            text = tok.decode([token])
            assert document_split(text, 42, 0.2, 0.2) == split
        x, y = get_batch(data, 2, 4, torch.Generator().manual_seed(42), "cpu")
        assert x.shape == y.shape == (2, 4)
        assert torch.equal(x[:, 1:], y[:, :-1])
        assert list(eval_batches(data, 2, 4, 2, "cpu"))
        del data
    assert payloads["train"].isdisjoint(payloads["val"] | payloads["test"])
    assert payloads["val"].isdisjoint(payloads["test"])


def test_content_hash_ignores_outer_whitespace():
    for i in range(100):
        text = f"a document {i}"
        assert document_split(text, 42, 0.2, 0.2) == document_split(f"  {text}\n", 42, 0.2, 0.2)


def test_exhaustion_does_not_publish_partial_corpus(tmp_path, corpus):
    tok, settings, rows = corpus
    out = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="exhausted"):
        prepare_rows(rows[:1], tok, out, settings, model_vocab=tok.vocab_size, seq_len=4)
    assert list(out.iterdir()) == []


def test_existing_data_is_preserved(tmp_path, corpus):
    tok, settings, rows = corpus
    out = tmp_path / "existing"
    out.mkdir()
    (out / "train.bin").write_bytes(b"existing corpus")
    with pytest.raises(FileExistsError):
        prepare_rows(rows, tok, out, settings, model_vocab=tok.vocab_size, seq_len=4)
    assert (out / "train.bin").read_bytes() == b"existing corpus"


def test_vocab_mismatch_fails_before_writing(tmp_path, corpus):
    tok, settings, rows = corpus
    out = tmp_path / "mismatch"
    with pytest.raises(ValueError, match="exceeds model vocab"):
        prepare_rows(rows, tok, out, settings, model_vocab=32, seq_len=4)
    assert not out.exists()


def test_invalid_token_cannot_wrap_into_uint16(tmp_path, corpus, monkeypatch):
    tok, settings, rows = corpus
    monkeypatch.setattr(tok, "encode", lambda text: [65536])
    out = tmp_path / "overflow"
    with pytest.raises(ValueError, match="outside the tokenizer vocabulary"):
        prepare_rows(rows, tok, out, settings, model_vocab=tok.vocab_size, seq_len=4)
    assert list(out.iterdir()) == []


@pytest.mark.parametrize("change", [
    {"val_fraction": 0}, {"test_fraction": 0.9}, {"train_tokens": 4},
])
def test_invalid_budget_or_split_fails_early(tmp_path, corpus, change):
    tok, settings, rows = corpus
    settings.update(change)
    with pytest.raises(ValueError):
        prepare_rows(rows, tok, tmp_path / "invalid", settings,
                     model_vocab=tok.vocab_size, seq_len=4)
