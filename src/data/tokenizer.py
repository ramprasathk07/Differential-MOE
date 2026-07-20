"""One tokenizer interface over three backends, so the rest of the pipeline
doesn't care which is in use.

The three have incompatible APIs -- a custom BPE returns `.encode(t).ids`, HF
returns a bare list, tiktoken has its own -- so everything downstream goes
through `TokenizerAdapter` instead.

A spec string selects the backend:
    data/tokenizer.json      -> custom BPE trained by train_tokenizer.py
    hf:Qwen/Qwen2.5-0.5B     -> HF AutoTokenizer (Qwen, 151,936)
    hf:Xenova/gpt-4          -> HF AutoTokenizer (cl100k, 100,277)
    hf:gpt2                  -> HF AutoTokenizer (50,257)
    tiktoken:cl100k_base     -> tiktoken, only if that package is installed

Note that anything above ~65k vocab forces uint32 token storage; see
`token_dtype()` and `src/data/prepare.py`.
"""

import json
import os
from typing import List, Sequence

import numpy as np

EOT = "<|endoftext|>"


def token_dtype(vocab_size: int):
    """uint16 tops out at 65535, so a frontier vocab needs uint32 (2x file size)."""
    return np.uint16 if vocab_size < 65536 else np.uint32


class TokenizerAdapter:
    """Uniform encode/decode/vocab_size/eot_id over whichever backend is loaded."""

    def __init__(self, backend, kind: str, spec: str):
        self._b = backend
        self.kind = kind
        self.spec = spec

    @property
    def vocab_size(self) -> int:
        if self.kind == "custom":
            return self._b.get_vocab_size()
        if self.kind == "hf":
            # len() reflects added/special tokens; config vocab_size can lag behind it
            return max(len(self._b), getattr(self._b, "vocab_size", 0))
        return self._b.n_vocab

    @property
    def eot_id(self) -> int:
        if self.kind == "custom":
            tid = self._b.token_to_id(EOT)
            assert tid is not None, "custom tokenizer is missing <|endoftext|>"
            return tid
        if self.kind == "hf":
            tid = self._b.eos_token_id
            if tid is None:
                tid = self._b.convert_tokens_to_ids(EOT)
            assert tid is not None, f"no EOS token on {self.spec}"
            return tid
        return self._b.eot_token

    def encode(self, text: str) -> List[int]:
        if self.kind == "custom":
            return self._b.encode(text).ids
        if self.kind == "hf":
            return self._b.encode(text, add_special_tokens=False)
        return self._b.encode(text, allowed_special="all")

    def decode(self, ids: Sequence[int]) -> str:
        return self._b.decode(list(ids))


def resolve_data_tokenizer(data_dir: str):
    """The tokenizer a prepared data_dir was built with, or None.

    Pretrained runs write no tokenizer.json -- the spec lives in meta.json --
    so callers must not test for that file directly.
    """
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            spec = json.load(f).get("tokenizer_spec")
        if spec:
            return load_tokenizer(spec)
    local = os.path.join(data_dir, "tokenizer.json")
    return load_tokenizer(local) if os.path.exists(local) else None


def load_tokenizer(spec: str) -> TokenizerAdapter:
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer

        return TokenizerAdapter(AutoTokenizer.from_pretrained(spec[3:]), "hf", spec)
    if spec.startswith("tiktoken:"):
        import tiktoken

        return TokenizerAdapter(tiktoken.get_encoding(spec[9:]), "tiktoken", spec)

    from tokenizers import Tokenizer

    return TokenizerAdapter(Tokenizer.from_file(spec), "custom", spec)
