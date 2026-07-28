"""Token storage width must survive the write->read round trip.

A frontier tokenizer (cl100k, Qwen) pushes vocab past 65535, so tokens no
longer fit in uint16. If prepare.py writes uint32 and load_tokens reads uint16
(or vice versa) the ids silently decode as garbage rather than erroring -- so
these guard the dtype handshake through meta.json.
"""

import json

import numpy as np

from src.data.dataset import load_tokens
from src.data.tokenizer import token_dtype


def test_dtype_follows_vocab_size():
    assert token_dtype(4096) is np.uint16
    assert token_dtype(65535) is np.uint16
    assert token_dtype(65536) is np.uint32     # first size that overflows uint16
    assert token_dtype(100352) is np.uint32    # cl100k, padded
    assert token_dtype(151680) is np.uint32    # Qwen, padded


def _write_split(tmp_path, ids, dtype):
    path = tmp_path / "train.bin"
    np.asarray(ids, dtype=dtype).tofile(path)
    (tmp_path / "meta.json").write_text(json.dumps({"dtype": np.dtype(dtype).name}))
    return str(path)


def test_uint32_round_trip_preserves_large_ids(tmp_path):
    ids = [0, 1, 65535, 65536, 100257]  # spans the uint16 boundary
    loaded = load_tokens(_write_split(tmp_path, ids, np.uint32))
    assert loaded.dtype == np.uint32
    assert loaded.tolist() == ids


def test_uint16_round_trip(tmp_path):
    ids = [0, 42, 4095]
    loaded = load_tokens(_write_split(tmp_path, ids, np.uint16))
    assert loaded.dtype == np.uint16
    assert loaded.tolist() == ids


def test_missing_meta_falls_back_to_uint16(tmp_path):
    """Data prepared before meta.json existed must still load."""
    ids = [7, 8, 9]
    path = tmp_path / "train.bin"
    np.asarray(ids, dtype=np.uint16).tofile(path)
    loaded = load_tokens(str(path))
    assert loaded.dtype == np.uint16
    assert loaded.tolist() == ids
