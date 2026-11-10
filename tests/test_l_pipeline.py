"""L-tier pipeline: per-source corpus preparation, source-weighted sampling,
GPU-count-independent batching, weights-only export and MLflow lineage."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from src.data.dataset import SourceSampler, eval_windows
from src.data.prepare_hf import prepare_rows
from src.data.tokenizer import TokenizerAdapter
from src.model.config import TrainConfig
from src.train import export_weights, resolve_accum

SOURCES = ["web", "book", "code"]


@pytest.fixture
def sourced_corpus(tmp_path):
    words = [f"w{i}" for i in range(40)]
    vocab = {"<|endoftext|>": 0, "[UNK]": 1, **{w: i + 2 for i, w in enumerate(words)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = TokenizerAdapter(backend, "custom", "fixture")
    rng = np.random.default_rng(0)
    rows = [{"text": " ".join(rng.choice(words, rng.integers(3, 30))) + f" w{i % 40}",
             "meta": {"set": SOURCES[i % 3]}} for i in range(3000)]
    rows.append({"text": "w1 w2", "meta": {"set": "unlisted"}})
    settings = dict(dataset="fixture", text_column="text", seed=7, train_tokens=5000,
                    val_tokens=300, test_tokens=300, val_fraction=0.15, test_fraction=0.15,
                    source_column="meta.set", sources=SOURCES)
    out = tmp_path / "data"
    meta = prepare_rows(rows, tok, out, settings, model_vocab=64, seq_len=8)
    return out, meta


def test_val_and_test_are_contiguous_per_source(sourced_corpus):
    out, meta = sourced_corpus
    spans = json.loads((out / "domain_offsets.json").read_text())
    for split in ("val", "test"):
        data = np.fromfile(out / f"{split}.bin", dtype=np.uint16)
        assert list(spans[split]) == SOURCES
        assert spans[split]["web"][0] == 0 and spans[split]["code"][1] == len(data) == 900
        for lo, hi in spans[split].values():
            assert hi - lo == 300 and data[hi - 1] == 0  # each source span ends with EOS
    assert meta["eot_id"] == 0 and meta["sources"] == SOURCES
    docs = np.load(out / "train_docs.npz")
    train = np.fromfile(out / "train.bin", dtype=np.uint16)
    assert docs["starts"][0] == 0 and np.all(np.diff(docs["starts"]) > 0)
    assert np.all(train[docs["starts"][1:] - 1] == 0), "every train doc starts right after an EOS"
    assert set(docs["source"].tolist()) == {0, 1, 2}


def test_source_sampler_follows_weights(sourced_corpus):
    out, _ = sourced_corpus
    n = len(np.fromfile(out / "train.bin", dtype=np.uint16))
    sampler = SourceSampler(str(out / "train_docs.npz"), n, 8, {"book": 4.0})
    starts, src = sampler.sample(20000, torch.Generator().manual_seed(0))
    assert min(starts) >= 0 and max(starts) <= n - 9
    frac = np.bincount(src, minlength=3) / len(src)
    # book has ~1/3 of tokens; weight 4 -> 4/6 of draws
    assert abs(frac[1] - 4 / 6) < 0.03
    with pytest.raises(ValueError, match="unknown sources"):
        SourceSampler(str(out / "train_docs.npz"), n, 8, {"books": 2.0})


def test_eval_windows_split_across_sources():
    windows = eval_windows(900, 8, 12, {"web": [0, 300], "book": [300, 600], "code": [600, 900]})
    assert [n for _, n in windows].count("book") == 4
    assert all(lo <= o and o + 9 <= hi for o, n in windows
               for lo, hi in [{"web": (0, 300), "book": (300, 600), "code": (600, 900)}[n]])
    assert eval_windows(100, 8, 3) == [(0, "all"), (8, "all"), (16, "all")]


@pytest.mark.parametrize("world, accum", [(1, 64), (2, 32), (4, 16)])
def test_global_batch_is_identical_on_any_gpu_count(world, accum):
    cfg = TrainConfig(batch_size=2, global_batch_tokens=524288)
    assert resolve_accum(cfg, 4096, world) == accum
    with pytest.raises(SystemExit):
        resolve_accum(cfg, 4096, 3)
    assert resolve_accum(TrainConfig(accum_steps=5), 4096, 4) == 5  # legacy: per rank


def test_export_weights_keeps_small_tensors_exact():
    sd = {"big": torch.randn(512, 256), "lambda": torch.randn(12, 64), "step": torch.tensor(3)}
    out = export_weights(sd)
    assert out["big"].dtype == torch.bfloat16
    assert out["lambda"].dtype == torch.float32 and out["step"].dtype == torch.int64


def test_mlflow_tracker_resumes_and_indexes_checkpoints(tmp_path, monkeypatch):
    calls = SimpleNamespace(params={}, metrics=[], dicts=[], tags={}, artifacts=[], runs=[])

    def start_run(run_id=None, run_name=None):
        calls.runs.append(run_id)
        return SimpleNamespace(info=SimpleNamespace(run_id=run_id or "run-1"))

    fake = SimpleNamespace(
        set_experiment=lambda name: None, start_run=start_run,
        get_run=lambda rid: SimpleNamespace(data=SimpleNamespace(params=dict(calls.params))),
        log_params=lambda p: calls.params.update(p),
        log_metrics=lambda m, step=None: calls.metrics.append((m, step)),
        log_dict=lambda d, path: calls.dicts.append((path, json.loads(json.dumps(d)))),
        set_tags=lambda t: calls.tags.update(t),
        log_artifact=lambda path, artifact_path=None: calls.artifacts.append((path, artifact_path)),
        end_run=lambda status="FINISHED": None,
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    from src.tracking import MLflowTracker

    t = MLflowTracker(tmp_path, "exp", "l_moe", resuming=False, log_checkpoints=True)
    t.log_params({"model": {"dim": 1536}, "train": {"lr": 3e-4}})
    t.log_metrics({"train/loss": 2.5, "val/nll": float("nan"), "flag": True}, 10)
    t.log_checkpoint("best", "ckpt/best/step500.pt", {"step": 500, "tokens": 1e9, "epoch": 0.08, "val_nll": 2.4})
    assert calls.params == {"model.dim": "1536", "train.lr": "0.0003"}
    assert calls.metrics[0] == ({"train/loss": 2.5}, 10)
    assert calls.dicts[-1][1]["checkpoints"][0]["epoch"] == 0.08
    assert calls.tags["best_step"] == "500"
    assert ("ckpt/best/step500.pt", "checkpoints/best") in calls.artifacts

    again = MLflowTracker(tmp_path, "exp", "l_moe", resuming=True)
    assert calls.runs[-1] == "run-1"
    again.log_params({"model": {"dim": 1536, "new_key": 1}})  # immutable params: only new keys
    assert calls.params["model.new_key"] == "1"
