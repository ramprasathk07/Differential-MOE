# Preparing the tokenized data off-Kaggle

Tokenizing is pure CPU work. Running it inside a Kaggle GPU session burns quota
on something a laptop does just as fast, and it has to be redone every session
because `/kaggle/working` is wiped. Do it once locally, upload the result as a
Kaggle Dataset, and every later session starts at the training cell.

Roughly what it costs on the `strict` track (100M words) with cl100k:

| | |
|---|---|
| source text downloaded | ~700 MB |
| `train.bin` | ~460 MB (uint32 — a frontier vocab cannot fit uint16) |
| `val.bin` + `test.bin` | ~90 MB |
| wall-clock | tens of minutes, CPU-bound |

## 1. Tokenize locally

```bash
python -m src.data.prepare --tokenizer hf:Xenova/gpt-4 --out_dir data_s --track strict
```

Swap in `hf:Qwen/Qwen2.5-0.5B` for Qwen — then set `vocab_size: 151680` in the
`s_*.yaml` configs, since the embedding table has to cover the larger vocabulary.

## 2. Verify before uploading

Tokenized data fails silently: a wrong dtype or a truncated write does not
raise, it decodes as noise and surfaces hours into a run as a loss that will
not come down. Uploading half a gigabyte and finding that out on Kaggle wastes
exactly the time this is meant to save.

```bash
python -m src.data.verify --data_dir data_s --config configs/s_dense.yaml
```

This checks each split loads at the dtype `meta.json` claims, that no token id
can index past the embedding table, that domain offsets tile the file with no
gaps, that byte sizes match token counts, and that ids decode back to readable
text. It exits non-zero on any failure.

## 3. Upload

`data_s/` should contain `train.bin`, `val.bin`, `test.bin`, `meta.json`, and
`domain_offsets.json`. Keep all five — `meta.json` carries the dtype and
tokenizer spec that the loader reads back, and without it the data is
ambiguous.

**Web UI**: kaggle.com → Datasets → New Dataset → drag the folder in.

**CLI** (faster for this size):
```bash
pip install kaggle          # needs ~/.kaggle/kaggle.json credentials
kaggle datasets create -p data_s --dir-mode zip
```
using a `data_s/dataset-metadata.json` like:
```json
{
  "title": "BabyLM strict cl100k tokenized",
  "id": "YOUR_USERNAME/babylm-strict-cl100k",
  "licenses": [{"name": "CC0-1.0"}]
}
```

## 4. Point the notebook at it

Attach the dataset to the notebook (right panel → Add Input), then in the
settings cell set:

```python
DATA_DIR = '/kaggle/input/babylm-strict-cl100k'
```

The data-prep cell sees `train.bin` already present and skips straight past.
`/kaggle/input` is read-only, which is fine — training only ever reads the data
and writes checkpoints to `--out_dir` under `/kaggle/working`.

Re-uploading is only necessary if the tokenizer or track changes. A different
model config, learning rate, or step count reuses the same dataset.
