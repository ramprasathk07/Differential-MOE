"""Print raw vs active parameter counts for a config. Never hand-compute these
numbers for README/blog tables -- always regenerate from here.

Usage:
    python -m src.params --config configs/a_dense.yaml
    python -m src.params --config configs/*.yaml   (shell-expanded)
"""

import argparse

from src.model import Config, Transformer, count_params


def fmt(n: int) -> str:
    return f"{n / 1e6:.2f}M"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, nargs="+", required=True)
    args = ap.parse_args()

    rows = []
    for path in args.config:
        cfg = Config.from_yaml(path)
        model = Transformer(cfg.model)
        counts = count_params(model)
        rows.append((path, cfg.model.attention, cfg.model.ffn, counts))

    path_width = max(28, max(len(path) for path, *_ in rows) + 2)
    attention_width = max(14, max(len(attn) for _, attn, *_ in rows) + 2)
    ffn_width = max(8, max(len(ffn) for _, _, ffn, _ in rows) + 2)
    header = (
        f"{'config':<{path_width}}{'attn':<{attention_width}}{'ffn':<{ffn_width}}"
        f"{'raw':>10}{'active':>10}{'non-emb raw':>14}{'non-emb active':>16}"
    )
    print(header)
    print("-" * len(header))
    for path, attn, ffn, c in rows:
        print(
            f"{path:<{path_width}}{attn:<{attention_width}}{ffn:<{ffn_width}}"
            f"{fmt(c['total']):>10}{fmt(c['active']):>10}"
            f"{fmt(c['non_embed_total']):>14}{fmt(c['non_embed_active']):>16}"
        )


if __name__ == "__main__":
    main()
