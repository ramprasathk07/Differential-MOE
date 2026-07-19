"""Shared constants and file access for the BabyLM Challenge corpus.

The corpus ships as one plain .txt file per domain (no HF `datasets` loading
script -- those are no longer supported by the `datasets` library, so we fetch
files directly via `huggingface_hub` and read them ourselves). Each file is one
line per utterance/sentence; lines are naturally sequential within a domain
(subtitle lines, dialogue turns, prose) so they're kept newline-joined, with a
single <|endoftext|> only at each *domain* boundary -- not after every line.

Domains are intentionally left in their native format (CHILDES/Switchboard
keep their `*CHI:\t` / `A:\t` speaker-turn tags) -- this is the data as
released for the challenge, not a cleaned derivative.

Two tracks, matching this repo's tier A / tier B split (docs/plan.md):
  strict-small (10M words) -- tier A, the fast ablation workhorse
  strict       (100M words) -- tier B, the headline run
"""

from typing import Iterator, List

from huggingface_hub import hf_hub_download

DOMAINS = ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"]

TRAIN_REPOS = {
    "strict-small": "BabyLM-community/BabyLM-2026-Strict-Small",
    "strict": "BabyLM-community/BabyLM-2026-Strict",
}
DEV_REPO = "BabyLM-community/BabyLM-dev"
TEST_REPO = "BabyLM-community/BabyLM-Test"


def domain_lines(repo_id: str, filename: str, max_lines: int = 0) -> List[str]:
    path = hf_hub_download(repo_id, filename, repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    if max_lines:
        lines = lines[:max_lines]
    return [line for line in lines if line]


def iter_domains(track: str, split: str, max_lines: int = 0) -> Iterator[tuple]:
    """Yields (domain_name, list_of_lines) for every domain in a track/split.

    split: "train" (from the track's own repo), "dev" (shared BabyLM-dev repo --
    used for periodic in-training validation / checkpoint selection), or "test"
    (shared BabyLM-Test repo -- held out, evaluate ONLY ONCE after training is
    finished and the checkpoint is already chosen; never used for any decision
    made during training).
    """
    if split == "train":
        repo_id = TRAIN_REPOS[track]
        filenames = [f"{d}.train.txt" for d in DOMAINS]
    elif split == "dev":
        repo_id = DEV_REPO
        filenames = [f"{d}.dev" for d in DOMAINS]
    elif split == "test":
        repo_id = TEST_REPO
        filenames = [f"{d}.test" for d in DOMAINS]
    else:
        raise ValueError(f"unknown split: {split!r}")

    for domain, filename in zip(DOMAINS, filenames):
        yield domain, domain_lines(repo_id, filename, max_lines)
