"""A token at position t must never see positions > t."""

import torch


def test_future_tokens_dont_leak(tiny_model):
    tiny_model.eval()
    torch.manual_seed(0)
    x = torch.randint(0, tiny_model.cfg.vocab_size, (1, tiny_model.cfg.seq_len))

    with torch.no_grad():
        logits_a, _, _ = tiny_model(x)

    x2 = x.clone()
    t = tiny_model.cfg.seq_len // 2
    x2[0, t:] = (x2[0, t:] + 1) % tiny_model.cfg.vocab_size  # perturb everything from t onward

    with torch.no_grad():
        logits_b, _, _ = tiny_model(x2)

    assert torch.allclose(logits_a[0, :t], logits_b[0, :t], atol=1e-4), (
        "logits before the perturbed position changed -- attention is leaking future tokens"
    )
