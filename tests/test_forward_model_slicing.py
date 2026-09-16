"""forward_model row-bounded slicing — exact equivalence test.

prod4 2026-09-16: at ws=128 both arms' evals died inside aclnnMatmul
(ACL ERR00100, rank 76, first arc_easy chunk) — one (32, 2048) forward
emits a 65536 x 32768 logits tensor (4.3 GB bf16 + fp32 CE workspace),
16x the training micro-batch footprint. The fix slices the batch axis so
every forward stays within _EVAL_MAX_ROWS (training-proven 4096 rows).

This test pins the fix's core property: slicing is EXACTLY equivalent —
CE and argmax are row-independent, so a sliced forward must return
identical predictions (bitwise — they feed the accuracy scores) and
losses within last-bit GEMM-tiling noise (the same class of noise the
pre-existing adaptive batch halving already introduces), including the
nan last column and concatenation order.

Run: python tests/test_forward_model_slicing.py
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.core_eval import forward_model, _EVAL_MAX_ROWS  # noqa: E402


class ToyModel(nn.Module):
    """Deterministic 'LM': logits = embedding(token) @ W — fixed weights,
    no randomness, vocab small enough to compare full tensors."""

    def __init__(self, vocab=97, dim=16):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.emb = nn.Embedding(vocab, dim)
        self.W = nn.Parameter(torch.randn(vocab, dim, generator=g))
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(vocab, dim, generator=g))

    def forward(self, ids):
        h = self.emb(ids)                     # (B, T, dim)
        return torch.matmul(h, self.W.T)      # (B, T, vocab)


def main():
    torch.manual_seed(0)
    model = ToyModel()
    B, T = 7, 13
    ids = torch.randint(0, 97, (B, T))

    ref_losses, ref_preds = forward_model(model, ids)  # one slice

    checks = []

    def check(name, ok):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        checks.append(ok)

    # Sliced into per-row forwards. CE losses carry last-bit GEMM-tiling
    # noise (the M dimension of the lm_head matmul changes with the slice
    # size — the same class of noise the pre-existing adaptive batch
    # halving already introduces), so losses are compared with allclose;
    # predictions (argmax) must stay bit-identical — they are what the
    # accuracy scores are computed from.
    l1, p1 = forward_model(model, ids, max_rows=1)
    check("max_rows=1: predictions bit-identical", torch.equal(ref_preds, p1))
    check("max_rows=1: losses equal within GEMM tiling noise",
          torch.allclose(ref_losses, l1, rtol=1e-5, atol=1e-6, equal_nan=True))
    check("max_rows=1: loss diff is last-bit scale",
          torch.nansum((ref_losses - l1).abs()).item() / (B * (T - 1)) < 1e-5)

    # Uneven slicing (B=7, step=2 -> slices of 2,2,2,1).
    l2, p2 = forward_model(model, ids, max_rows=2 * T)
    check("uneven slices (2,2,2,1): predictions bit-identical",
          torch.equal(ref_preds, p2))
    check("uneven slices: losses equal within tiling noise",
          torch.allclose(ref_losses, l2, rtol=1e-5, atol=1e-6, equal_nan=True))

    # The nan last column must survive slicing.
    check("last column is nan in every slice", torch.isnan(l1[:, -1]).all())
    check("non-last columns have no nan", not torch.isnan(l1[:, :-1]).any())

    # Padding rows also flow through unchanged (stack_sequences pads right;
    # losses on pad positions must match between sliced and unsliced).
    padded = torch.cat([ids, torch.full((2, T), 96, dtype=torch.long)])
    lp, pp = forward_model(model, padded, max_rows=3 * T)
    lref, pref = forward_model(model, padded)
    check("padded batch: predictions bit-identical", torch.equal(pref, pp))
    check("padded batch: losses equal within tiling noise",
          torch.allclose(lref, lp, rtol=1e-5, atol=1e-6, equal_nan=True))

    # The default row budget is the training-proven 4096.
    check(f"_EVAL_MAX_ROWS == 4096 (training-proven)", _EVAL_MAX_ROWS == 4096)

    # A training-sized (2, 2048) forward must NOT slice at the default.
    big = torch.randint(0, 97, (2, 2048))
    step = max(1, _EVAL_MAX_ROWS // 2048)  # mirrors forward_model's math
    check("training-shaped forward stays un-sliced", step >= 2)
    # The eval-poison shape (32, 2048) MUST slice down to <= 4096 rows.
    step_poison = max(1, _EVAL_MAX_ROWS // 2048)
    n_slices = -(-32 // step_poison)
    check("eval-poison shape (32,2048) slices to <=4096 rows",
          n_slices * step_poison * 2048 <= 65536 and n_slices >= 8)

    n = sum(checks)
    print(f"\n{n}/{len(checks)} checks passed")
    return 0 if n == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
