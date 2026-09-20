"""Ragged-decode KV-cache fast path tests (no NPU needed).

C2 of eval-gen-perf. The legacy flash_attn_with_kvcache per_row path cost
~28 device syncs per decode step (8 per-row .item() writes + .max().item()
+ per_row checks across 28 layers), ~168 mask kernels (rebuilt per layer)
and ~7.5GB of .contiguous() copy traffic (4 full cache copies per layer at
B=8). The fast path replaces all of that with: one advanced-index cache
write per layer (device-tensor indices, no sync), CPU-known end position
(KVCache._max_pos_cpu), per-step masks reused across layers, and zero-copy
strided views into the cache.

These tests pin:
  - KVCache CPU-mirror invariants (ragged flag, max pos, mask invalidation)
  - step_attn_mask == the legacy inline mask semantics (full + windowed)
  - Engine.generate_batch_prompts: fast vs legacy bit-equal token streams
  - both paths == the naive no-cache full-forward oracle (cache correctness)
  - uniform (equal-length) batches still take the legacy path and match

Opt out entirely with NANOCHAT_KV_FAST=0 (the legacy branch is kept and
exercised by these tests via the env toggle).

Run: python tests/test_kv_fastpath.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from nanochat.gpt import GPT, GPTConfig  # noqa: E402
from nanochat.engine import KVCache, Engine  # noqa: E402

BOS, ASSISTANT_END = 0, 1


def build_tiny_gpt():
    """Tiny real GPT on CPU; window_pattern SSSL exercises windowed layers."""
    cfg = GPTConfig(sequence_len=128, vocab_size=96, n_layer=2, n_head=4,
                    n_kv_head=2, n_embd=64, window_pattern="SSSL")
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    model.eval()
    return model


class StubTokenizer:
    def get_bos_token_id(self):
        return BOS

    def encode_special(self, s):
        return {"<|bos|>": BOS, "<|assistant_end|>": ASSISTANT_END,
                "<|python_start|>": 2, "<|python_end|>": 3,
                "<|output_start|>": 4, "<|output_end|>": 5}[s]

    def decode(self, tokens):
        return "".join(chr(65 + (t % 26)) for t in tokens)


def naive_generate(model, prompt, steps):
    """No-cache oracle: full forward over the whole prefix each step."""
    seq = list(prompt)
    out = []
    with torch.no_grad():
        for _ in range(steps):
            logits = model(torch.tensor([seq], dtype=torch.long))
            nxt = int(logits[0, -1].argmax())
            if nxt in (BOS, ASSISTANT_END):
                break
            seq.append(nxt)
            out.append(nxt)
    return out


def main():
    checks = []

    def check(name, ok):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        checks.append(ok)

    torch.manual_seed(1234)

    # --- 1. KVCache CPU-mirror invariants ---
    kv = KVCache(batch_size=2, seq_len=64, num_heads=2, head_dim=8,
                 num_layers=2, device=torch.device("cpu"), dtype=torch.float32)
    check("fresh cache: not ragged, max 0", kv._ragged is False and kv._max_pos_cpu == 0)
    kv.advance(3)
    check("advance(3): max 3, still uniform", kv._max_pos_cpu == 3 and not kv._ragged)

    kv2 = KVCache(batch_size=2, seq_len=64, num_heads=2, head_dim=8,
                  num_layers=2, device=torch.device("cpu"), dtype=torch.float32)
    singles = []
    for pos in (5, 9):
        c = KVCache(batch_size=1, seq_len=64, num_heads=2, head_dim=8,
                    num_layers=2, device=torch.device("cpu"), dtype=torch.float32)
        c.advance(pos)
        singles.append(c)
    kv2.merge_from_list(singles)
    check("merge [5,9]: ragged, max 9",
          kv2._ragged is True and kv2._max_pos_cpu == 9 and kv2.has_per_row_positions())
    kv2.advance(1)  # positions now [6, 10]
    check("advance(1) after merge: max 10", kv2._max_pos_cpu == 10)

    # --- 2. step_attn_mask == legacy inline mask semantics ---
    # rows at [6, 10]; full mask (window_left=-1): col <= pos_i
    m = kv2.step_attn_mask(-1, torch.float32)
    ref = torch.zeros(2, 11)
    ref[0, 7:] = -1e9   # row 0: cols 0..6 valid
    ref[1, :] = 0.0     # row 1: cols 0..10 valid (pos 10, Tk 11)
    check("full mask values == reference",
          m.shape == (2, 1, 1, 11) and torch.equal(m.squeeze(1).squeeze(1), ref))
    # windowed mask (window_left=4): also col >= pos_i - 4
    mw = kv2.step_attn_mask(4, torch.float32)
    refw = torch.zeros(2, 11)
    refw[0, :2] = -1e9   # row 0: cols 2..6 valid
    refw[0, 7:] = -1e9
    refw[1, :6] = -1e9   # row 1: cols 6..10 valid
    check("windowed mask values == reference",
          torch.equal(mw.squeeze(1).squeeze(1), refw))
    kv2.step_attn_mask(-1, torch.float32)
    check("mask cached until position change",
          kv2._step_masks is not None and len(kv2._step_masks) > 0)
    kv2.advance(1)
    check("advance invalidates mask cache", len(kv2._step_masks) == 0)

    # --- 3. ragged decode: fast vs legacy bit-equal, and both == naive oracle ---
    model = build_tiny_gpt()
    tok = StubTokenizer()
    prompts = [[10, 11, 12, 13, 14, 15, 16], [20, 21, 22, 23, 24], [30, 31, 32, 33, 34, 35, 36, 37, 38]]
    steps = 12

    outs = {}
    for label, fast in (("fast", True), ("legacy", False)):
        os.environ["NANOCHAT_KV_FAST"] = "1" if fast else "0"
        try:
            engine = Engine(model, tok)
            results, _ = engine.generate_batch_prompts(prompts, max_tokens=steps, temperature=0)
            outs[label] = [r[len(p):] for r, p in zip(results, prompts)]
        finally:
            del os.environ["NANOCHAT_KV_FAST"]

    check("ragged decode: fast == legacy (bit-equal)", outs["fast"] == outs["legacy"])

    oracle_ok = True
    for p, gen in zip(prompts, outs["fast"]):
        ref = naive_generate(model, p, steps)
        if list(gen) != ref:
            oracle_ok = False
            print(f"    row mismatch: prompt={p}")
            print(f"      fast : {list(gen)}")
            print(f"      naive: {ref}")
    check("ragged decode: fast == naive no-cache oracle", oracle_ok)

    # --- 4. uniform batch: legacy path taken, still correct ---
    uni_prompts = [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
    os.environ["NANOCHAT_KV_FAST"] = "1"
    try:
        engine = Engine(model, tok)
        results, _ = engine.generate_batch_prompts(uni_prompts, max_tokens=steps, temperature=0)
    finally:
        del os.environ["NANOCHAT_KV_FAST"]
    uni_gen = [r[len(p):] for r, p in zip(results, uni_prompts)]
    uni_ok = all(list(g) == naive_generate(model, p, steps) for p, g in zip(uni_prompts, uni_gen))
    check("uniform batch: legacy path correct vs oracle", uni_ok)

    print(f"\n{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
