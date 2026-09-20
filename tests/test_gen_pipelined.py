"""Pipelined greedy decode tests (no NPU needed).

C3 of eval-gen-perf. The legacy generate_batch_prompts loop did a
device->CPU .tolist() plus a CPU->device torch.tensor() on EVERY decode
step, serializing CPU and NPU (the CPU can never run ahead). The pipelined
path runs pipe_k forwards with sampled ids living entirely on device and
harvests once per window — bit-identical token streams by construction
(same argmax over the same logits, same tokens fed in the same order).

These tests pin:
  - real tiny GPT: pipelined == legacy bit-equal (results AND masks),
    and both == the naive no-cache full-forward oracle
  - deterministic scripted model: EOS mid-window, all-rows-EOS early exit,
    exact max_tokens boundary with pipe_k not dividing max_tokens
  - pipelined=True with temperature>0 is rejected
  - NANOCHAT_GEN_PIPELINED=0 kill switch takes the legacy path

Run: python tests/test_gen_pipelined.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from dataclasses import dataclass  # noqa: E402

from nanochat.engine import Engine  # noqa: E402
from tests.test_kv_fastpath import (  # noqa: E402
    build_tiny_gpt, StubTokenizer, naive_generate, BOS, ASSISTANT_END,
)


@dataclass
class ScriptedConfig:
    n_kv_head: int = 2
    n_head: int = 4
    n_embd: int = 64
    n_layer: int = 2
    sequence_len: int = 128


class ScriptedModel:
    """Argmax follows a per-row script (one token per forward call, prefill
    included). After a row's script is exhausted it emits filler tokens that
    are never EOS, so rows run to max_tokens deterministically."""

    def __init__(self, scripts, vocab_size=96):
        self.scripts = [list(s) for s in scripts]
        self.vocab_size = vocab_size
        self.config = ScriptedConfig()
        self._device = torch.device("cpu")
        self.steps = [0] * len(scripts)
        self.prefills_done = 0

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None):
        B, T = ids.shape
        if kv_cache is not None:
            kv_cache.advance(T)
        logits = torch.full((B, T, self.vocab_size), -10.0)
        if B == 1 and self.prefills_done < len(self.scripts):
            # serial prefill call for script row `prefills_done` (tensor row 0)
            r_script = self.prefills_done
            self.prefills_done += 1
            s = self.scripts[r_script]
            tok = s[self.steps[r_script]] if self.steps[r_script] < len(s) else 42
            logits[0, -1, tok] = 10.0
            self.steps[r_script] += 1
            return logits
        for r in range(B):
            s = self.scripts[r]
            tok = s[self.steps[r]] if self.steps[r] < len(s) else 42
            logits[r, -1, tok] = 10.0
            self.steps[r] += 1
        return logits


def run_engine(model, tok, prompts, max_tokens, pipe_k=None, env=None):
    old = os.environ.get("NANOCHAT_GEN_PIPELINED")
    try:
        if env is not None:
            os.environ["NANOCHAT_GEN_PIPELINED"] = env
        eng = Engine(model, tok)
        kwargs = dict(max_tokens=max_tokens, temperature=0)
        if pipe_k is not None:
            kwargs["pipe_k"] = pipe_k
        results, masks = eng.generate_batch_prompts([list(p) for p in prompts], **kwargs)
        return results, masks
    finally:
        if env is not None:
            if old is None:
                os.environ.pop("NANOCHAT_GEN_PIPELINED", None)
            else:
                os.environ["NANOCHAT_GEN_PIPELINED"] = old


def main():
    checks = []

    def check(name, ok):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        checks.append(ok)

    torch.manual_seed(1234)

    # --- 1. real tiny GPT: pipelined == legacy == naive oracle ---
    model = build_tiny_gpt()
    tok = StubTokenizer()
    prompts = [[10, 11, 12, 13, 14, 15, 16], [20, 21, 22, 23, 24], [30, 31, 32, 33, 34, 35, 36, 37, 38]]
    steps = 12

    res_pipe, mask_pipe = run_engine(model, tok, prompts, steps, pipe_k=3)
    res_leg, mask_leg = run_engine(model, tok, prompts, steps, pipe_k=3, env="0")
    check("real GPT: pipelined == legacy results (bit-equal)", res_pipe == res_leg)
    check("real GPT: pipelined == legacy masks", mask_pipe == mask_leg)
    oracle_ok = all(
        list(r[len(p):]) == naive_generate(model, p, steps)
        for r, p in zip(res_pipe, prompts)
    )
    check("real GPT: pipelined == naive no-cache oracle", oracle_ok)

    # --- 2. scripted EOS mid-window (row 0 EOS at position 2 of a K=8 window) ---
    eos = ASSISTANT_END
    scripts = [[5, 6, eos, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    sm_pipe = ScriptedModel([list(s) for s in scripts])
    sm_leg = ScriptedModel([list(s) for s in scripts])
    prompts2 = [[1, 2, 3], [4, 5, 6, 7]]
    rp, _ = run_engine(sm_pipe, tok, prompts2, max_tokens=8, pipe_k=8)
    rl, _ = run_engine(sm_leg, tok, prompts2, max_tokens=8, pipe_k=8, env="0")
    check("scripted: EOS mid-window == legacy", rp == rl)
    exp0 = [1, 2, 3, 5, 6]          # row 0: emits 5,6 then EOS (not recorded)
    check("scripted: row 0 stops at EOS, EOS not recorded", rp[0] == exp0)

    # --- 3. all rows EOS mid-window: early exit, both modes equal ---
    scripts3 = [[5, eos, 7], [9, eos, 11]]
    sm3p = ScriptedModel([list(s) for s in scripts3])
    sm3l = ScriptedModel([list(s) for s in scripts3])
    rp3, _ = run_engine(sm3p, tok, prompts2, max_tokens=8, pipe_k=8)
    rl3, _ = run_engine(sm3l, tok, prompts2, max_tokens=8, pipe_k=8, env="0")
    check("scripted: all-EOS early exit == legacy", rp3 == rl3)
    check("scripted: all-EOS results frozen at EOS",
          rp3[0] == [1, 2, 3, 5] and rp3[1] == [4, 5, 6, 7, 9])

    # --- 4. exact max_tokens boundary (pipe_k=3 does not divide 5) ---
    scripts4 = [[50, 51], [60, 61]]
    sm4p = ScriptedModel([list(s) for s in scripts4])
    sm4l = ScriptedModel([list(s) for s in scripts4])
    rp4, _ = run_engine(sm4p, tok, prompts2, max_tokens=5, pipe_k=3)
    rl4, _ = run_engine(sm4l, tok, prompts2, max_tokens=5, pipe_k=3, env="0")
    check("scripted: max_tokens boundary == legacy", rp4 == rl4)
    check("scripted: exactly max_tokens tokens, none beyond",
          all(len(r) - len(p) == 5 for r, p in zip(rp4, prompts2))
          and rp4[0] == [1, 2, 3, 50, 51, 42, 42, 42])

    # --- 5. mode guards ---
    try:
        Engine(model, tok).generate_batch_prompts(
            [list(prompts[0])], max_tokens=4, temperature=1.0, pipelined=True)
        check("pipelined+temperature>0 rejected", False)
    except AssertionError:
        check("pipelined+temperature>0 rejected", True)

    res_off, _ = run_engine(model, tok, prompts, steps, pipe_k=3, env="0")
    res_on, _ = run_engine(model, tok, prompts, steps, pipe_k=3)
    check("env kill switch: NANOCHAT_GEN_PIPELINED=0 == on", res_off == res_on)

    print(f"\n{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
