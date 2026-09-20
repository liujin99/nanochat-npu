"""Stop-string early completion tests (no NPU needed).

C4 of eval-gen-perf. With stop strings active (D17 protocol), rows whose
answer is already complete keep generating to the cap because the engine
only stops on chat EOS. The pipelined path now freezes a row once a stop
string occurs in its decoded generated span (checked once per window, so
up to pipe_k tokens late).

Score-neutrality argument (what these tests pin): the scoring side re-decodes
the final token list and truncates at the FIRST stop-string occurrence
(core_eval.apply_stop_strings); BPE decode is byte-prefix monotone, so the
extra tokens generated past the occurrence are exactly the ones discarded.
Therefore scoring-visible text is identical between engine-early-stop and
generate-to-cap; the engine result is a token prefix of the legacy result.

Run: python tests/test_gen_stop_early.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from nanochat.engine import Engine  # noqa: E402
from nanochat.core_eval import apply_stop_strings  # noqa: E402
from tests.test_kv_fastpath import build_tiny_gpt, StubTokenizer, BOS, ASSISTANT_END  # noqa: E402
from tests.test_gen_pipelined import ScriptedModel  # noqa: E402

# token ids that decode to letters (CharTokenizer below): A=65 .. Z=90
S, T, O, P = 83, 84, 79, 80
A, B_, C, D = 65, 66, 67, 68
STOP_STR = "STOP"


class CharTokenizer(StubTokenizer):
    """Decodes tokens 65..90 as 'A'..'Z' so scripted tokens form readable text."""

    def decode(self, tokens):
        return "".join(chr(t) if 65 <= t <= 90 else "?" for t in tokens)


def run_engine(model, tok, prompts, max_tokens, stop_strings=None, pipe_k=None, env=None):
    old = os.environ.get("NANOCHAT_GEN_PIPELINED")
    try:
        if env is not None:
            os.environ["NANOCHAT_GEN_PIPELINED"] = env
        eng = Engine(model, tok)
        kwargs = dict(max_tokens=max_tokens, temperature=0)
        if stop_strings is not None:
            kwargs["stop_strings"] = stop_strings
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
    tok = CharTokenizer()
    prompts = [[1, 2, 3], [4, 5, 6, 7]]

    # --- 1. early stop: scoring-visible text identical, engine result is a prefix ---
    # row 0 emits "ABSTOP" then more letters; row 1 emits filler (never stops)
    script = [[A, B_, S, T, O, P, C, D, C, D], [70, 71, 72]]
    sm_stop = ScriptedModel([list(s) for s in script])
    sm_leg = ScriptedModel([list(s) for s in script])
    rs, _ = run_engine(sm_stop, tok, prompts, max_tokens=20, stop_strings=[STOP_STR], pipe_k=4)
    rl, _ = run_engine(sm_leg, tok, prompts, max_tokens=20, pipe_k=4, env="0")

    txt_s = apply_stop_strings(tok.decode(rs[0][len(prompts[0]):]), [STOP_STR])
    txt_l = apply_stop_strings(tok.decode(rl[0][len(prompts[0]):]), [STOP_STR])
    check("scoring-visible text identical (both == 'AB')", txt_s == txt_l == "AB")
    check("engine result is a token prefix of legacy", rl[0][:len(rs[0])] == rs[0])
    check("early stop actually saved tokens", len(rs[0]) < len(rl[0]))
    check("non-stopping row unaffected (runs to cap)",
          len(rs[1]) - len(prompts[1]) == 20
          and rs[1] == rl[1])

    # --- 2. EOS wins when it comes first (no interference between the two) ---
    script2 = [[A, ASSISTANT_END, S, T, O, P], [A, B_, C]]
    sm2 = ScriptedModel([list(s) for s in script2])
    r2, _ = run_engine(sm2, tok, prompts, max_tokens=10, stop_strings=[STOP_STR], pipe_k=4)
    check("EOS before stop string: row stops at EOS, EOS not recorded",
          r2[0] == [1, 2, 3, A])

    # --- 3. all rows stopped via stop strings: early exit, clean results ---
    script3 = [[S, T, O, P, C], [A, S, T, O, P]]
    sm3 = ScriptedModel([list(s) for s in script3])
    r3, _ = run_engine(sm3, tok, prompts, max_tokens=30, stop_strings=[STOP_STR], pipe_k=4)
    check("all rows stopped: both rows frozen",
          apply_stop_strings(tok.decode(r3[0][3:]), [STOP_STR]) == ""
          and apply_stop_strings(tok.decode(r3[1][4:]), [STOP_STR]) == "A"
          and len(r3[0]) - 3 < 30 and len(r3[1]) - 4 < 30)

    # --- 4. stop strings that never occur: bit-equal to no-stop pipelined ---
    model = build_tiny_gpt()
    stub = StubTokenizer()
    real_prompts = [[10, 11, 12, 13, 14, 15, 16], [20, 21, 22, 23, 24], [30, 31, 32, 33, 34, 35, 36, 37, 38]]
    r_none, m_none = run_engine(model, stub, real_prompts, 12, stop_strings=None, pipe_k=3)
    r_never, m_never = run_engine(model, stub, real_prompts, 12, stop_strings=["ZZZZQ"], pipe_k=3)
    check("never-occurring stop string: results + masks bit-equal",
          r_none == r_never and m_none == m_never)

    print(f"\n{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
