"""Eval-time one-shot weight pre-cast equivalence tests (no NPU needed).

P0 of the eval generation-perf work (branch eval-gen-perf). Training keeps
fp32 master weights and Linear.forward casts them to the input dtype on every
call (gpt.py Linear docstring). In eval decode (tiny batch, 1 token/step)
that per-op cast dominates step time and keeps a bf16 copy pool resident.
The fix casts Linear/Embedding weights to COMPUTE_DTYPE once at load time.

Value-identity argument (what these tests pin):
  - Linear: per-op cast produces the same bf16 values every call, so a
    pre-cast weight makes the cast a no-op — same matmul inputs, bit-equal
    outputs.
  - Embedding: GPT.forward casts the lookup output to COMPUTE_DTYPE right
    after lookup, so pre-cast storage is identical.
  - Per-layer scalars (resid/x0/smear/backout lambdas) are NOT touched: they
    act via 0-dim promotion / explicit .to(x.dtype).
  - Checkpoints already store wte + value_embeds in bf16 (init_weights casts
    them, gpt.py "optimizer can tolerate reduced-precision embeddings"), so
    the real cast targets are the fp32 Linear weights.

The full-model test mirrors the production NPU dtype flow on CPU: patch
nanochat.gpt.COMPUTE_DTYPE to bf16, build under meta device, to_empty,
init_weights (-> bf16 embeddings/rotary, fp32 Linears), compare logits
before/after the pre-cast bit-exactly.

End-to-end backstop = Gate A on the server: old-protocol gsm8k/math evals
must reproduce the recorded numbers byte-exactly (gsm8k 0.119030/0.642035).

Run: python tests/test_eval_weight_precast.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import nanochat.gpt as gpt_mod  # noqa: E402
from nanochat.gpt import GPT, GPTConfig, Linear  # noqa: E402
from nanochat.checkpoint_manager import precast_eval_weights  # noqa: E402

BF16 = torch.bfloat16


def build_tiny_gpt():
    """Tiny GPT in the production dtype layout (bf16 embeddings/rotary, fp32 Linears)."""
    cfg = GPTConfig(sequence_len=128, vocab_size=96, n_layer=2, n_head=4,
                    n_kv_head=2, n_embd=64, window_pattern="L")
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


def main():
    checks = []

    def check(name, ok):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        checks.append(ok)

    # --- 1. Linear: per-op cast vs pre-cast, bit-equal (the core identity) ---
    lin = Linear(16, 8, bias=False)
    x = torch.randn(4, 16).to(BF16)
    with torch.no_grad():
        ref = lin(x)  # legacy path: weight.to(bf16) inside forward
    lin.weight.data = lin.weight.data.to(BF16)
    with torch.no_grad():
        new = lin(x)  # cast is now a no-op
    check("Linear pre-cast bit-equal", torch.equal(ref, new))

    # --- 2. Embedding: pre-cast storage identical to cast-after-lookup ---
    emb = nn.Embedding(32, 16)
    gen = torch.Generator().manual_seed(7)
    ids = torch.randint(0, 32, (2, 5), generator=gen)
    ref = emb(ids).to(BF16)
    emb.weight.data = emb.weight.data.to(BF16)
    new = emb(ids)
    check("Embedding pre-cast bit-equal", torch.equal(ref, new))

    # --- 3. Full GPT model: logits bit-equal across the pre-cast ---
    old_dtype = gpt_mod.COMPUTE_DTYPE
    gpt_mod.COMPUTE_DTYPE = BF16
    try:
        model = build_tiny_gpt()

        # production layout checks before the cast
        check("wte already bf16 from init (checkpoint layout)",
              model.transformer.wte.weight.dtype == BF16)
        n_linear = sum(isinstance(m, nn.Linear) for m in model.modules())
        lin_fp32 = [m for m in model.modules()
                    if isinstance(m, nn.Linear) and m.weight.dtype == torch.float32]
        check("all Linear weights start fp32", len(lin_fp32) == n_linear and n_linear > 0)

        gen = torch.Generator().manual_seed(1234)
        ids = torch.randint(0, model.config.vocab_size, (1, 32), generator=gen)
        with torch.no_grad():
            ref = model(ids)  # legacy: per-op casts on fp32 Linears

        n_cast, n_already = precast_eval_weights(model, dtype=BF16, selfcheck=False)

        with torch.no_grad():
            new = model(ids)
        check("GPT logits bit-equal across pre-cast", torch.equal(ref, new))
        check("cast count == fp32 Linears", n_cast == len(lin_fp32))
        check("already-cast count == bf16 embeddings (wte + value_embeds)",
              n_already == sum(isinstance(m, nn.Embedding) for m in model.modules()))

        # scalars keep their checkpoint dtype
        scalars_ok = (model.resid_lambdas.dtype == torch.float32
                      and model.x0_lambdas.dtype == torch.float32
                      and model.smear_lambda.dtype == torch.float32
                      and model.backout_lambda.dtype == torch.float32)
        check("scalar lambdas untouched (fp32)", scalars_ok)

        # idempotence: nothing left to cast
        n_cast2, _ = precast_eval_weights(model, dtype=BF16, selfcheck=False)
        check("idempotent (second call casts nothing)", n_cast2 == 0)

        # --- 4. self-check path passes on a healthy model ---
        model2 = build_tiny_gpt()
        try:
            precast_eval_weights(model2, dtype=BF16, selfcheck=True)
            check("self-check passes on healthy model", True)
        except RuntimeError as e:
            print(f"    self-check raised: {e}")
            check("self-check passes on healthy model", False)

        # --- 5. env gate: NANOCHAT_EVAL_WEIGHTCAST=0 is an exact no-op ---
        model3 = build_tiny_gpt()
        fp32_before = [m.weight.dtype for m in model3.modules() if isinstance(m, nn.Linear)]
        os.environ["NANOCHAT_EVAL_WEIGHTCAST"] = "0"
        try:
            n_c, n_a = precast_eval_weights(model3, dtype=BF16)
            fp32_after = [m.weight.dtype for m in model3.modules() if isinstance(m, nn.Linear)]
            check("env gate disables cast", n_c == 0 and n_a == 0 and fp32_before == fp32_after)
        finally:
            del os.environ["NANOCHAT_EVAL_WEIGHTCAST"]

        # --- 6. fp32 compute dtype: Linears untouched; the bf16 embeddings
        # get an upcast that is value-identical to the forward's .to() ---
        model4 = build_tiny_gpt()
        n_c, n_a = precast_eval_weights(model4, dtype=torch.float32, selfcheck=False)
        n_emb = sum(isinstance(m, nn.Embedding) for m in model4.modules())
        lins_fp32 = all(m.weight.dtype == torch.float32 for m in model4.modules()
                        if isinstance(m, nn.Linear))
        check("fp32 target: only embeddings upcast, Linears untouched",
              n_c == n_emb and n_a == n_linear and lins_fp32)
    finally:
        gpt_mod.COMPUTE_DTYPE = old_dtype

    print(f"\n{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
