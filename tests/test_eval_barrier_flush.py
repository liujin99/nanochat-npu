"""EL0004-at-first-collective regression test (prod4 2026-09-17 anchor).

The base_eval_check anchor (job b9e3def2) died at the FIRST task's final
dist.barrier() — HCCL could not allocate its 401MiB comm buffer
(Memory_Allocation_Failure EL0004, 420478976 bytes, rank 6): the torch
pool held everything up to its ceiling after the chunk loop, and the
collective's comm buffers are allocated OUTSIDE that pool. Same failure
class as the 2026-08-28 speedrun Step 7 (401MiB, core_eval.py:412 then).
The generation path already flushed before its collectives; the MC
path's barrier was bare.

Fix under test: evaluate_task flushes the accelerator cache between the
chunk loop and the ws>1 collectives.

Run: python tests/test_eval_barrier_flush.py
"""
import os
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nanochat.core_eval as ce  # noqa: E402

checks = []


def check(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    checks.append(ok)


# ── 1. source placement: the flush sits between the chunk loop and the
#       ws>1 barrier inside evaluate_task (the device branch is inline,
#       matching the generation path's existing flush style) ──────────────
src = open(os.path.abspath(ce.__file__)).read()
fn = src.index("def evaluate_task")
loop_end = src.index("chunk_start += len(chunk)", fn)
barrier = src.index("dist.barrier()", fn)
window = src[loop_end:barrier]
check("evaluate_task: accelerator flush precedes the ws>1 barrier",
      loop_end < barrier and "empty_cache()" in window)
check("evaluate_task: the flush covers both npu and cuda",
      "torch.npu.empty_cache()" in window
      and "torch.cuda.empty_cache()" in window)


# ── 2. behavioral smoke: fake dist (ws=2) — collective order intact and
#       the task still evaluates end-to-end with the flush in place ──────
class ToyModel(nn.Module):
    def __init__(self, vocab=97, dim=16):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.emb = nn.Embedding(vocab, dim)
        self.W = nn.Parameter(torch.randn(vocab, dim, generator=g))
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(vocab, dim, generator=g))

    def forward(self, ids):
        return torch.matmul(self.emb(ids), self.W.T)


class FakeTok:
    def get_bos_token_id(self):
        return 0

    def __call__(self, prompts, prepend=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        return [[(prepend if prepend is not None else 0)]
                + [1 + (ord(c) % 90) for c in p] for p in prompts]


class FakeDist:
    ReduceOp = types.SimpleNamespace(SUM="sum")

    def __init__(self):
        self.log = []

    def is_initialized(self):
        return True

    def get_rank(self):
        return 0

    def get_world_size(self):
        return 2

    def barrier(self):
        self.log.append("barrier")

    def all_reduce(self, tensor, op=None):
        self.log.append("all_reduce")


data = [{"query": f"Question number {i} is",
         "choices": [" yes", " no", " maybe"],
         "gold": i % 3} for i in range(4)]
task_meta = {"task_type": "multiple_choice", "num_fewshot": 0,
             "continuation_delimiter": " Answer:"}

fake_dist = FakeDist()
orig_dist = ce.dist
ce.dist = fake_dist
try:
    acc, nll = ce.evaluate_task(ToyModel(), FakeTok(), data,
                                torch.device("cpu"), task_meta,
                                eval_batch_size=2)
finally:
    ce.dist = orig_dist

check("evaluate_task(ws=2): collective order barrier -> all_reduce x2",
      fake_dist.log == ["barrier", "all_reduce", "all_reduce"])
check("evaluate_task(ws=2): finite means",
      torch.isfinite(torch.tensor(acc)) and torch.isfinite(torch.tensor(nll)))

real_state = ce.dist
ce.dist = types.SimpleNamespace(
    is_initialized=lambda: False, ReduceOp=types.SimpleNamespace(SUM="sum"))
try:
    acc1, _ = ce.evaluate_task(ToyModel(), FakeTok(), data,
                               torch.device("cpu"), task_meta, eval_batch_size=2)
    check("evaluate_task(ws=1): evaluates without any collective",
          0.0 <= acc1 <= 1.0)
finally:
    ce.dist = real_state

print(f"\n{'=' * 50}")
print(f"{'PASS' if all(checks) else 'FAIL'}: {sum(checks)}/{len(checks)} checks")
sys.exit(0 if all(checks) else 1)
