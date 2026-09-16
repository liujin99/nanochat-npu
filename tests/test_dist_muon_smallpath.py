"""DistMuonAdamW small-group path (all_reduce/broadcast) vs stacked path —
gloo CPU equivalence test with THREE anchors.

Background (prod4 2026-09-15, job 10c9fc37): the stacked reduce_scatter pads
a Muon group to chunk_size * world_size slots, so a group SMALLER than
world_size costs world_size x shape comm memory regardless of n — d28's
28-layer groups at ws=64 needed 4 GiB bf16 and the first optimizer.step()
OOM'd all 64 ranks. The fix routes small groups (n < world_size, multinode
only, _SMALL_PATH_MIN_WS) through per-param all_reduce + round-robin
ownership + owner-broadcast, plus builds the stacked path's buffer in place
(no torch.stack transient).

Anchors:
  1. OLD implementation (git HEAD) vs NEW, bitwise — plumbing equivalence.
  2. Single-GPU MuonAdamW (untouched reference class) fed PRE-AVERAGED
     grads vs BOTH distributed paths, bitwise — semantic equivalence
     against the canonical algorithm, independent of the old dist code.
  3. Cross-rank consistency (data-parallel invariant).

Dyadic-valued gradients (multiples of 1/4 with rank offsets) make AVG exact
in any reduction order at power-of-two world sizes, so bitwise comparison
is meaningful. A sync-shim covers gloo's missing Work.getFuture (semantics
identical; applied uniformly).

Run: python tests/test_dist_muon_smallpath.py   (also pytest-compatible)
"""
import os
import subprocess
import sys
import tempfile
import importlib.util

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("TEST_PORT", "29633"))

HP_MUON = dict(lr=0.02, momentum=0.95, ns_steps=5, beta2=0.99, weight_decay=0.01)
HP_ADAMW = dict(lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_impls():
    """Old = git HEAD (pre-fix), New = working tree."""
    with tempfile.NamedTemporaryFile(
            suffix="_optim_old.py", delete=False, mode="w") as f:
        old_src = subprocess.run(
            ["git", "-C", REPO, "show", "HEAD:nanochat/optim.py"],
            capture_output=True, text=True, check=True).stdout
        f.write(old_src)
        old_path = f.name
    new_path = os.path.join(REPO, "nanochat", "optim.py")
    return (_load_module("optim_old", old_path),
            _load_module("optim_new", new_path), old_path)


def build_params(sizes):
    """Same-seed identical init across ranks AND the single-GPU reference.
    sizes: tuple of (n, shape) for the Muon groups."""
    torch.manual_seed(1234)
    def mk(n, shape):
        return [torch.nn.Parameter(
            torch.randn(shape, dtype=torch.float32) * 0.1) for _ in range(n)]
    groups = [dict(params=mk(n, shape), kind="muon", **HP_MUON)
              for n, shape in sizes]
    adamw = dict(params=[torch.nn.Parameter(torch.randn(64) * 0.1)],
                 kind="adamw", **HP_ADAMW)
    return groups, adamw


def _base_grads(groups, adamw, step):
    """Deterministic dyadic base values; identical sequence everywhere."""
    g = torch.Generator().manual_seed(10_000 + step)
    bases = []
    for grp in groups + [adamw]:
        for p in grp["params"]:
            bases.append(torch.randint(-7, 8, p.shape, generator=g).float() / 4.0)
    return bases


def set_grads(groups, adamw, step, rank):
    """rank-aware grads: base + rank*0.25 + step*0.5 (all dyadic)."""
    bases = _base_grads(groups, adamw, step)
    k = 0
    for grp in groups + [adamw]:
        for p in grp["params"]:
            p.grad = bases[k] + rank * 0.25 + step * 0.5
            k += 1


def set_avg_grads(groups, adamw, step, world):
    """pre-averaged grads for the single-GPU reference: base + (ws-1)/2*0.25
    + step*0.5 — the exact AVG the distributed ranks compute."""
    bases = _base_grads(groups, adamw, step)
    off = (world - 1) / 2 * 0.25 + step * 0.5
    k = 0
    for grp in groups + [adamw]:
        for p in grp["params"]:
            p.grad = bases[k] + off
            k += 1


class _SyncWork:
    """gloo's Work lacks getFuture (HCCL-only in production). Run the
    collective synchronously and hand back an already-completed future —
    identical semantics, only the async overlap is lost (irrelevant for
    equivalence testing; applied uniformly to old and new impls)."""
    def __init__(self):
        self._fut = torch.futures.Future()
        self._fut.set_result(True)

    def get_future(self):
        return self._fut

    def wait(self):
        return True


def _patch_dist_sync():
    def wrap(fn):
        def w(*args, **kwargs):
            kwargs["async_op"] = False
            fn(*args, **kwargs)
            return _SyncWork()
        return w
    for name in ("all_reduce", "reduce_scatter_tensor", "broadcast",
                 "all_gather_into_tensor"):
        setattr(dist, name, wrap(getattr(dist, name)))


def _flat(groups, adamw):
    return torch.cat([p.detach().reshape(-1)
                      for grp in groups + [adamw] for p in grp["params"]])


def run_rank(rank, world, use_new, gate, port, outdir, sizes):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    _patch_dist_sync()
    old_mod, new_mod, _ = load_impls()
    mod = new_mod if use_new else old_mod
    groups, adamw = build_params(sizes)
    opt = mod.DistMuonAdamW(groups + [adamw])
    if use_new and gate is not None:
        opt._SMALL_PATH_MIN_WS = gate  # instance attr shadows the class default
    for step in range(3):
        set_grads(groups, adamw, step, rank)
        opt.step()
    torch.save(_flat(groups, adamw), os.path.join(outdir, f"rank{rank}.pt"))
    dist.destroy_process_group()


def run_impl(use_new, gate, world, port, sizes):
    outdir = tempfile.mkdtemp(prefix="muon_eq_")
    mp.start_processes(run_rank,
                       args=(world, use_new, gate, port, outdir, sizes),
                       nprocs=world, join=True, start_method="spawn")
    return [torch.load(os.path.join(outdir, f"rank{r}.pt"))
            for r in range(world)]


def single_gpu_reference(world, sizes):
    """Anchor 2: the untouched single-GPU class on pre-averaged grads."""
    _, new_mod, _ = load_impls()
    groups, adamw = build_params(sizes)
    opt = new_mod.MuonAdamW(groups + [adamw])
    for step in range(3):
        set_avg_grads(groups, adamw, step, world)
        opt.step()
    return _flat(groups, adamw)


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""),
          flush=True)
    return ok


def main():
    r = []
    BASE = ((3, (8, 8)), (9, (12, 5)), (2, (5, 12)))

    # ── anchor 1+3: old vs new, gate off (stacked both; validates the
    #    in-place stack build bitwise)
    old = run_impl(False, None, 4, PORT, BASE)
    new = run_impl(True, None, 4, PORT + 1, BASE)
    r.append(check("ws=4 gate=8 (stacked both): old == new bitwise",
                   all(torch.equal(a, b) for a, b in zip(old, new))))
    r.append(check("ws=4: ranks consistent within impl",
                   all(torch.equal(old[0], o) for o in old)
                   and all(torch.equal(new[0], n) for n in new)))

    # ── anchor 1+3: gate on (small path n<4) — bitwise vs old
    new_small = run_impl(True, 1, 4, PORT + 2, BASE)
    r.append(check("ws=4 gate=1 (small path n<4): old == new bitwise",
                   all(torch.equal(a, b) for a, b in zip(old, new_small))))
    r.append(check("ws=4 small path: ranks consistent",
                   all(torch.equal(new_small[0], n) for n in new_small)))

    # ── anchor 2: single-GPU reference on pre-averaged grads, both impls
    ref = single_gpu_reference(4, BASE)
    r.append(check("ws=4: single-GPU reference == dist old (bitwise)",
                   torch.equal(ref, old[0])))
    r.append(check("ws=4: single-GPU reference == dist new small path (bitwise)",
                   torch.equal(ref, new_small[0])))

    # ── boundary sizes at ws=8: n = 7 (small), 8 (stacked, zero pad),
    #    9 (stacked, chunk=2 padded=16) — gate on
    B8 = ((7, (8, 8)), (8, (6, 6)), (9, (12, 5)))
    old8 = run_impl(False, None, 8, PORT + 3, B8)
    new8 = run_impl(True, 1, 8, PORT + 4, B8)
    r.append(check("ws=8 boundary n=7/8/9: old == new bitwise",
                   all(torch.equal(a, b) for a, b in zip(old8, new8))))
    ref8 = single_gpu_reference(8, B8)
    r.append(check("ws=8: single-GPU reference == dist new (bitwise)",
                   torch.equal(ref8, new8[0])))
    r.append(check("ws=8: ranks consistent",
                   all(torch.equal(old8[0], o) for o in old8)
                   and all(torch.equal(new8[0], n) for n in new8)))

    # ── ws=16, boundary n=15/16/17 — wider small path
    B16 = ((15, (8, 8)), (16, (6, 6)), (17, (12, 5)))
    old16 = run_impl(False, None, 16, PORT + 5, B16)
    new16 = run_impl(True, 1, 16, PORT + 6, B16)
    r.append(check("ws=16 boundary n=15/16/17: old == new bitwise",
                   all(torch.equal(a, b) for a, b in zip(old16, new16))))
    ref16 = single_gpu_reference(16, B16)
    r.append(check("ws=16: single-GPU reference == dist new (bitwise)",
                   torch.equal(ref16, new16[0])))

    # ── sanity: params actually train
    init = build_params(BASE)[0][0]["params"][0].detach().clone()
    r.append(check("params actually updated (not a no-op)",
                   not torch.equal(init, new16[0][:init.numel()]
                                   .reshape(init.shape))))

    n_pass = sum(r)
    print(f"\n{n_pass}/{len(r)} checks passed")
    return 0 if n_pass == len(r) else 1


if __name__ == "__main__":
    sys.exit(main())
