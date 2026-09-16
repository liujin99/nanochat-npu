"""DistMuonAdamW small-group path (all_reduce/broadcast) vs stacked path —
gloo CPU equivalence test.

Background (prod4 2026-09-15, job 10c9fc37): the stacked reduce_scatter pads
a Muon group to chunk_size * world_size slots, so a group SMALLER than
world_size costs world_size x shape comm memory regardless of n — d28's
28-layer groups at ws=64 needed 4 GiB bf16 and the first optimizer.step()
OOM'd all 64 ranks. The fix routes small groups (n < world_size, multinode
only, _SMALL_PATH_MIN_WS) through per-param all_reduce + round-robin
ownership + owner-broadcast, plus builds the stacked path's buffer in place
(no torch.stack transient).

Equivalence strategy: dyadic-valued gradients (multiples of 1/4 with
rank offsets) make AVG exact in any reduction order at power-of-two world
sizes, so the OLD implementation (loaded from git HEAD) and the NEW one
must produce BIT-IDENTICAL parameters — the test verifies the plumbing
(scatter/gather vs reduce/broadcast), not collective numerics.

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


def build_params():
    """Same-seed identical init across ranks; returns (muon_groups, adamw_group)."""
    torch.manual_seed(1234)
    def mk(n, shape):
        return [torch.nn.Parameter(
            torch.randn(shape, dtype=torch.float32) * 0.1) for _ in range(n)]
    groups = [
        dict(params=mk(3, (8, 8)), kind="muon", lr=0.02, momentum=0.95,
             ns_steps=5, beta2=0.99, weight_decay=0.01),
        dict(params=mk(9, (12, 5)), kind="muon", lr=0.02, momentum=0.95,
             ns_steps=5, beta2=0.99, weight_decay=0.01),
        dict(params=mk(2, (5, 12)), kind="muon", lr=0.02, momentum=0.95,
             ns_steps=5, beta2=0.99, weight_decay=0.01),
    ]
    adamw = dict(params=[torch.nn.Parameter(torch.randn(64) * 0.1)],
                 kind="adamw", lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.01)
    return groups, adamw


def set_dyadic_grads(groups, adamw, step, rank):
    """Dyadic values (multiples of 1/4): AVG is exact in any reduction order
    at power-of-two world sizes — old vs new must match bitwise."""
    g = torch.Generator().manual_seed(10_000 + step)
    for grp in groups + [adamw]:
        for i, p in enumerate(grp["params"]):
            base = torch.randint(-7, 8, p.shape, generator=g).float() / 4.0
            val = base + rank * 0.25 + step * 0.5
            p.grad = val.clone()


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


def run_rank(rank, world, use_new, gate, port, outdir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    _patch_dist_sync()
    old_mod, new_mod, _ = load_impls()
    mod = new_mod if use_new else old_mod
    groups, adamw = build_params()
    opt = mod.DistMuonAdamW(groups + [adamw])
    if use_new and gate is not None:
        opt._SMALL_PATH_MIN_WS = gate  # instance attr shadows the class default
    for step in range(3):
        set_dyadic_grads(groups, adamw, step, rank)
        opt.step()
    flat = torch.cat([p.detach().reshape(-1)
                      for grp in groups + [adamw] for p in grp["params"]])
    torch.save(flat, os.path.join(outdir, f"rank{rank}.pt"))
    dist.destroy_process_group()


def run_impl(use_new, gate, world, port):
    import tempfile
    outdir = tempfile.mkdtemp(prefix="muon_eq_")
    mp.start_processes(run_rank, args=(world, use_new, gate, port, outdir),
                       nprocs=world, join=True, start_method="spawn")
    return [torch.load(os.path.join(outdir, f"rank{r}.pt"))
            for r in range(world)]


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""),
          flush=True)
    return ok


def main():
    results = []

    # ── 1. gate OFF (default 8, ws=4): stacked path both impls — validates
    #       the in-place stack build (Fix B) is copy-equivalent, bitwise.
    old = run_impl(False, None, 4, PORT)
    new = run_impl(True, None, 4, PORT + 1)
    results.append(check("ws=4 gate=8 (stacked both): old == new bitwise",
                         all(torch.equal(a, b) for a, b in zip(old, new))))
    results.append(check("ws=4: ranks consistent within impl",
                         all(torch.equal(old[0], o) for o in old)
                         and all(torch.equal(new[0], n) for n in new)))

    # ── 2. gate ON (ws=4): groups n=3 and n=2 take the small path in NEW,
    #       stacked in OLD — must still match bitwise (dyadic AVG is exact).
    new_small = run_impl(True, 1, 4, PORT + 2)
    results.append(check("ws=4 gate=1 (small path n<4): old == new bitwise",
                         all(torch.equal(a, b) for a, b in zip(old, new_small))))
    results.append(check("ws=4 small path: ranks consistent",
                         all(torch.equal(new_small[0], n) for n in new_small)))

    # ── 3. ws=8, gate=1: n=3/n=2 small, n=9 stacked(chunk=2, padded=16).
    old8 = run_impl(False, None, 8, PORT + 3)
    new8 = run_impl(True, 1, 8, PORT + 4)
    results.append(check("ws=8 gate=1 (mixed paths): old == new bitwise",
                         all(torch.equal(a, b) for a, b in zip(old8, new8))))
    results.append(check("ws=8: ranks consistent within impl",
                         all(torch.equal(old8[0], o) for o in old8)
                         and all(torch.equal(new8[0], n) for n in new8)))

    # ── 4. sanity: the two impls actually train (params moved from init)
    init = build_params()[0][0]["params"][0].detach().clone()
    results.append(check("params actually updated (not a no-op)",
                         not torch.equal(init, new8[0][:init.numel()]
                                         .reshape(init.shape))))

    n_pass = sum(results)
    print(f"\n{n_pass}/{len(results)} checks passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
