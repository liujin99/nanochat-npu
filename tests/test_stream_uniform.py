#!/usr/bin/env python3
"""stream_texts_uniform: the incremental-active rewrite must be
yield-for-yield identical to the verbatim old implementation.

The determinism contract is subtle: the production mix (mix_general_data /
stream_mix) interleaves its own random.random() draws with the generators'
random.choice calls on ONE shared global random state — so equality must
hold call-for-call, under arbitrary interleaving, across endless_generator
re-creation cycles (each cycle re-seeds the shared state mid-trace). The
oracle below is the pre-2026-09-16 implementation copied verbatim.

Also proves the point of the rewrite: the old code rescanned ALL files to
rebuild the active-reader list for EVERY yielded doc (O(docs x files));
prod4's random3b mix drew from 397 STEM files at ~28K docs/s single-core.

Run:  python3 tests/test_stream_uniform.py   (exit 0 = all green)
"""
import itertools
import os
import random
import sys
import tempfile
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))

from nanochat.dataset import stream_texts_uniform as new_stream  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


# ── oracle: verbatim pre-2026-09-16 implementation ────────────────────
def ref_stream_texts_uniform(file_list, shuffle_buffer=10000):
    import random
    random.seed(42)

    readers = []
    for f in file_list:
        try:
            readers.append(pq.ParquetFile(f))
        except:
            continue

    if not readers:
        return

    row_group_indices = [0] * len(readers)
    current_batches = [None] * len(readers)
    current_ptrs = [0] * len(readers)

    while True:
        try:
            active = []
            for i, r in enumerate(readers):
                if row_group_indices[i] < r.num_row_groups or (current_batches[i] is not None and current_ptrs[i] < len(current_batches[i])):
                    active.append(i)

            if not active:
                return

            idx = random.choice(active)
            reader = readers[idx]

            if current_batches[idx] is None or current_ptrs[idx] >= len(current_batches[idx]):
                try:
                    rg = row_group_indices[idx]
                    batch = reader.read_row_group(rg, columns=["text"])
                    current_batches[idx] = batch["text"].to_pylist()
                    current_ptrs[idx] = 0
                    row_group_indices[idx] += 1
                except:
                    row_group_indices[idx] += 1
                    continue

            text = current_batches[idx][current_ptrs[idx]]
            current_ptrs[idx] += 1
            if text and len(text.strip()) > 0:
                yield text
        except GeneratorExit:
            return
        except:
            continue


def _endless(gen_func, files):
    while True:
        gen = gen_func(files)
        yield from gen


def _mk(d, name, docs, rg):
    pq.write_table(pa.table({"text": pa.array(docs, type=pa.string())}),
                   os.path.join(d, name), row_group_size=rg)


# ── A. full drain: varied row groups + blanks + corrupt + empty file ──
with tempfile.TemporaryDirectory() as td:
    files = []
    for i, (n, rg) in enumerate([(50, 17), (3, 1), (120, 40), (7, 3), (0, 1)]):
        docs = []
        for j in range(n):
            if j % 13 == 0:
                docs.append("   ")      # whitespace-only: must be skipped
            elif j % 29 == 0:
                docs.append("")         # empty string: must be skipped
            else:
                docs.append("f%d-d%d" % (i, j))
        _mk(td, "shard_%05d.parquet" % i, docs, rg)
        files.append(os.path.join(td, "shard_%05d.parquet" % i))
    with open(os.path.join(td, "corrupt.parquet"), "wb") as f:
        f.write(b"definitely not parquet")
    files.insert(2, os.path.join(td, "corrupt.parquet"))

    a = list(ref_stream_texts_uniform(files))
    b = list(new_stream(files))
    check("drain: identical full sequence (corrupt/blank/empty included)",
          a == b, f"{len(a)} vs {len(b)} docs")
    check("drain: blanks/empties never yielded",
          all(t and t.strip() for t in b))
    ia = list(itertools.islice(ref_stream_texts_uniform(files), 37))
    ib = list(itertools.islice(new_stream(files), 37))
    check("drain: identical 37-doc prefix (partial consumption)",
          ia == ib and len(ia) == 37)

# ── B. single file ────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as td:
    _mk(td, "one.parquet", ["a%d" % i for i in range(23)], 5)
    one = [os.path.join(td, "one.parquet")]
    check("single file: identical drain",
          list(ref_stream_texts_uniform(one)) == list(new_stream(one)))

# ── C. all-blank file: no yields, terminates ──────────────────────────
with tempfile.TemporaryDirectory() as td:
    _mk(td, "blank.parquet", ["  ", ""] * 10, 4)
    check("all-blank: empty drain, terminates",
          list(new_stream([os.path.join(td, "blank.parquet")])) == [])

# ── D. THE production contract: interleaved random.random() with two ──
# ── generators on one shared global state, across re-creation cycles ──
with tempfile.TemporaryDirectory() as td:
    stem_files, gen_files = [], []
    for i in range(6):
        _mk(td, "stem_%05d.parquet" % i,
            ["s%d-%d" % (i, j) for j in range(40)], 13)
        stem_files.append(os.path.join(td, "stem_%05d.parquet" % i))
    for i in range(2):
        _mk(td, "gen_%05d.parquet" % i,
            ["g%d-%d" % (i, j) for j in range(25)], 10)
        gen_files.append(os.path.join(td, "gen_%05d.parquet" % i))

    def _trace(impl):
        random.seed(42)                  # _mix_data_locked / stream_mix
        ga = _endless(impl, stem_files)  # 240 stem docs -> 2+ cycles
        gb = _endless(impl, gen_files)   # 50 gen docs -> 4+ cycles
        out = []
        for _ in range(700):
            if random.random() < 0.7:
                out.append(("s", next(ga)))
            else:
                out.append(("g", next(gb)))
        return out

    ta = _trace(ref_stream_texts_uniform)
    tb = _trace(new_stream)
    check("production interleave: identical 700-doc trace across cycles",
          ta == tb)
    check("production interleave: both sides drawn",
          any(k == "g" for k, _ in tb) and any(k == "s" for k, _ in tb))

# ── E. perf: the O(files)-per-doc scan is gone ────────────────────────
with tempfile.TemporaryDirectory() as td:
    files = []
    for i in range(200):
        _mk(td, "p%05d.parquet" % i,
            ["doc-%d-%d" % (i, j) for j in range(200)], 100)
        files.append(os.path.join(td, "p%05d.parquet" % i))
    t0 = time.time()
    n_ref = sum(1 for _ in ref_stream_texts_uniform(files))
    t_ref = time.time() - t0
    t0 = time.time()
    n_new = sum(1 for _ in new_stream(files))
    t_new = time.time() - t0
    check("perf: same doc count (40000)", n_ref == n_new == 40000)
    check("perf: incremental active is faster",
          t_new < t_ref,
          "ref %.2fs vs new %.2fs (%.1fx)" % (t_ref, t_new,
                                              t_ref / max(t_new, 1e-9)))

print()
if FAILED:
    print(f"FAILED: {len(FAILED)}: {', '.join(FAILED)}")
    sys.exit(1)
print("All checks passed.")
