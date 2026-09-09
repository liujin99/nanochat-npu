"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

import bisect
import torch
import numpy as np
import pyarrow.parquet as pq
import logging

from nanochat.common import get_dist_info, print0
from nanochat.dataset import list_parquet_files

logger = logging.getLogger(__name__)

def _document_batches(split, resume_state_dict, tokenizer_batch_size, data_dir=None):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).

    DDP sharding flattens every (file, row_group) pair into ONE global sequence
    (files in list order, row groups ascending within each file) and deals them
    round-robin: global index k goes to rank k % world_size. When every file's
    row-group count is divisible by world_size this is IDENTICAL to the old
    per-file scheme (rg_idx = rank, stride world_size) — same ranks, same row
    groups, same order, same resume states — so existing single-node runs
    (ws=8 vs 16-row-group shards) are unaffected. Unlike the per-file scheme it
    cannot STARVE top ranks when a file has fewer row groups than world_size
    (multi-node 2026-09-09: ws=24/32 vs 16-row-group mixture shards spun
    forever reopening files without ever yielding).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy
    parquet_paths = list_parquet_files(data_dir=data_dir, warn_on_legacy=warn_on_legacy)
    assert len(parquet_paths) != 0, f"No dataset parquet files found in {data_dir}, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    assert len(parquet_paths) != 0, f"split '{split}' got 0 parquet files from {data_dir}, did you run dataset.py?"

    # Pre-scan row-group counts (footer reads only, no data). Corrupt shards
    # are logged + dropped once here so EVERY rank computes the same global
    # layout (the old lazy scheme made a corrupt shard visible only to ranks
    # that had row groups in it). Empty files contribute 0 and vanish from
    # the layout naturally.
    layout = []  # (original pq_idx, path, num_row_groups)
    for pq_idx, filepath in enumerate(parquet_paths):
        try:
            n = pq.ParquetFile(filepath).metadata.num_row_groups
        except Exception as e:
            logger.warning(f"Skipping corrupted shard {filepath}: {e}")
            continue
        layout.append((pq_idx, filepath, n))
    total_row_groups = sum(n for _, _, n in layout)
    if total_row_groups < ddp_world_size:
        raise RuntimeError(
            f"row-group starvation: world_size={ddp_world_size} but the "
            f"'{split}' split has only {total_row_groups} row groups across "
            f"{len(layout)} readable shards — ranks {total_row_groups}.."
            f"{ddp_world_size - 1} would read no data and the loader would "
            f"spin forever; add data or reduce world size")

    # k_base[f] = global index of layout slot f's first row group
    k_base = [0]
    for _, _, n in layout:
        k_base.append(k_base[-1] + n)

    def slot_for_k(k):
        # layout slot containing global row-group index k
        return bisect.bisect_right(k_base, k) - 1

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        if first_pass:
            first_pass = False
            # First pass starts at the resume position (if any); later
            # passes always restart from the beginning of the sequence.
            k_start = 0
            if resume_rg_idx is not None:
                # map the saved (pq_idx, rg_idx) back to its global index
                # and advance past it to this rank's NEXT row group
                k_resume = None
                for slot, (orig_idx, _, n) in enumerate(layout):
                    if orig_idx == resume_pq_idx:
                        k_resume = k_base[slot] + resume_rg_idx
                        break
                if k_resume is not None:
                    # smallest k > k_resume with k % world_size == rank
                    k_start = k_resume + 1 + (ddp_rank - (k_resume + 1)) % ddp_world_size
                else:
                    # the resume shard is unreadable NOW (was readable when
                    # the state was saved): skip to the next readable file,
                    # matching the old scheme's corrupt-shard skip
                    for slot, (orig_idx, _, _) in enumerate(layout):
                        if orig_idx > resume_pq_idx:
                            k_start = k_base[slot]
                            break
                    else:
                        k_start = total_row_groups  # nothing left this pass
            elif resume_state_dict is not None:
                # state without rg_idx: start of the saved file (old scheme
                # resumed at rg_idx = ddp_rank of that file)
                for slot, (orig_idx, _, _) in enumerate(layout):
                    if orig_idx == resume_pq_idx:
                        k_start = k_base[slot]
                        break
        else:
            k_start = 0

        k = k_start
        cur_slot, cur_pf = -1, None
        while k < total_row_groups:
            if k % ddp_world_size != ddp_rank:
                k += 1
                continue
            slot = slot_for_k(k)
            orig_idx, filepath, _ = layout[slot]
            if slot != cur_slot:
                try:
                    cur_pf = pq.ParquetFile(filepath)
                    cur_slot = slot
                except Exception as e:
                    # readable at pre-scan but not now; every rank that
                    # reaches this file sees the same and skips its share
                    logger.warning(f"Skipping corrupted shard {filepath}: {e}")
                    k = k_base[slot + 1]
                    cur_slot, cur_pf = -1, None
                    continue
            rg_idx = k - k_base[slot]
            try:
                rg = cur_pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
            except Exception as e:
                logger.warning(f"Skipping corrupted row_group {rg_idx} in {filepath}: {e}")
                k += 1
                continue
            for i in range(0, len(batch), tokenizer_batch_size):
                yield batch[i:i+tokenizer_batch_size], (orig_idx, rg_idx, epoch)
            k += 1
        epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000, data_dir=None
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size, data_dir=data_dir)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Copy to pinned CPU buffer, then single HtoD transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        # Single HtoD copy into persistent GPU buffer and yield
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets


def tokenizing_distributed_data_loader_with_state_flat(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000, data_dir=None
):
    """
    Flat concatenation dataloader with zero token waste.

    Documents are tokenized with BOS prepended, then concatenated into a continuous
    token stream. Batches are formed by slicing the stream into rows of length T+1.
    No tokens are discarded — long documents simply span multiple rows.

    Key properties:
    - Zero token waste (no cropping, no padding)
    - Complete document content preserved (no lost answers/conclusions)
    - BOS token marks document boundaries (soft boundary, no attention masking)
    - Same output format as BOS-bestfit: (inputs [B,T], targets [B,T], state_dict)

    Inspired by DeepSeek V3's document packing (Section 4.1): packing for data
    integrity without cross-sample attention masking.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    needed_tokens = B * row_capacity
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size, data_dir=data_dir)
    bos_token = tokenizer.get_bos_token_id()
    token_lists = []
    current_idx = 0
    current_offset = 0
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        tokenized = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in tokenized:
            token_lists.append(np.array(tokens, dtype=np.int64))

    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    flat_buffer = np.empty(needed_tokens, dtype=np.int64)
    _first_batch = True

    while True:
        pos = 0
        while pos < needed_tokens:
            while len(token_lists) - current_idx < buffer_size:
                refill_buffer()

            arr = token_lists[current_idx]
            available = len(arr) - current_offset
            to_copy = min(available, needed_tokens - pos)
            flat_buffer[pos:pos + to_copy] = arr[current_offset:current_offset + to_copy]
            pos += to_copy
            current_offset += to_copy
            if current_offset >= len(arr):
                current_idx += 1
                current_offset = 0

        if current_idx > 1000:
            del token_lists[:current_idx]
            current_idx = 0

        if _first_batch:
            print0(f"[flat_loader] B={B}, T={T}, needed_tokens={needed_tokens}, "
                   f"flat_buffer[:12]={flat_buffer[:min(12, needed_tokens)].tolist()}, "
                   f"current_idx={current_idx}, current_offset={current_offset}, "
                   f"buffer_docs={len(token_lists)}")
            _first_batch = False

        row_buffer.copy_(torch.from_numpy(flat_buffer).view(B, row_capacity))
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict


def tokenizing_distributed_data_loader_flat(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_flat(*args, **kwargs):
        yield inputs, targets