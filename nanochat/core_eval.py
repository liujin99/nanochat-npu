"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794

TODOs:
- All tasks ~match except for squad. We get 31% reference is 37%. Figure out why.
"""
import re
import random

from jinja2 import Template
import torch
import torch.distributed as dist
from nanochat.common import print0

# -----------------------------------------------------------------------------
# Prompt rendering utilities

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple choice question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(choice=choice, **context) for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    Render complete prompt for a language modeling task.
    Notice that we manually trim the context in the template,
    which in some datasets seems to have trailing whitespace (which we don't want).
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    # Return two prompts: without and with the continuation
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    # Due to the way the data seems to be stored, I think I need to strip in the case of LM here.
    # Otherwise we may get trailing whitespaces in prompt_without (which get absorbed into the next
    # token in prompt_with), meaning we don't get a nice and clean prefix in the token space
    # to detect the final continuation. Tokenizers...
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


# -----------------------------------------------------------------------------
# Answer extraction utilities for generation tasks

def extract_gsm8k_answer(text):
    match = re.search(r'####\s*(\-?[0-9\.,]+)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    match = re.search(r'The answer is\s*(\-?[0-9\.,]+)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    numbers = re.findall(r'\-?[0-9]+\.?[0-9]*', text.replace(',', ''))
    if numbers:
        return numbers[-1]
    return None


def extract_math_answer(text):
    idx = text.rfind('\\boxed{')
    if idx >= 0:
        depth = 1
        for i in range(idx + 7, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return text[idx + 7:i].strip()
    match = re.search(r'The\s+answer\s+is\s*(.+?)\.?', text)
    if match:
        return match.group(1).strip()
    return None


def normalize_math_answer(answer):
    answer = answer.strip().replace(',', '').replace('\\ ', '').replace('\\!', '')
    while '{' in answer and '}' in answer:
        answer = answer.replace('{', '').replace('}', '')
    return answer.strip()


def extract_answer(text, extractor_type):
    if extractor_type == 'gsm8k':
        return extract_gsm8k_answer(text)
    elif extractor_type == 'math':
        return extract_math_answer(text)
    raise ValueError(f"Unknown answer extractor: {extractor_type}")


# Stop-string truncation for generation tasks (lm-eval-harness `until`
# semantics). A base model in few-shot completion never emits the chat EOS,
# so after finishing the target answer it keeps generating hallucinated next
# rounds ("Question: ... \nAnswer: ..." / "Problem: ... \n\nSolution: ...").
# Our few-shot exemplars render as "{question}{delimiter}{answer}\n\n" with
# BARE question text (no "Question:"/"Problem:" prefix like lm-eval's
# templates), so the delimiter itself re-appearing is the only reliable
# hallucination boundary. Gold-data verified 2026-09-17: 0 occurrences of
# these strings inside the gsm8k/math answers AND questions (N=1319/500),
# so a well-formed solution can never be cut mid-way.
def apply_stop_strings(text, stop_strings):
    """Cut generated text at the earliest stop-string occurrence."""
    if not stop_strings:
        return text
    cut = len(text)
    for s in stop_strings:
        i = text.find(s)
        if i != -1 and i < cut:
            cut = i
    return text[:cut]


# Per-extractor "the model produced a parseable answer" marker for the
# generation diagnostics (format-following proxy, independent of correctness).
_ANSWER_MARKERS = {'gsm8k': '####', 'math': '\\boxed{'}

# Teacher-forced NLL prompts stay pinned at the HISTORICAL budget
# (max_seq_len - 256: every generation task ran with max_gen_tokens=256
# until 2026-09-17). The generation cap is now a per-task protocol knob
# (math_cot_500: 1024), and the generation prompt budget shrinks with it —
# but letting the NLL prompt shrink too would silently re-define the NLL
# column against every historical search point (math NLL is that benchmark's
# main live signal). Registry override: 'nll_prompt_budget'.
_NLL_FROZEN_GEN_CAP = 256


def compare_answers(pred_answer, gold_answer, extractor_type):
    if pred_answer is None or gold_answer is None:
        return False
    if extractor_type == 'gsm8k':
        try:
            return abs(float(pred_answer) - float(gold_answer)) < 1e-6
        except (ValueError, TypeError):
            return pred_answer == gold_answer
    elif extractor_type == 'math':
        pred_norm = normalize_math_answer(pred_answer)
        gold_norm = normalize_math_answer(gold_answer)
        if pred_norm == gold_norm:
            return True
        try:
            return abs(float(pred_norm) - float(gold_norm)) < 1e-6
        except (ValueError, TypeError):
            return False
    return pred_answer == gold_answer


def render_prompt_generation(item, continuation_delimiter, fewshot_examples=None):
    """Render few-shot ICL prompt for generation task."""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.question }}{{ continuation_delimiter }}{{ example.answer }}

{% endfor -%}
{{ item.question }}{{ continuation_delimiter }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    return template.render(
        fewshot_examples=fewshot_examples,
        continuation_delimiter=continuation_delimiter,
        item=item
    )


def find_common_length(token_sequences, direction='left'):
    """
    Find the length of the common prefix or suffix across token sequences
    - direction: 'left' for prefix, 'right' for suffix
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    # Find the first position where the token sequences differ
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """Stack up a list of token sequences, pad to longest on the right"""
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


# Row budget for forward_model's batch-axis slicing. 4096 = the training
# micro-batch's per-forward row count (2 x 2048), proven over 2 x 2861
# steps at ws=128 tonight; see forward_model's docstring for the OOM it
# prevents.
_EVAL_MAX_ROWS = 4096


def batch_sequences_mc(tokenizer, prompts):
    # In multiple choice, contexts are the same but the continuation is different (common prefix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each continuation
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    # In schema tasks, contexts vary but continuation is the same (common suffix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each context
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    # In LM tasks, we have two prompts: without and with continuation
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    # we only need the with continuation prompt in the LM task, i.e. batch size of 1
    return [tokens_with], [start_idx], [end_idx]


@torch.no_grad()
def forward_model(model, input_ids, max_rows=None):
    """
    Take BxT tensor of token ids, return BxT tensor of losses and argmax predictions.
    The last column of losses is set to nan because we don't have autoregressive targets there.

    Row-bounded forward: one (B, T) chunk with B*T rows makes the lm_head
    emit a (B*T, vocab) logits tensor — at the target arms' eval batch
    (8 examples x 4 MC choices, T up to 2048) that is 65536 x 32768 =
    4.3 GB bf16 plus the fp32 CE workspace, a 16x larger footprint than
    the training micro-batch (2 x 2048). At ws=128 (prod4 2026-09-16,
    both arms, job 2e58bdab + d9b8dae3) the first such chunk failed
    inside aclnnMatmul (ACL ERR00100, deterministically on rank 76 —
    the only chunk that combined a full 8 examples with a 2048-truncated
    one). Slice the batch axis so every forward stays within
    _EVAL_MAX_ROWS (training-proven row count); CE and argmax are
    row-independent, so slicing is exactly equivalent.
    """
    batch_size, seq_len = input_ids.size()
    if max_rows is None:
        max_rows = _EVAL_MAX_ROWS
    step = max(1, max_rows // seq_len)
    losses_parts, pred_parts = [], []
    for i in range(0, batch_size, step):
        ids = input_ids[i:i + step]
        b = ids.size(0)
        outputs = model(ids)
        # Roll the tensor to the left by one position to get the (autoregressive) target ids
        target_ids = torch.roll(ids, shifts=-1, dims=1)
        # Calculate cross entropy at all positions
        losses = torch.nn.functional.cross_entropy(
            outputs.view(b * seq_len, -1),
            target_ids.view(b * seq_len),
            reduction='none'
        ).view(b, seq_len)
        # Set the last column to be nan because there is no autoregressive loss there
        losses[:, -1] = float('nan')
        # Get the argmax predictions at each position
        predictions = outputs.argmax(dim=-1)
        losses_parts.append(losses)
        pred_parts.append(predictions)
    if len(losses_parts) == 1:
        return losses_parts[0], pred_parts[0]
    return torch.cat(losses_parts, dim=0), torch.cat(pred_parts, dim=0)


def prepare_example(idx, data, tokenizer, task_meta, max_seq_len=None):
    """
    Render + tokenize + truncate a single example.
    Returns (tokens, start_idxs, end_idxs, task_type, gold).
    """
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    if max_seq_len is not None:
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_seq_len:
                num_to_crop = len(t) - max_seq_len
                new_tokens.append(t[-max_seq_len:])
                new_start_idxs.append(s - num_to_crop)
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0, "this should never happen right?"
                assert e - num_to_crop >= 0, "this should never happen right?"
            else:
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    return tokens, start_idxs, end_idxs, task_type, item.get('gold')


def judge_example(losses, predictions, input_ids, local_start_idxs, local_end_idxs, task_type, gold):
    """Judge correctness from pre-computed losses/predictions for one example's slice.
    Returns (is_correct, nll_gold) where nll_gold is the teacher-forced NLL on the gold answer.
    """
    if task_type == 'language_modeling':
        si = local_start_idxs[0]
        ei = local_end_idxs[0]
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
        nll_gold = losses[0, si-1:ei-1].mean().item()
        return is_correct, nll_gold
    elif task_type in ['multiple_choice', 'schema']:
        mean_losses = [losses[i, si-1:ei-1].mean().item()
                        for i, (si, ei) in enumerate(zip(local_start_idxs, local_end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))
        return pred_idx == gold, mean_losses[gold]
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """Evaluate a single example, return (is_correct, nll_gold)."""
    max_seq_len = getattr(model, 'max_seq_len', None)
    tokens, start_idxs, end_idxs, task_type, gold = prepare_example(
        idx, data, tokenizer, task_meta, max_seq_len)

    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    losses, predictions = forward_model(model, input_ids)
    return judge_example(losses, predictions, input_ids, start_idxs, end_idxs, task_type, gold)


def evaluate_task(model, tokenizer, data, device, task_meta, eval_batch_size=1):
    """
    Evaluate one task across many examples with DDP striding.
    When eval_batch_size > 1, multiple examples are batched into a single forward pass.
    Returns (mean_correct, mean_nll) where mean_nll is the average teacher-forced NLL on gold answers.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    nlls = torch.zeros(len(data), dtype=torch.float32, device=device)

    max_seq_len = getattr(model, 'max_seq_len', None)
    pad_token_id = tokenizer.get_bos_token_id()

    my_indices = list(range(rank, len(data), world_size))

    if eval_batch_size <= 1:
        for idx in my_indices:
            is_correct, nll_gold = evaluate_example(idx, model, tokenizer, data, device, task_meta)
            correct[idx] = float(is_correct)
            nlls[idx] = nll_gold
    else:
        current_batch_size = eval_batch_size
        chunk_start = 0
        while chunk_start < len(my_indices):
            chunk = my_indices[chunk_start:chunk_start + current_batch_size]

            all_tokens = []
            all_meta = []
            boundaries = [0]

            for idx in chunk:
                tokens, start_idxs, end_idxs, task_type, gold = prepare_example(
                    idx, data, tokenizer, task_meta, max_seq_len)
                all_tokens.extend(tokens)
                all_meta.append((len(tokens), start_idxs, end_idxs, task_type, gold))
                boundaries.append(boundaries[-1] + len(tokens))

            input_ids = stack_sequences(all_tokens, pad_token_id)
            input_ids = input_ids.to(device)

            try:
                losses, predictions = forward_model(model, input_ids)
            except RuntimeError as e:
                # NPU operator failures surface as "The Inner error is
                # reported as above ... operator name is aclnnXxx"
                # (ERR00100) — allocation failures that, like CUDA OOM,
                # respond to halving the batch. prod4 2026-09-16: the
                # original catch missed this signature and the eval died.
                err = str(e).lower()
                if 'out of memory' in err or 'aclnn' in err or 'inner error' in err:
                    if device.type == "npu":
                        torch.npu.empty_cache()
                    elif device.type == "cuda":
                        torch.cuda.empty_cache()
                    if current_batch_size > 1:
                        current_batch_size = max(1, current_batch_size // 2)
                        continue
                    for idx in chunk:
                        is_correct, nll_gold = evaluate_example(idx, model, tokenizer, data, device, task_meta)
                        correct[idx] = float(is_correct)
                        nlls[idx] = nll_gold
                    chunk_start += len(chunk)
                    continue
                raise

            for j, idx in enumerate(chunk):
                b, e = boundaries[j], boundaries[j + 1]
                n_seqs, _, _, task_type, gold = all_meta[j]
                local_losses = losses[b:e]
                local_preds = predictions[b:e]
                local_input = input_ids[b:e]
                is_correct, nll_gold = judge_example(
                    local_losses, local_preds, local_input,
                    all_meta[j][1], all_meta[j][2], task_type, gold)
                correct[idx] = float(is_correct)
                nlls[idx] = nll_gold
            chunk_start += len(chunk)

    if world_size > 1:
        # Hot chunk loop -> pool reserved up to its ceiling -> the first
        # collective needs driver-side memory HCCL cannot get (EL0004
        # Memory_Allocation_Failure, the 401MiB comm-buffer class:
        # 2026-08-28 speedrun Step 7 and prod4 2026-09-17 anchor
        # b9e3def2 both died at exactly this barrier). Flush first —
        # the generation task already flushes before its collectives.
        if device.type == "npu":
            torch.npu.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(nlls, op=dist.ReduceOp.SUM)
    mean_correct = correct.mean().item()
    mean_nll = nlls.mean().item()
    return mean_correct, mean_nll


@torch.no_grad()
def evaluate_generation_task(model, tokenizer, data, device, task_meta, gen_batch_size=8):
    """Evaluate generation task: accuracy via autoregressive generation + teacher-forced NLL.
    Returns (mean_correct, mean_nll).
    """
    import gc
    from nanochat.engine import Engine

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']
    answer_extractor = task_meta.get('answer_extractor', 'gsm8k')
    max_gen_tokens = task_meta.get('max_gen_tokens', 512)
    stop_strings = task_meta.get('stop_strings', [])
    marker = _ANSWER_MARKERS.get(answer_extractor)
    label = task_meta.get('label', 'unknown')
    max_seq_len = getattr(model, 'max_seq_len', None)
    # The generation prompt must reserve the full generation window inside
    # the model context; the NLL prompt budget is decoupled and frozen (see
    # _NLL_FROZEN_GEN_CAP).
    max_prompt_len = (max_seq_len - max_gen_tokens) if max_seq_len else None
    # Registry 'nll_prompt_budget' may exist with value None (base_eval
    # passes task.get() unconditionally) — fall to the frozen default.
    nll_prompt_budget = task_meta.get('nll_prompt_budget')
    if nll_prompt_budget is None:
        nll_prompt_budget = (max_seq_len - _NLL_FROZEN_GEN_CAP) if max_seq_len else None

    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    nlls = torch.zeros(len(data), dtype=torch.float32, device=device)
    my_indices = list(range(rank, len(data), world_size))

    engine = Engine(model, tokenizer)
    bos_token_id = tokenizer.get_bos_token_id()

    oom_flag = torch.tensor([0.0], device=device)

    # Generation diagnostics, all-reduced after the loop:
    # [hit_cap, stopped, early_stop_no_marker, has_marker, n, shots_sum, gen_len_sum]
    diag = torch.zeros(7, dtype=torch.float32, device=device)

    if device.type == "npu":
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    def build_prompt(item, idx, budget):
        # Deterministic per-idx few-shot selection (seed 1234+idx), popping
        # the oldest exemplar until the prompt fits the budget. Operation
        # order is identical to the pre-decoupling code, so equal budgets
        # reproduce byte-identical prompts.
        fewshot_examples = []
        if num_fewshot > 0:
            rng = random.Random(1234 + idx)
            available_indices = [i for i in range(len(data)) if i != idx]
            fewshot_indices = rng.sample(available_indices, min(num_fewshot, len(available_indices)))
            fewshot_examples = [data[i] for i in fewshot_indices]
            if budget is not None:
                while fewshot_examples:
                    prompt = render_prompt_generation(item, continuation_delimiter, fewshot_examples)
                    prompt_tokens = tokenizer(prompt, prepend=bos_token_id)
                    if len(prompt_tokens) <= budget:
                        break
                    fewshot_examples.pop(0)
        prompt = render_prompt_generation(item, continuation_delimiter, fewshot_examples)
        prompt_tokens = tokenizer(prompt, prepend=bos_token_id)
        if budget and len(prompt_tokens) > budget:
            prompt_tokens = prompt_tokens[:budget]
        return prompt, prompt_tokens, len(fewshot_examples)

    def account(p, raw_text, gen_len):
        # Stop-string truncation + diagnostic accounting for one generation.
        text = apply_stop_strings(raw_text, stop_strings)
        stopped = len(text) < len(raw_text)
        has_marker = marker is not None and marker in text
        diag[0] += float(gen_len >= max_gen_tokens)
        diag[1] += float(stopped)
        diag[2] += float(stopped and not has_marker)
        diag[3] += float(has_marker)
        diag[4] += 1.0
        diag[5] += float(p['shots'])
        diag[6] += float(gen_len)
        return text

    prepared = []
    for idx in my_indices:
        item = data[idx]
        gold_answer = extract_answer(item['answer'], answer_extractor)
        if 'gold_answer' in item and item['gold_answer']:
            gold_answer = item['gold_answer']

        gen_prompt, gen_tokens, n_shots = build_prompt(item, idx, max_prompt_len)
        if nll_prompt_budget == max_prompt_len:
            nll_prompt, nll_tokens = gen_prompt, gen_tokens
        else:
            nll_prompt, nll_tokens, _ = build_prompt(item, idx, nll_prompt_budget)
        prepared.append({
            'idx': idx, 'tokens': gen_tokens, 'gold': gold_answer, 'prompt': gen_prompt,
            'nll_prompt': nll_prompt, 'nll_tokens': nll_tokens, 'shots': n_shots,
        })

    local_correct = 0.0
    local_done = 0
    current_batch_size = gen_batch_size
    batch_start = 0
    while batch_start < len(prepared):
        batch_end = min(batch_start + current_batch_size, len(prepared))
        batch = prepared[batch_start:batch_end]
        batch_prompts = [p['tokens'] for p in batch]
        batch_indices = [p['idx'] for p in batch]
        batch_golds = [p['gold'] for p in batch]

        try:
            results, _ = engine.generate_batch_prompts(
                batch_prompts, max_tokens=max_gen_tokens, temperature=0,
                stop_strings=stop_strings,
            )
        except RuntimeError as e:
            err_str = str(e).lower()
            is_oom = 'out of memory' in err_str or ('npu' in err_str and ('memory' in err_str or 'alloc' in err_str or '507048' in str(e)))
            if device.type == "npu":
                torch.npu.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            oom_flag[0] = 1.0
            print0(f"  [{label}] RuntimeError at batch {batch_start}: {str(e)[:200]}")
            if is_oom and current_batch_size > 1:
                current_batch_size = max(1, current_batch_size // 2)
                continue
            for p in batch:
                local_done += 1
                try:
                    generated, _ = engine.generate_batch(
                        p['tokens'], num_samples=1, max_tokens=max_gen_tokens, temperature=0
                    )
                    raw_text = tokenizer.decode(generated[0][len(p['tokens']):])
                    generated_text = account(p, raw_text, len(generated[0]) - len(p['tokens']))
                    pred_answer = extract_answer(generated_text, answer_extractor)
                    is_correct = compare_answers(pred_answer, p['gold'], answer_extractor)
                    correct[p['idx']] = float(is_correct)
                    local_correct += float(is_correct)
                except RuntimeError:
                    diag[4] += 1.0
                    diag[5] += float(p['shots'])
                    correct[p['idx']] = 0.0
                    print0(f"  [{label}] Error at example {p['idx']}, skipping")
            batch_start += len(batch)
            continue

        for i, (idx, gold) in enumerate(zip(batch_indices, batch_golds)):
            raw_text = tokenizer.decode(results[i][len(batch_prompts[i]):])
            gen_len = len(results[i]) - len(batch_prompts[i])
            generated_text = account(batch[i], raw_text, gen_len)
            pred_answer = extract_answer(generated_text, answer_extractor)
            is_correct = compare_answers(pred_answer, gold, answer_extractor)
            correct[idx] = float(is_correct)
            local_correct += float(is_correct)
            local_done += 1

        del results
        gc.collect()
        if device.type == "npu":
            torch.npu.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        if batch_start > 0 and (batch_start // gen_batch_size) % 10 == 0:
            count = batch_start
            partial_acc = local_correct / local_done if local_done > 0 else 0.0
            print0(f"  [{label}] {count}/{len(my_indices)} examples, partial acc: {partial_acc:.4f}")

        batch_start = batch_end

    # Generation diagnostics (all ranks contribute; print on master).
    if world_size > 1:
        dist.all_reduce(diag, op=dist.ReduceOp.SUM)
    n_total = max(int(diag[4].item()), 1)
    # With engine early-stop active, hit_cap/avg_gen_len reflect engine-stopped
    # lengths (a row that hit its stop string no longer "hits the cap") —
    # scores are unaffected; append the note so longitudinal diag comparisons
    # don't misread the semantic shift.
    diag_note = (" (note: engine early-stop active, hit_cap/avg_gen_len are "
                 "engine-stopped lengths)" if stop_strings else "")
    print0(f"  [{label}] GEN-DIAG: N={int(diag[4].item())} "
           f"marker_rate={diag[3].item()/n_total:.4f} hit_cap={diag[0].item()/n_total:.4f} "
           f"stopped={diag[1].item()/n_total:.4f} early_no_marker={diag[2].item()/n_total:.4f} "
           f"avg_shots={diag[5].item()/n_total:.2f} avg_gen_len={diag[6].item()/n_total:.1f}"
           f"{diag_note}")

    # Compute teacher-forced NLL on gold answers
    print0(f"  [{label}] Computing teacher-forced NLL on gold answers...")
    for p in prepared:
        idx = p['idx']
        item = data[idx]
        answer_text = str(item['answer'])
        full_tokens = tokenizer(p['nll_prompt'] + answer_text, prepend=bos_token_id)
        start_idx = len(p['nll_tokens'])
        end_idx = len(full_tokens)
        if end_idx <= start_idx:
            nlls[idx] = 0.0
            continue
        if max_seq_len and len(full_tokens) > max_seq_len:
            num_to_crop = len(full_tokens) - max_seq_len
            full_tokens = full_tokens[-max_seq_len:]
            start_idx = max(0, start_idx - num_to_crop)
            end_idx = len(full_tokens)
        if end_idx <= start_idx:
            nlls[idx] = 0.0
            continue
        input_ids = stack_sequences([full_tokens], bos_token_id).to(device)
        try:
            losses, _ = forward_model(model, input_ids)
            nlls[idx] = losses[0, start_idx-1:end_idx-1].mean().item()
        except RuntimeError:
            nlls[idx] = 0.0
            print0(f"  [{label}] NLL computation failed at example {idx}, skipping")

    if device.type == "npu":
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    if world_size > 1:
        try:
            dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)
            any_oom = oom_flag[0].item() > 0.5
            if any_oom:
                print0(f"  [{label}] OOM detected, returning partial result")
                return correct.mean().item(), nlls.mean().item()
            dist.barrier()
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(nlls, op=dist.ReduceOp.SUM)
        except RuntimeError as e:
            print0(f"  [{label}] Collective communication failed: {str(e)[:200]}, returning local result")
            return correct.mean().item(), nlls.mean().item()
    return correct.mean().item(), nlls.mean().item()
