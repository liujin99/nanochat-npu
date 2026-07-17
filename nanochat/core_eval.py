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
def forward_model(model, input_ids):
    """
    Take BxT tensor of token ids, return BxT tensor of losses and argmax predictions.
    The last column of losses is set to nan because we don't have autoregressive targets there.
    """
    batch_size, seq_len = input_ids.size()
    outputs = model(input_ids)
    # Roll the tensor to the left by one position to get the (autoregressive) target ids
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    # Calculate cross entropy at all positions
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none'
    ).view(batch_size, seq_len)
    # Set the last column to be nan because there is no autoregressive loss there
    losses[:, -1] = float('nan')
    # Get the argmax predictions at each position
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


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
    """Judge correctness from pre-computed losses/predictions for one example's slice."""
    if task_type == 'language_modeling':
        si = local_start_idxs[0]
        ei = local_end_idxs[0]
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        return torch.all(predicted_tokens == actual_tokens).item()
    elif task_type in ['multiple_choice', 'schema']:
        mean_losses = [losses[i, si-1:ei-1].mean().item()
                        for i, (si, ei) in enumerate(zip(local_start_idxs, local_end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))
        return pred_idx == gold
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """Evaluate a single example, return True if correct, False otherwise"""
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
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)

    max_seq_len = getattr(model, 'max_seq_len', None)
    pad_token_id = tokenizer.get_bos_token_id()

    my_indices = list(range(rank, len(data), world_size))

    if eval_batch_size <= 1:
        for idx in my_indices:
            is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
            correct[idx] = float(is_correct)
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
                if 'out of memory' in str(e).lower():
                    if device.type == "npu":
                        torch.npu.empty_cache()
                    elif device.type == "cuda":
                        torch.cuda.empty_cache()
                    if current_batch_size > 1:
                        current_batch_size = max(1, current_batch_size // 2)
                        continue
                    for idx in chunk:
                        is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
                        correct[idx] = float(is_correct)
                    chunk_start += len(chunk)
                    continue
                raise

            for j, idx in enumerate(chunk):
                b, e = boundaries[j], boundaries[j + 1]
                n_seqs, _, _, task_type, gold = all_meta[j]
                local_losses = losses[b:e]
                local_preds = predictions[b:e]
                local_input = input_ids[b:e]
                is_correct = judge_example(
                    local_losses, local_preds, local_input,
                    all_meta[j][1], all_meta[j][2], task_type, gold)
                correct[idx] = float(is_correct)
            chunk_start += len(chunk)

    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    mean_correct = correct.mean().item()
    return mean_correct


@torch.no_grad()
def evaluate_generation_task(model, tokenizer, data, device, task_meta):
    """
    Evaluate a generation task (GSM8K, MATH, etc.) using autoregressive generation.
    Renders few-shot ICL prompt, generates completion, extracts answer, compares with gold.
    Truncates prompt (by dropping few-shot examples) if it exceeds max_seq_len - max_gen_tokens.
    """
    from nanochat.engine import Engine

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']
    answer_extractor = task_meta.get('answer_extractor', 'gsm8k')
    max_gen_tokens = task_meta.get('max_gen_tokens', 512)
    label = task_meta.get('label', 'unknown')
    max_seq_len = getattr(model, 'max_seq_len', None)
    max_prompt_len = (max_seq_len - max_gen_tokens) if max_seq_len else None

    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    my_indices = list(range(rank, len(data), world_size))

    engine = Engine(model, tokenizer)
    bos_token_id = tokenizer.get_bos_token_id()

    for count, idx in enumerate(my_indices):
        item = data[idx]

        gold_answer = extract_answer(item['answer'], answer_extractor)
        if 'gold_answer' in item and item['gold_answer']:
            gold_answer = item['gold_answer']

        effective_fewshot = num_fewshot
        fewshot_examples = []
        if effective_fewshot > 0:
            rng = random.Random(1234 + idx)
            available_indices = [i for i in range(len(data)) if i != idx]
            fewshot_indices = rng.sample(available_indices, min(effective_fewshot, len(available_indices)))
            fewshot_examples = [data[i] for i in fewshot_indices]

            if max_prompt_len is not None:
                while fewshot_examples:
                    prompt = render_prompt_generation(item, continuation_delimiter, fewshot_examples)
                    prompt_tokens = tokenizer(prompt, prepend=bos_token_id)
                    if len(prompt_tokens) <= max_prompt_len:
                        break
                    fewshot_examples.pop(0)

        prompt = render_prompt_generation(item, continuation_delimiter, fewshot_examples)
        prompt_tokens = tokenizer(prompt, prepend=bos_token_id)

        if max_prompt_len and len(prompt_tokens) > max_prompt_len:
            prompt_tokens = prompt_tokens[:max_prompt_len]

        try:
            generated, _ = engine.generate_batch(
                prompt_tokens, num_samples=1, max_tokens=max_gen_tokens, temperature=0
            )
            generated_text = tokenizer.decode(generated[0][len(prompt_tokens):])
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                if device.type == "npu":
                    torch.npu.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
                correct[idx] = 0.0
                continue
            raise

        pred_answer = extract_answer(generated_text, answer_extractor)
        is_correct = compare_answers(pred_answer, gold_answer, answer_extractor)
        correct[idx] = float(is_correct)

        if count % 10 == 0 and count > 0:
            partial_acc = correct[:idx+1].mean().item()
            print0(f"  [{label}] {count}/{len(my_indices)} examples, partial acc: {partial_acc:.4f}")

    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    return correct.mean().item()
