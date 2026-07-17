"""
Unified evaluation script for base models.

Supports three evaluation modes (comma-separated):
  --eval core    : CORE metric (accuracy on ICL tasks)
  --eval bpb     : Bits per byte on train/val splits
  --eval sample  : Generate samples from the model

Default is all three: --eval core,bpb,sample

Examples:

    # Evaluate a HuggingFace model (e.g. GPT-2 124M) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.base_eval --hf-path openai-community/gpt2

    # Evaluate a nanochat model (e.g. d24) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.base_eval --model-tag d24 --device-batch-size=16

    # Quick/approximate evaluation using a single GPU
    python -m scripts.base_eval --model-tag d24 --device-batch-size=16 --max-per-task=100 --split-tokens=524288
"""
import os
import csv
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile
import argparse
import torch
try:
    import torch_npu
except ImportError:
    pass  # torch_npu not available
import ssl
import urllib
ssl._create_default_https_context = ssl._create_unverified_context

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock
from nanochat.tokenizer import HuggingFaceTokenizer, get_token_bytes
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import evaluate_task, evaluate_generation_task
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine

# -----------------------------------------------------------------------------
# HuggingFace loading utilities

class ModelWrapper:
    """Lightweight wrapper to give HuggingFace models a nanochat-compatible interface."""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        logits = self.model(input_ids).logits
        if targets is None:
            return logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction
        )
        return loss

    def get_device(self):
        return next(self.model.parameters()).device


def load_hf_model(hf_path: str, device):
    """Load a HuggingFace model and tokenizer."""
    print0(f"Loading HuggingFace model from: {hf_path}")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer


def get_hf_token_bytes(tokenizer, device="cpu"):
    """Compute token_bytes tensor for a HuggingFace tokenizer."""
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    for token_id in range(vocab_size):
        token_str = tokenizer.tokenizer.decode([token_id])
        token_bytes[token_id] = len(token_str.encode('utf-8'))
    return token_bytes

# -----------------------------------------------------------------------------
# CORE evaluation

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
EVAL_STEM_URL = "https://hf-mirror.com/datasets/liujin99/nanochat-npu-stem-eval/resolve/main/eval_stem.zip"

STEM_SUBJECT_KEYWORDS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'college_biology', 'college_chemistry',
    'college_computer_science', 'college_mathematics', 'college_physics', 'computer_security',
    'conceptual_physics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
    'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
    'high_school_mathematics', 'high_school_physics', 'high_school_statistics',
    'machine_learning', 'medical_genetics', 'virology',
]


STEM_TASKS = [
    {
        'label': 'gpqa_diamond',
        'dataset_uri': 'gpqa_diamond.jsonl',
        'num_fewshot': [0],
        'icl_task_type': 'multiple_choice',
        'continuation_delimiter': '\nAnswer: ',
    },
    {
        'label': 'gsm8k_cot',
        'dataset_uri': 'gsm8k.jsonl',
        'num_fewshot': [8],
        'icl_task_type': 'generation',
        'continuation_delimiter': '\nAnswer: ',
        'answer_extractor': 'gsm8k',
        'max_gen_tokens': 512,
    },
    {
        'label': 'math_cot',
        'dataset_uri': 'math500.jsonl',
        'num_fewshot': [4],
        'icl_task_type': 'generation',
        'continuation_delimiter': '\n\nSolution: ',
        'answer_extractor': 'math',
        'max_gen_tokens': 512,
    },
    {
        'label': 'mmlu_zeroshot',
        'dataset_uri': 'mmlu.jsonl',
        'num_fewshot': [0],
        'icl_task_type': 'multiple_choice',
        'continuation_delimiter': '\nAnswer: ',
    },
    {
        'label': 'mmlu_stem',
        'dataset_uri': 'mmlu_stem.jsonl',
        'num_fewshot': [0],
        'icl_task_type': 'multiple_choice',
        'continuation_delimiter': '\nAnswer: ',
    },
]

STEM_BENCHMARK_LABELS = ['arc_easy', 'arc_challenge', 'mmlu_stem', 'gpqa_diamond', 'gsm8k_cot', 'math_cot']


def place_eval_stem(file_path):
    """Unzip eval_stem.zip and place it in the base directory."""
    base_dir = get_base_dir()
    eval_stem_dir = os.path.join(base_dir, "eval_stem")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_dir = os.path.join(tmpdir, "eval_stem")
        shutil.move(extracted_dir, eval_stem_dir)
    print0(f"Placed eval_stem directory at {eval_stem_dir}")


def prepare_stem_eval_data():
    """Download pre-packaged STEM evaluation data zip and extract.
    Falls back to individual HuggingFace downloads if zip download fails.
    """
    base_dir = get_base_dir()
    eval_stem_dir = os.path.join(base_dir, "eval_stem")

    if os.path.exists(eval_stem_dir):
        return eval_stem_dir, get_available_stem_tasks(os.path.join(eval_stem_dir, "eval_data"))

    # Try downloading the pre-packaged zip first
    try:
        print0("Downloading STEM evaluation data package...")
        download_file_with_lock(EVAL_STEM_URL, "eval_stem.zip", postprocess_fn=place_eval_stem)
        return eval_stem_dir, get_available_stem_tasks(os.path.join(eval_stem_dir, "eval_data"))
    except Exception as e:
        print0(f"WARNING: Could not download eval_stem.zip: {e}")
        print0("Falling back to individual dataset downloads from HuggingFace...")

    # Fallback: download individual datasets from HuggingFace
    eval_data_dir = os.path.join(eval_stem_dir, "eval_data")
    os.makedirs(eval_data_dir, exist_ok=True)
    print0("Preparing STEM evaluation data individually (GPQA, GSM8K, MATH-500, MMLU)...")

    from datasets import load_dataset as hf_load_dataset
    available_tasks = []

    # --- GPQA Diamond ---
    try:
        print0("Downloading GPQA Diamond (may take a while due to large CSV)...")
        old_timeout = os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT', '')
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
        gpqa_ds = hf_load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        gpqa_data = []
        rng = random.Random(42)
        for row in gpqa_ds:
            choices = [str(row['Correct Answer']), str(row['Incorrect Answer 1']),
                       str(row['Incorrect Answer 2']), str(row['Incorrect Answer 3'])]
            correct_answer = str(row['Correct Answer'])
            rng.shuffle(choices)
            gold = choices.index(correct_answer)
            query = f"{row['Question']}\nChoices:\n(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}"
            gpqa_data.append({"query": query, "choices": choices, "gold": gold})
        gpqa_path = os.path.join(eval_data_dir, "gpqa_diamond.jsonl")
        with open(gpqa_path, 'w', encoding='utf-8') as f:
            for item in gpqa_data:
                f.write(json.dumps(item) + '\n')
        print0(f"GPQA Diamond: {len(gpqa_data)} examples -> {gpqa_path}")
        available_tasks.append('gpqa_diamond')
    except Exception as e:
        print0(f"WARNING: Could not download GPQA (gated dataset): {e}")
        print0("To add GPQA manually: download from https://huggingface.co/datasets/Idavidrein/gpqa,")
        print0("convert to JSONL format {query, choices, gold}, and place at eval_stem/eval_data/gpqa_diamond.jsonl")
    finally:
        if old_timeout:
            os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = old_timeout
        else:
            os.environ.pop('HF_HUB_DOWNLOAD_TIMEOUT', None)

    # --- GSM8K ---
    try:
        print0("Downloading GSM8K...")
        gsm8k_ds = hf_load_dataset("openai/gsm8k", "main", split="test")
        gsm8k_data = []
        for row in gsm8k_ds:
            gsm8k_data.append({"question": row['question'], "answer": row['answer']})
        gsm8k_path = os.path.join(eval_data_dir, "gsm8k.jsonl")
        with open(gsm8k_path, 'w', encoding='utf-8') as f:
            for item in gsm8k_data:
                f.write(json.dumps(item) + '\n')
        print0(f"GSM8K: {len(gsm8k_data)} examples -> {gsm8k_path}")
        available_tasks.append('gsm8k_cot')
    except Exception as e:
        print0(f"WARNING: Could not download GSM8K: {e}")

    # --- MATH-500 ---
    try:
        print0("Downloading MATH-500...")
        math_ds = hf_load_dataset("HuggingFaceH4/MATH-500", split="test")
        math_data = []
        for row in math_ds:
            math_data.append({"question": row['problem'], "answer": row['solution'], "gold_answer": row['answer']})
        math_path = os.path.join(eval_data_dir, "math500.jsonl")
        with open(math_path, 'w', encoding='utf-8') as f:
            for item in math_data:
                f.write(json.dumps(item) + '\n')
        print0(f"MATH-500: {len(math_data)} examples -> {math_path}")
        available_tasks.append('math_cot')
    except Exception as e:
        print0(f"WARNING: Could not download MATH-500: {e}")

    # --- MMLU (from HuggingFace, answer is int 0-3) ---
    try:
        print0("Downloading MMLU...")
        mmlu_ds = hf_load_dataset("cais/mmlu", "all", split="test")
        mmlu_data = []
        stem_data = []
        for row in mmlu_ds:
            choices = row['choices']
            correct_idx = row['answer']
            query = f"{row['question']}\nChoices:\n(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}"
            item = {"query": query, "choices": choices, "gold": correct_idx}
            mmlu_data.append(item)
            if row['subject'] in STEM_SUBJECT_KEYWORDS:
                stem_data.append(item)
        mmlu_path = os.path.join(eval_data_dir, "mmlu.jsonl")
        with open(mmlu_path, 'w', encoding='utf-8') as f:
            for item in mmlu_data:
                f.write(json.dumps(item) + '\n')
        print0(f"MMLU zeroshot: {len(mmlu_data)} examples -> {mmlu_path}")
        available_tasks.append('mmlu_zeroshot')

        mmlu_stem_path = os.path.join(eval_data_dir, "mmlu_stem.jsonl")
        with open(mmlu_stem_path, 'w', encoding='utf-8') as f:
            for item in stem_data:
                f.write(json.dumps(item) + '\n')
        print0(f"MMLU STEM: {len(stem_data)} examples -> {mmlu_stem_path}")
        available_tasks.append('mmlu_stem')
    except Exception as e:
        print0(f"WARNING: Could not download MMLU: {e}")

    # --- Write eval_meta_data.csv ---
    csv_path = os.path.join(eval_stem_dir, "eval_meta_data.csv")
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Eval Task', 'Random baseline'])
        writer.writerow(['gpqa_diamond', '25.0'])
        writer.writerow(['gsm8k_cot', '0.0'])
        writer.writerow(['math_cot', '0.0'])
        writer.writerow(['mmlu_zeroshot', '25.0'])
        writer.writerow(['mmlu_stem', '25.0'])

    print0(f"STEM evaluation data prepared at {eval_stem_dir}")
    print0(f"Available STEM tasks: {available_tasks}")
    return eval_stem_dir, available_tasks


def get_available_stem_tasks(eval_data_dir):
    """Check which STEM task data files exist and return available task labels."""
    available = []
    for task in STEM_TASKS:
        data_path = os.path.join(eval_data_dir, task['dataset_uri'])
        if os.path.exists(data_path):
            available.append(task['label'])
    return available


def place_eval_bundle(file_path):
    """Unzip eval_bundle.zip and place it in the base directory."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


def evaluate_core(model, tokenizer, device, max_per_task=-1, core_eval_batch_size=1, benchmarks=None):
    """
    Evaluate a base model on selected benchmarks.
    Returns dict with results, centered_results, core_metric.
    core_metric is averaged only over the DCLM core tasks.
    
    benchmarks controls which tasks to evaluate:
      None or 'all'  : evaluate all (core + stem)
      'core'         : evaluate only DCLM core tasks
      'stem'         : evaluate only STEM benchmarks (GPQA, GSM8K, MATH_COT, MMLU)
      comma-separated: evaluate specific benchmarks by label (e.g. 'mmlu_fewshot,gsm8k_cot')
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # Download the eval bundle if needed
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    core_tasks = config['icl_tasks']
    core_task_labels = [t['label'] for t in core_tasks]

    # Determine which STEM tasks are available
    need_stem = benchmarks is None or benchmarks == 'all' or benchmarks == 'stem' or \
        (benchmarks and any(t['label'] in benchmarks for t in STEM_TASKS))
    if need_stem:
        eval_stem_dir, available_stem_labels = prepare_stem_eval_data()
        stem_tasks = [t for t in STEM_TASKS if t['label'] in available_stem_labels]
    else:
        stem_tasks = []
        eval_stem_dir = None

    # Select tasks based on benchmarks parameter
    if benchmarks is None or benchmarks == 'all':
        selected_core = core_tasks
        selected_stem = stem_tasks
    elif benchmarks == 'core':
        selected_core = core_tasks
        selected_stem = []
    elif benchmarks == 'stem':
        # STEM preset: includes tasks matching STEM_BENCHMARK_LABELS from both core.yaml and stem data
        selected_core = [t for t in core_tasks if t['label'] in STEM_BENCHMARK_LABELS]
        selected_stem = [t for t in stem_tasks if t['label'] in STEM_BENCHMARK_LABELS]
    else:
        # Specific benchmark labels
        benchmark_set = set(benchmarks)
        selected_core = [t for t in core_tasks if t['label'] in benchmark_set]
        selected_stem = [t for t in stem_tasks if t['label'] in benchmark_set]

    # Load random baselines (merge core + stem)
    random_baselines = {}
    for csv_path in [
        os.path.join(eval_bundle_dir, "eval_meta_data.csv"),
        os.path.join(eval_stem_dir or "", "eval_meta_data.csv"),
    ]:
        if csv_path and os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    random_baselines[row['Eval Task']] = float(row['Random baseline'])

    # All task groups: (task_list, data_base_path)
    all_task_groups = [
        (selected_core, os.path.join(eval_bundle_dir, "eval_data")),
    ]
    if eval_stem_dir and selected_stem:
        all_task_groups.append((selected_stem, os.path.join(eval_stem_dir, "eval_data")))

    # Evaluate each task
    results = {}
    centered_results = {}
    for tasks, data_base_path in all_task_groups:
        for task in tasks:
            start_time = time.time()
            label = task['label']
            task_meta = {
                'task_type': task['icl_task_type'],
                'dataset_uri': task.get('dataset_uri', ''),
                'num_fewshot': task['num_fewshot'][0],
                'continuation_delimiter': task.get('continuation_delimiter', ' '),
                'answer_extractor': task.get('answer_extractor', 'gsm8k'),
                'max_gen_tokens': task.get('max_gen_tokens', 512),
                'label': label,
            }
            print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

            data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
            with open(data_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line.strip()) for line in f]

            # Shuffle for consistent subsampling when using max_per_task
            shuffle_rng = random.Random(1337)
            shuffle_rng.shuffle(data)
            if max_per_task > 0:
                data = data[:max_per_task]

            if task_meta['task_type'] == 'generation':
                accuracy = evaluate_generation_task(model, tokenizer, data, device, task_meta)
            else:
                accuracy = evaluate_task(model, tokenizer, data, device, task_meta, eval_batch_size=core_eval_batch_size)
            if device.type == "npu":
                torch.npu.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
            results[label] = accuracy
            random_baseline = random_baselines.get(label, 0.0)
            if random_baseline > 0:
                centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
            else:
                centered_result = accuracy
            centered_results[label] = centered_result
            elapsed = time.time() - start_time
            print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s")

    # CORE metric: only meaningful when all 22 DCLM core tasks are evaluated
    core_centered = {k: v for k, v in centered_results.items() if k in core_task_labels}
    if len(core_centered) == len(core_task_labels):
        core_metric = sum(core_centered.values()) / len(core_centered)
    else:
        core_metric = None

    # STEM metric: average over STEM_BENCHMARK_LABELS that were actually evaluated
    stem_centered = {k: v for k, v in centered_results.items() if k in STEM_BENCHMARK_LABELS}
    stem_metric = sum(stem_centered.values()) / len(stem_centered) if stem_centered else None

    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
        "stem_metric": stem_metric,
    }
    return out

# -----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="Base model evaluation")
    parser.add_argument('--eval', type=str, default='core,bpb,sample', help='Comma-separated evaluations to run: core,bpb,sample (default: all)')
    parser.add_argument('--hf-path', type=str, default=None, help='HuggingFace model path (e.g. openai-community/gpt2-xl)')
    parser.add_argument('--model-tag', type=str, default=None, help='nanochat model tag to identify the checkpoint directory')
    parser.add_argument('--step', type=int, default=None, help='Model step to load (default = last)')
    parser.add_argument('--model-type', type=str, default='base', choices=['base', 'mid', 'sft', 'rl'], help='Type of model to evaluate (base, mid, sft, rl)')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per CORE task (-1 = all)')
    parser.add_argument('--device-batch-size', type=int, default=32, help='Per-device batch size for BPB evaluation')
    parser.add_argument('--core-eval-batch-size', type=int, default=16, help='Number of examples to batch per forward pass in CORE eval (1 = original behavior)')
    parser.add_argument('--eval-benchmarks', type=str, default=None, help='Benchmarks to evaluate: all, core, stem, or comma-separated labels (e.g. mmlu_fewshot,gsm8k_cot). Default: all')
    parser.add_argument('--split-tokens', type=int, default=40*524288, help='Number of tokens to evaluate per split for BPB')
    parser.add_argument('--device-type', type=str, default='', help='cuda|cpu|mps (empty = autodetect)')
    args = parser.parse_args()

    # Parse evaluation modes
    eval_modes = set(mode.strip() for mode in args.eval.split(','))
    valid_modes = {'core', 'bpb', 'sample'}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")

    # Parse benchmarks parameter
    benchmarks = args.eval_benchmarks
    if benchmarks is not None and benchmarks not in ('all', 'core', 'stem'):
        benchmarks = [b.strip() for b in benchmarks.split(',')]

    # Distributed / precision setup
    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    master_process = ddp_rank == 0  # 仅主进程打印日志、保存结果
    # # ===================== NPU适配：混合精度上下文 =====================
    # if device_type == "cuda":
    #     autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    # elif device_type == "npu":
    #     autocast_ctx = torch.npu.amp.autocast(dtype=torch.bfloat16)  # NPU原生BF16
    # else:
    #     autocast_ctx = nullcontext()


    # Load model and tokenizer
    is_hf_model = args.hf_path is not None
    if is_hf_model:
        model, tokenizer = load_hf_model(args.hf_path, device)
        sequence_len = model.max_seq_len or 1024
        token_bytes = get_hf_token_bytes(tokenizer, device=device)
        model_name = args.hf_path
        model_slug = args.hf_path.replace("/", "-")
    else:
        model, tokenizer, meta = load_model(args.model_type, device, phase="eval", model_tag=args.model_tag, step=args.step)
        sequence_len = meta["model_config"]["sequence_len"]
        token_bytes = get_token_bytes(device=device)
        model_name = f"{args.model_type}_model (step {meta['step']})"
        model_slug = f"{args.model_type}_model_{meta['step']:06d}"

    print0(f"Evaluating model: {model_name}")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

    # Results to log
    core_results = None
    bpb_results = {}
    samples = []
    unconditioned_samples = []

    # --- Sampling ---
    if 'sample' in eval_modes and not is_hf_model:
        print0("\n" + "="*80)
        print0("Model Samples")
        print0("="*80)
        if ddp_rank == 0:
            prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]
            engine = Engine(model, tokenizer)
            print0("\nConditioned samples:")
            for prompt in prompts:
                tokens = tokenizer(prompt, prepend="<|bos|>")
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                sample_str = tokenizer.decode(sample[0])
                print0("-" * 80)
                print0(sample_str)
                samples.append(sample_str)

            print0("\nUnconditioned samples:")
            tokens = tokenizer("", prepend="<|bos|>")
            uncond, _ = engine.generate_batch(tokens, num_samples=8, max_tokens=128, temperature=1.0)
            for sample in uncond:
                sample_str = tokenizer.decode(sample)
                print0("-" * 80)
                print0(sample_str)
                unconditioned_samples.append(sample_str)
    elif 'sample' in eval_modes and is_hf_model:
        print0("\nSkipping sampling for HuggingFace models (not supported)")

    # --- BPB evaluation ---
    if 'bpb' in eval_modes:
        print0("\n" + "="*80)
        print0("BPB Evaluation")
        print0("="*80)
        tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
        if args.split_tokens % tokens_per_step != 0:
            # Adjust to nearest multiple
            args.split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step
            print0(f"Adjusted split_tokens to {args.split_tokens} (must be divisible by {tokens_per_step})")
        steps = args.split_tokens // tokens_per_step

        for split_name in ["train", "val"]:
            loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, sequence_len, split_name, device=device)
            bpb = evaluate_bpb(model, loader, steps, token_bytes)
            bpb_results[split_name] = bpb
            print0(f"{split_name} bpb: {bpb:.6f}")

    # --- CORE evaluation ---
    if 'core' in eval_modes:
        print0("\n" + "="*80)
        print0("CORE Evaluation")
        print0("="*80)
        core_results = evaluate_core(model, tokenizer, device, max_per_task=args.max_per_task, core_eval_batch_size=args.core_eval_batch_size, benchmarks=benchmarks)

        # Write CSV output
        if ddp_rank == 0:
            base_dir = get_base_dir()
            output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                for label in core_results["results"]:
                    acc = core_results["results"][label]
                    centered = core_results["centered_results"][label]
                    f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
                if core_results['core_metric'] is not None:
                    f.write(f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")
                if core_results['stem_metric'] is not None:
                    f.write(f"{'STEM':<35}, {'':<10}, {core_results['stem_metric']:<10.6f}\n")
            print0(f"\nResults written to: {output_csv_path}")
            if core_results['core_metric'] is not None:
                print0(f"CORE metric: {core_results['core_metric']:.4f}")
            if core_results['stem_metric'] is not None:
                print0(f"STEM metric: {core_results['stem_metric']:.4f}")

    # --- Log to report ---
    from nanochat.report import get_report
    report_data = [{"model": model_name}]

    if core_results:
        report_data[0]["CORE metric"] = core_results["core_metric"]
        report_data.append(core_results["centered_results"])

    if bpb_results:
        report_data[0]["train bpb"] = bpb_results.get("train")
        report_data[0]["val bpb"] = bpb_results.get("val")

    if samples:
        report_data.append({f"sample {i}": s for i, s in enumerate(samples)})
    if unconditioned_samples:
        report_data.append({f"unconditioned {i}": s for i, s in enumerate(unconditioned_samples)})

    # 记得改
    get_report().log(section=f"{args.model_type.capitalize()} model evaluation", data=report_data)

    compute_cleanup()
    # 资源释放与优化
    if device_type == "npu":
        torch.npu.empty_cache()

if __name__ == "__main__":
    main()