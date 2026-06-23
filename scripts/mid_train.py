"""
Middle training the model. Between pretraining and SFT, using high-quality data for annealing.
Run as:

python -m scripts.mid_train

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device-batch-size=8
"""

import gc
import argparse
import os
import time
import math
import wandb
import torch
from contextlib import nullcontext
from nanochat.gpt import GPT, GPTConfig
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type, \
    get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, get_dist_info
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_model, load_optimizer_state
from nanochat.loss_eval import evaluate_bpb
import torch.distributed as dist
from nanochat.flash_attention import HAS_FA3
from nanochat.engine import Engine
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, \
    tokenizing_distributed_data_loader_with_state_bos_bestfit
from scripts.base_eval import evaluate_core

import torch_npu

torch.npu.empty_cache()
import ssl
import urllib

ssl._create_default_https_context = ssl._create_unverified_context

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Middle training the model")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps|npu (empty = autodetect)")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--load-optimizer", type=int, default=1,
                    help="warm-start optimizer from pretrained checkpoint (0=no, 1=yes)")
parser.add_argument("--num-iterations", type=int, default=-1,
                    help="number of optimization steps (-1 = calculate from other params)")
parser.add_argument("--target-param-data-ratio", type=float, default=0.1,
                    help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1, help="target FLOPs to train for (-1 = disable)")
parser.add_argument("--max-seq-len", type=int, default=None, help="max context length (default: inherit from pretrain)")
parser.add_argument("--device-batch-size", type=int, default=None,
                    help="per-device batch size (default: inherit from pretrain)")
parser.add_argument("--total-batch-size", type=int, default=None,
                    help="total batch size in tokens (default: inherit from pretrain)")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")

# -------------- 修复缺失的参数，解决报错 --------------
parser.add_argument("--embedding-lr", type=float, default=None,
                    help="learning rate for embedding parameters (Adam) (default: inherit from pretrain)")
parser.add_argument("--unembedding-lr", type=float, default=None,
                    help="learning rate for unembedding parameters (Adam) (default: inherit from pretrain)")
parser.add_argument("--matrix-lr", type=float, default=None,
                    help="learning rate for matrix parameters (Muon) (default: inherit from pretrain)")
# -------------------------------------------------------

parser.add_argument("--lr-scale", type=float, default=1.0, help="工业界：直接缩放预训练学习率 (1.0 = 完全接续)")
parser.add_argument("--weight-decay", type=float, default=0.28,
                    help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.9, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR (与预训练一致)")
parser.add_argument("--eval-every", type=int, default=100, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=20 * 524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=500,
                    help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=500, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
default_mid_train_data = os.path.join(get_base_dir(), "mid_train_data")
parser.add_argument("--data-dir", type=str, default=default_mid_train_data,
                    help="directory containing high-quality training data")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------


# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

if device_type == "cuda":
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=COMPUTE_DTYPE)
    synchronize = torch.cuda.synchronize
    get_max_memory = torch.cuda.max_memory_allocated
elif device_type == "npu":
    autocast_ctx = torch.npu.amp.autocast(dtype=COMPUTE_DTYPE)
    synchronize = torch.npu.synchronize
    get_max_memory = torch.npu.max_memory_allocated
else:
    autocast_ctx = nullcontext()
    synchronize = lambda: None
    get_max_memory = lambda: 0

if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
elif device_type == "npu":
    npu_device_name = torch.npu.get_device_name(0)
    gpu_peak_flops = get_peak_flops(npu_device_name)
    print0(f"NPU: {npu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-mid", name=args.run, config=user_config)

# Load the model and tokenizer
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)

# Inherit training hyperparameters from pretrained checkpoint
pretrain_user_config = meta.get("user_config", {})
for name, fallback, source in [
    ("max_seq_len", 2048, meta),
    ("device_batch_size", 32, meta),
    ("total_batch_size", 524288, meta),
    ("embedding_lr", 0.3, pretrain_user_config),
    ("unembedding_lr", 0.008, pretrain_user_config),
    ("matrix_lr", 0.02, pretrain_user_config),
]:
    arg_val = getattr(args, name)
    pretrain_val = source.get(name)
    if arg_val is None:
        resolved = pretrain_val if pretrain_val is not None else fallback
        setattr(args, name, resolved)
        print0(f"Inherited {name}={resolved} from pretrained checkpoint")
    elif pretrain_val is not None and arg_val != pretrain_val:
        print0(f"NOTE: --{name.replace('_', '-')}={arg_val} overrides pretrained value of {pretrain_val}")
    else:
        print0(f"Using {name}={arg_val}")

orig_model = model
depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size

total_batch_size = args.total_batch_size
if total_batch_size % world_tokens_per_fwdbwd != 0:
    recommended_k = (total_batch_size + world_tokens_per_fwdbwd - 1) // world_tokens_per_fwdbwd
    recommended_total_batch_size = recommended_k * world_tokens_per_fwdbwd
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        import warnings

        warnings.warn(f"Adjusted total_batch_size to {recommended_total_batch_size}")
    total_batch_size = recommended_total_batch_size

assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Grad accum steps: {grad_accum_steps}")
token_bytes = get_token_bytes(device=device)
base_dir = get_base_dir()


def get_scaling_params(m):
    params_counts = m.num_scaling_params()
    return params_counts['transformer_matrices'] + params_counts['lm_head']


num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params)


def build_model_meta(depth):
    config = model.config
    with torch.device("meta"):
        return GPT(config)


d12_ref = build_model_meta(12)
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref)
B_REF = 2 ** 19
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
optimizer = model.setup_optimizer(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr,
                                  matrix_lr=args.matrix_lr, weight_decay=weight_decay_scaled)

# ----------------===== 工业界正确：接续预训练最终学习率 =====----------------
if args.load_optimizer:
    optimizer_data = load_optimizer_state("base", device, rank=ddp_rank, model_tag=args.model_tag, step=args.model_step)
    if optimizer_data is not None:
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data
        for group in optimizer.param_groups:
            group["lr"] = group["lr"] * args.lr_scale
            group["initial_lr"] = group["lr"]
    else:
        print0("WARNING: optimizer checkpoint not found")
# -----------------------------------------------------------------------------

scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None

train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="train",
    device=device, tokenizer_threads=16, tokenizer_batch_size=256, buffer_size=2000, data_dir=args.data_dir)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="val",
    device=device, tokenizer_threads=16, tokenizer_batch_size=256, buffer_size=2000, data_dir=args.data_dir)
x, y, dataloader_state_dict = next(train_loader)
x = x.to(device, non_blocking=True)
y = y.to(device, non_blocking=True)

# Compute iterations
param_counts = model.num_scaling_params()
num_scaling_params = param_counts['total']
if args.num_iterations > 0:
    num_iterations = args.num_iterations
elif args.target_flops > 0:
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
else:
    target_tokens = int(args.target_param_data_ratio * num_scaling_params)
    num_iterations = target_tokens // total_batch_size
total_tokens = total_batch_size * num_iterations
print0(f"Total tokens: {total_tokens:,}, Steps: {num_iterations:,}")


# LR scheduler
def get_lr_multiplier(it, num_iterations):
    warmup_iters = int(args.warmup_ratio * num_iterations)
    warmdown_iters = int(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac


def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))


min_val_bpb = float("inf")
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0
step = 0
val_bpb = None

while True:
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({"step": step, "val/bpb": val_bpb})
        model.train()

    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({"step": step, "core_metric": results["core_metric"]})
        model.train()

    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = ["The capital of France is", "The chemical symbol of gold is",
                   "If yesterday was Friday, then tomorrow will be", "The planets of the solar system are:"]
        engine = Engine(orig_model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    if last_step or (step > 0 and args.save_every > 0 and step % args.save_every == 0):
        output_dirname = args.model_tag if args.model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "mid_checkpoints", output_dirname)

        model_config_save = {
            "sequence_len": model.config.sequence_len,
            "vocab_size": tokenizer.get_vocab_size(),
            "n_layer": model.config.n_layer,
            "n_head": model.config.n_head,
            "n_kv_head": model.config.n_kv_head,
            "n_embd": model.config.n_embd,
            "window_pattern": model.config.window_pattern,
        }

        save_checkpoint(
            checkpoint_dir, step, orig_model.state_dict(), optimizer.state_dict(), {
                "step": step, "val_bpb": val_bpb,
                "model_config": model_config_save,
                "user_config": user_config,
                "device_batch_size": args.device_batch_size,
                "total_batch_size": total_batch_size
            },
            rank=ddp_rank, )

    if last_step:
        break

    synchronize()
    t0 = time.time()
    train_loss_f = 0.0
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        loss = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(train_loader)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        train_loss_f += loss.detach().item()

    lrm = get_lr_multiplier(step, num_iterations)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group.get('kind') == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay

    if scaler is not None:
        scaler.unscale_(optimizer)
        if ddp:
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        has_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in model.parameters())
        if dist.is_initialized():
            dev = next(model.parameters()).device
            nan_flag = torch.tensor([1.0 if has_nan else 0.0], device=dev)
            dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)
            has_nan = nan_flag.item() > 0
        if has_nan:
            print(f"[WARNING] NaN gradients detected at step {step}, skipping optimizer.step()")
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    dt = time.time() - t0

    # ===================== 日志打印 =====================
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds / 60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time / 60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)
    # ====================================================================

    step += 1

from nanochat.report import get_report

final_flops_per_sec = num_flops_per_token * total_batch_size / dt if dt > 0 else 0
final_mfu = 100 * final_flops_per_sec / (gpu_peak_flops * ddp_world_size)

# 写入报告
get_report().log(section="Mid model training", data=[
    user_config,
    {
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "device_batch_size": args.device_batch_size,
        "total_batch_size": total_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "warmup_ratio": args.warmup_ratio,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    {
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None) if 'results' in locals() else None,
        "Final MFU %": f"{final_mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time / 60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

wandb_run.finish()
compute_cleanup()

if device_type == "npu":
    torch.npu.empty_cache()
    gc.collect()