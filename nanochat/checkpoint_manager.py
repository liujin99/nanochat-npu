"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
import json
import logging
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0(f"Patching missing window_pattern in model config to 'L'")

def _patch_missing_keys(model_data, model_config, device=None):
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layer
    kwargs = {"device": device} if device is not None else {}
    # resid_lambdas defaults to 1.0 (identity scaling)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer, **kwargs)
        log0(f"Patching missing resid_lambdas in model data to 1.0")
    # x0_lambdas defaults to 0.0 (disabled)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer, **kwargs)
        log0(f"Patching missing x0_lambdas in model data to 0.0")
    # smear_lambda defaults to 0.0 (disabled)
    if "smear_lambda" not in model_data:
        model_data["smear_lambda"] = torch.zeros(1, **kwargs)
        log0(f"Patching missing smear_lambda in model data to 0.0")
    # backout_lambda defaults to 0.2
    if "backout_lambda" not in model_data:
        model_data["backout_lambda"] = 0.2 * torch.ones(1, **kwargs)
        log0(f"Patching missing backout_lambda in model data to 0.2")
    # smear_gate.weight defaults to uniform(0.0, 0.02), shape (1, 24)
    if "smear_gate.weight" not in model_data:
        model_data["smear_gate.weight"] = torch.empty(1, 24, **kwargs).uniform_(0.0, 0.02)
        log0(f"Patching missing smear_gate.weight in model data")

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0, param_names=None):
    # =========== 新增：若目标文件夹已存在，清空已有文件夹内容（避免混淆）==============
    if rank == 0:
        # 先判断文件夹是否存在
        if os.path.exists(checkpoint_dir):
            # 遍历文件夹内所有文件/子文件夹，安全删除
            for filename in os.listdir(checkpoint_dir):
                file_path = os.path.join(checkpoint_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.error(f'删除旧文件失败 {file_path}: {e}')
            logger.info(f"已清空 checkpoint 目录: {checkpoint_dir}")
    # ============================================================================

    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")
        if param_names is not None:
            names_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}_names.pt")
            torch.save(param_names, names_path)

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config, device)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)

def load_optimizer_state(source, device, rank, model_tag=None, step=None):
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
    if not os.path.exists(optimizer_path):
        log0(f"Optimizer checkpoint not found: {optimizer_path}")
        return None
    log0(f"Loading optimizer state from {optimizer_path}")
    optimizer_data = torch.load(optimizer_path, map_location=device)
    return optimizer_data


def migrate_optimizer_state(optimizer, saved_state_dict, model, saved_param_names=None):
    """
    Migrate optimizer state from a checkpoint with potentially different param groups.
    Uses parameter NAME matching (not positional) to correctly map state across
    different optimizer group structures.

    Args:
        optimizer: the new optimizer with current param groups
        saved_state_dict: the loaded optimizer state_dict from checkpoint
        model: the current model (used to build name→param mapping)
        saved_param_names: list of param names saved alongside the checkpoint (optional)
    """
    old_state = saved_state_dict['state']
    old_param_groups = saved_state_dict['param_groups']
    new_param_groups = optimizer.param_groups

    # --- Build id→name mapping for new optimizer params ---
    id_to_name = {id(p): name for name, p in model.named_parameters()}

    # --- Build old_id → name mapping ---
    if saved_param_names is not None:
        old_flat_ids = [pid for g in old_param_groups for pid in g['params']]
        if len(saved_param_names) == len(old_flat_ids):
            old_id_to_name = dict(zip(old_flat_ids, saved_param_names))
        else:
            old_id_to_name = _infer_old_param_names(old_param_groups, new_param_groups, model, old_state)
    else:
        old_id_to_name = _infer_old_param_names(old_param_groups, new_param_groups, model, old_state)

    # --- Build name → new_param mapping ---
    name_to_new_param = dict(model.named_parameters())

    # --- Build kind lookup ---
    old_id_to_kind = {}
    for g in old_param_groups:
        for pid in g['params']:
            old_id_to_kind[pid] = g.get('kind', 'adamw')
    new_name_to_kind = {}
    for g in new_param_groups:
        for p in g['params']:
            name = id_to_name.get(id(p))
            if name:
                new_name_to_kind[name] = g.get('kind', 'adamw')

    # --- Migrate AdamW state (per-param: exp_avg, exp_avg_sq, step) ---
    migrated_count = 0
    for old_id, state in old_state.items():
        name = old_id_to_name.get(old_id)
        if name is None or name not in name_to_new_param:
            continue
        new_param = name_to_new_param[name]
        old_kind = old_id_to_kind.get(old_id, 'adamw')
        new_kind = new_name_to_kind.get(name, 'adamw')
        if old_kind == 'adamw' and new_kind == 'adamw':
            optimizer.state[new_param] = state
            migrated_count += 1

    # --- Migrate Muon state (stacked buffers stored in first param of each group) ---
    old_muon = [(i, g) for i, g in enumerate(old_param_groups) if g.get('kind') == 'muon']
    new_muon = [(i, g) for i, g in enumerate(new_param_groups) if g.get('kind') == 'muon']

    old_id_to_shape = {old_id: name_to_new_param[name].shape
                       for old_id, name in old_id_to_name.items()
                       if name in name_to_new_param}

    used_old_muon = set()
    muon_migrated = 0
    for new_idx, new_group in new_muon:
        new_shapes = tuple(p.shape for p in new_group['params'])
        for old_idx, old_group in old_muon:
            if old_idx in used_old_muon:
                continue
            old_shapes = tuple(old_id_to_shape.get(pid, torch.Size()) for pid in old_group['params'])
            if old_shapes == new_shapes and len(old_group['params']) == len(new_group['params']):
                old_first_id = old_group['params'][0]
                new_first_param = new_group['params'][0]
                if old_first_id in old_state and 'momentum_buffer' in old_state[old_first_id]:
                    new_s = optimizer.state.get(new_first_param, {})
                    new_s['momentum_buffer'] = old_state[old_first_id]['momentum_buffer']
                    if 'second_momentum_buffer' in old_state[old_first_id]:
                        new_s['second_momentum_buffer'] = old_state[old_first_id]['second_momentum_buffer']
                    optimizer.state[new_first_param] = new_s
                    muon_migrated += 1
                used_old_muon.add(old_idx)
                break

    total_old = len(old_state)
    total_migrated = migrated_count + muon_migrated
    log0(f"Migrated optimizer state: {total_migrated}/{total_old} states "
         f"(AdamW: {migrated_count}, Muon groups: {muon_migrated})")
    return total_migrated > 0


def _infer_old_param_names(old_param_groups, new_param_groups, model, old_state):
    """
    Infer old param names by matching groups by shape signature,
    then using new model's param names within matched groups.
    Fallback for old checkpoints that don't have saved param names.
    """
    id_to_name = {id(p): name for name, p in model.named_parameters()}

    old_adamw = [g for g in old_param_groups if g.get('kind') != 'muon']
    new_adamw = [g for g in new_param_groups if g.get('kind') != 'muon']
    old_muon = [g for g in old_param_groups if g.get('kind') == 'muon']
    new_muon = [g for g in new_param_groups if g.get('kind') == 'muon']

    old_id_to_name = {}

    def match_groups(old_groups, new_groups):
        used_new = set()
        for old_g in old_groups:
            for ng_idx, new_g in enumerate(new_groups):
                if ng_idx in used_new:
                    continue
                if len(old_g['params']) != len(new_g['params']):
                    continue
                new_params = new_g['params']
                all_ok = True
                for i, pid in enumerate(old_g['params']):
                    if pid in old_id_to_name:
                        continue
                    name = id_to_name.get(id(new_params[i]))
                    if name is None:
                        all_ok = False
                        break
                    if pid in old_state and 'exp_avg' in old_state[pid]:
                        if old_state[pid]['exp_avg'].shape != new_params[i].shape:
                            all_ok = False
                            break
                    old_id_to_name[pid] = name
                if all_ok:
                    used_new.add(ng_idx)
                    break

    match_groups(old_adamw, new_adamw)
    match_groups(old_muon, new_muon)

    return old_id_to_name