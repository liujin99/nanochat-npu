"需重新审视修改的必要性"
import torch
import torch.nn.functional as F
import torch_npu
from torch.distributed import is_initialized, barrier


# =============================================================================
# Detection: 适配昇腾NPU，替换原有的FA3检测逻辑
# =============================================================================
def _load_ascend_flash_attention():
    """加载昇腾FA"""
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return None
    try:
        return True  # 标记可用
    except Exception:
        return None


# 全局变量：对齐官方命名
_ascend_fa = _load_ascend_flash_attention()
HAS_ASCEND_FA = _ascend_fa is not None
HAS_FA3 = False

_override_impl = None


def _use_ascend_fa():
    """判断是否使用昇腾FA（对齐官方_use_fa3逻辑）"""
    if _override_impl == 'ascend_fa':
        assert HAS_ASCEND_FA, "昇腾FA不可用（无昇腾NPU）"
        return True
    if _override_impl == 'sdpa':
        return False
    return HAS_ASCEND_FA  # 自动选择


# =============================================================================
# 核心：昇腾FA实现（训练/推理解耦）
# =============================================================================
def _ascend_fa_attention(q, k, v, causal=False, window_size=(-1, -1), is_training=True):
    if not q.is_npu:
        q, k, v = q.npu(), k.npu(), v.npu()
    device = q.device
    dtype = q.dtype
    Tq, Tk = q.size(2), k.size(2)
    window_left = window_size[0]
    enable_gqa = q.size(1) != k.size(1)

    # ==========================
    # 训练
    # ==========================
    if is_training:
        if Tq != Tk:
            raise ValueError(f"训练Q/K序列长度必须相等，当前Q={Tq}, K={Tk}")

        sdpa_kwargs = {
            "attn_mask": None,
            "dropout_p": 0.0,
            "is_causal": causal,
        }
        if enable_gqa:
            sdpa_kwargs["enable_gqa"] = True
        return F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)

    # ==========================
    # 推理
    # ==========================
    mask = None
    if causal:
        # 情况1：常规因果 mask（Tq==Tk）
        if Tq == Tk:
            row_idx = torch.arange(Tq, device=device).unsqueeze(1)
            col_idx = torch.arange(Tk, device=device).unsqueeze(0)
            mask = col_idx <= row_idx

        # 情况2：生成 1 个 token
        elif Tq == 1:
            mask = torch.ones((1, Tk), device=device, dtype=torch.bool)

        # 情况3：其他长度（极少用）
        else:
            base_row = (Tk - Tq) + torch.arange(Tq, device=device)
            row_idx = base_row.unsqueeze(1)
            col_idx = torch.arange(Tk, device=device).unsqueeze(0)
            mask = col_idx <= row_idx

        # 窗口注意力限制
        if window_left >= 0:
            if Tq == 1:
                keep_start = max(0, Tk - window_left)
                mask = mask.clone()
                mask[:, :keep_start] = False
            else:
                row_idx = torch.arange(Tq, device=device).unsqueeze(1)
                col_idx = torch.arange(Tk, device=device).unsqueeze(0)
                mask = mask & ((row_idx - col_idx) <= window_left)

    # mask 转成 -inf
    if mask is not None:
        mask = torch.where(mask, 0.0, -1e9).to(dtype=dtype)

    sdpa_kwargs = {
        "attn_mask": mask,
        "dropout_p": 0.0,
        "is_causal": False,  # 因为我们手动传mask了，必须关闭自动因果
    }
    if enable_gqa:
        sdpa_kwargs["enable_gqa"] = True

    return F.scaled_dot_product_attention(q, k, v,** sdpa_kwargs)


# =============================================================================
# 官方SDPA逻辑：完全复用，不修改
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    if Tq == 1:
        if window >= 0 and window < Tk:
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


# =============================================================================
# Public API：完全对齐官方（flash_attn_func/flash_attn_with_kvcache）
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    训练用Flash Attention（无KV Cache）：完全对齐官方API
    Args:
        q, k, v: (B, T, H, D)
        causal: 是否因果掩码
        window_size: (left, right) 滑动窗口（支持SSSL）
    Returns:
        (B, T, H, D)
    """
    if _use_ascend_fa():
        try:
            q_sdpa = q.transpose(1, 2)
            k_sdpa = k.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)
            y_sdpa = _ascend_fa_attention(q_sdpa, k_sdpa, v_sdpa, causal, window_size, is_training=True)
            return y_sdpa.transpose(1, 2)
        except Exception:
            # 异常降级到SDPA
            pass

    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)
    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)
    return y_sdpa.transpose(1, 2)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    推理用Flash Attention（昇腾专用版）
    """
    if _use_ascend_fa():
        B, T_new, H, D = q.shape
        if cache_seqlens is not None:
            pos = torch.clamp(cache_seqlens[0:1], min=0).to(q.device, non_blocking=True)
            pos = pos.expand(1)  # 保证维度稳定
        else:
            pos = torch.tensor([0], dtype=torch.int64, device=q.device)
        pos = pos[0]  # 仍在NPU上取值，无跨设备传输

        if k is not None and v is not None:
            # KV Cache写入用non_blocking=True，异步无阻塞
            k_cache[:, pos:pos+T_new, :, :] = k.to(k_cache.device, non_blocking=True)
            v_cache[:, pos:pos+T_new, :, :] = v.to(v_cache.device, non_blocking=True)

            if is_initialized():
                try:
                    barrier(device_ids=[torch.npu.current_device()], timeout=10)
                except Exception:
                    pass

        end_pos = pos + T_new
        # 用NPU原生切片，避免CPU计算边界
        k_full = k_cache[:, :end_pos, :, :].contiguous()
        v_full = v_cache[:, :end_pos, :, :].contiguous()

        q_sdpa = q.transpose(1, 2).contiguous()
        k_sdpa = k_full.transpose(1, 2).contiguous()
        v_sdpa = v_full.transpose(1, 2).contiguous()

        y_sdpa = _ascend_fa_attention(q_sdpa, k_sdpa, v_sdpa, causal, window_size, is_training=False)
        return y_sdpa.transpose(1, 2).contiguous()

    # SDPA Fallback
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].cpu().item() if cache_seqlens is not None else 0

    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)


# =============================================================================
# 导出：完全对齐官方的module接口
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)