#!/bin/bash
set -e
set -o noglob  # 新增：防止空参数解析错误


export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR


# 自动获取脚本所在目录的上一级（项目根）
SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
NANOCHAT_ROOT=$(dirname "$SCRIPT_DIR")
# 切换到项目根目录，配置Python路径（解决ModuleNotFoundError）
cd "$NANOCHAT_ROOT"

# 使用 nano conda 环境的绝对路径（避免 conda activate 不生效）
PYTHON="/home/ma-user/.conda/envs/nano/bin/python"
export PATH="/home/ma-user/.conda/envs/nano/bin:$PATH"
export PYTHON_EXECUTABLE="$PYTHON"  # 强制 torch.distributed.run 使用正确的 Python


# 加载昇腾CANN环境（适配CANN-8.3.RC2）
source /usr/local/Ascend/ascend-toolkit/set_env.sh


## ========== 多卡HCCL通信核心 ==========
# 指定HCCL路径（指向你的hccl目录）
export ASCEND_HCCL_PATH=/usr/local/Ascend/ascend-toolkit/latest/hccl
# 追加HCCL库路径（多卡通信必需）
export LD_LIBRARY_PATH=${ASCEND_HCCL_PATH}/lib64:$LD_LIBRARY_PATH
# 延长HCCL通信超时（避免8卡握手失败）
export HCCL_CONNECT_TIMEOUT=1200
# 禁用HCCL白名单（无root集群适配）
export HCCL_WHITELIST_DISABLE=1
# 禁用IB网卡（适配以太网通信，多数集群用）
export NCCL_IB_DISABLE=1
# 分布式训练优化
export NCCL_SOCKET_IFNAME=eth0

# 昇腾NPU环境配置
export PYTORCH_ALLOC_CONF=expandable_segments:True
export ASCEND_GLOBAL_LOG_LEVEL=3 # 降低日志级别，减少干扰

# 确保NPU设备ID与LOCAL_RANK对应
if [ -z "$ASCEND_DEVICE_ID" ] && [ -n "$LOCAL_RANK" ]; then
    export ASCEND_DEVICE_ID=$LOCAL_RANK
elif [ -z "$ASCEND_DEVICE_ID" ]; then
    export ASCEND_DEVICE_ID=0
fi


# 8卡NPU核心配置（替换原有单卡配置）
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 8卡NPU
export RANK_SIZE=8                              # 分布式总进程数
export MASTER_ADDR=127.0.0.1                    # 单机通信地址
export MASTER_PORT=29500                        # 通信端口
export HCCL_CONNECT_TIMEOUT=1200                 # 分布式通信超时
export HCCL_EXEC_TIMEOUT=1200
export ASCEND_COMPILER_PATH=/home/ma-user/Ascend/ascend-toolkit/latest/compiler # 配置编译器路径（提升算子编译效率）
export ASCEND_DISABLE_MEM_SWAP=1   # 禁用NPU内存交换（提速30%）
export ASCEND_LAUNCH_BLOCKING=0   # 禁用多卡训练的 “同步阻塞”
export NPU_DISABLE_RECORD=1     # 关闭冗余日志记录
export PYTHONUNBUFFERED=1      # 关闭Python缓冲，降低IO耗时
export ASCEND_COMPILE_OPT_LEVEL=O3  # 最高级编译优化
export TORCH_NPU_LAZY_COMPILE=1     # 惰性编译
export OMP_NUM_THREADS=1
export PYTHONPRELOAD=torch_npu                  # 预加载torch-npu，避免import报错
# export NANOCHAT_BASE_DIR="$NANOCHAT_ROOT"
# export TIKTOKEN_CACHE_DIR="$NANOCHAT_ROOT/resource"  # 启用缓存目录
export TORCH_NPU_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256,memory_pool:True"
# 优化NPU内存管理和性能设置
export PYTORCH_NPU_ALLOC_MAX_SIZE=60G  # 增加可用内存
export ASCEND_ENABLE_CACHE=1  # 启用算子缓存
export TORCH_NPU_LAZY_COMPILE=1  # 惰性编译
export ASCEND_CACHE_POLICY=2  # 智能缓存策略
export ASCEND_FUSION_ENABLE=1  # 启用算子融合
export ASCEND_GEMM_DTiling=1  # 启用GEMM融合
export TORCH_NPU_ENABLE_NUMA=1  # 启用NUMA优化
export ASCEND_MEMORY_COPY_MODE=1  # 启用快速内存拷贝
export ASCEND_HBM_ALLOC_TYPE=1  # 优化HBM分配
export ASCEND_OPP_LEVEL=O3  # 最高级算子优化
export ASCEND_FUSION_PASS_ENABLE=1  # 启用融合优化
export ASCEND_GEMM_BTiling=1
export ASCEND_GEMM_ATiling=1
export ASCEND_CONV_ALGO_SELECTION=1  # 启用卷积算法选择
export ASCEND_ENABLE_TRANSFORMER_FUSION=1  # 启用Transformer融合
export ASCEND_MEMORY_REUSE_MODE=2  # 更激进的内存复用
export ASCEND_ENABLE_PREFETCH=1  # 启用数据预取
export ASCEND_NPU_ENABLE_UNIFIED_MEMORY=1  # 启用统一内存
export ASCEND_OPTIMIZER_AGGRESSIVE_MODE=1  # 启用激进优化模式
export ASCEND_SYNCHRONIZATION_MODE=0  # 禁用同步模式，提高并行性能
export PYTORCH_NPU_ENABLE_LARGE_CONCAT=1
export PYTORCH_NPU_ENABLE_TORCHscript=1
export NPU_PERF_MODE=high_performance #开启高性能模式
# 设置精度为 bf16
export NANOCHAT_DTYPE=bfloat16

run_only_on_rank0() {
    if [ -z "$RANK" ] || [ "$RANK" -eq 0 ]; then
        echo "[Rank 0] $*"
        "$@"
    fi
}


if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi


# ===================== 模型tag & 报告重置 =====================
# model_tag=d24_0326
model_tag=d24_0320

# echo -e "\n===== 重置报告 =====\n"
# python -m nanochat.report reset


# ===================== 基础训练&中训练数据集下载 =====================
echo -e "\n===== 基础训练集下载 =====\n"
$PYTHON -m nanochat.dataset -n 8 -d base
$PYTHON -m nanochat.dataset -n 170 -d base &
DATASET_DOWNLOAD_PID=$!
echo -e "\n===== 中训练数据集下载 =====\n"
$PYTHON -m nanochat.dataset -n 30 -d mid_train &
DATASET_DOWNLOAD_PID2=$!


# ===================== tokenizer训练 =====================
echo -e "\n===== tokenizer训练 =====\n"
$PYTHON -m scripts.tok_train
$PYTHON -m scripts.tok_eval
wait $DATASET_DOWNLOAD_PID
wait $DATASET_DOWNLOAD_PID2

# ===================== 基础训练 =====================
echo -e "\n===== 基础模型训练 =====\n"
$PYTHON -m torch.distributed.run --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=9.5 --device-batch-size=8 --run=$WANDB_RUN  --model-tag $model_tag  --core-metric-every=500  --sample-every=500

echo -e "\n===== 基础模型评估 =====\n"
$PYTHON -m torch.distributed.run --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=32  --model-tag $model_tag --model-type=base


# ===================== 中训练 =====================
echo -e "\n===== 模型中训练 =====\n"
$PYTHON -m torch.distributed.run --standalone --nproc_per_node=8 -m scripts.mid_train -- --target-param-data-ratio=0.5 --device-batch-size=8 --run=$WANDB_RUN  --model-tag $model_tag  --core-metric-every=500


echo -e "\n===== 中训练评估 =====\n"
$PYTHON -m torch.distributed.run --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=32 --model-tag $model_tag --model-type=mid  # --max-per-task=-1


# # ===================== sft训练 =====================
# echo -e "\n===== sft训练集下载 =====\n"
# curl -k -L -o ${NANOCHAT_BASE_DIR}/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl || {
#     echo "⚠️  数据下载失败，跳过SFT训练（可手动下载后重试）"
#     exit 0
# }

# echo -e "\n===== sft训练 =====\n"
# $PYTHON -m torch.distributed.run --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=8 --run=$WANDB_RUN --model-tag $model_tag --source=mid

# echo -e "\n===== sft评估 =====\n"
# $PYTHON -m torch.distributed.run --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft -g $model_tag --b=32


# # # ===================== 模型对话 =====================
# echo -e "\n===== 模型对话 =====\n"
# $PYTHON -m scripts.chat_cli -g $model_tag   -p "Why is the sky orange?"
$PYTHON -m scripts.chat_cli -g $model_tag
# $PYTHON -m torch.distributed.run --standalone --nproc_per_node=8 -m scripts.chat_cli -g $model_tag   -p "Why is the sky orange?"


# # web对话（暂不支持）
# # python -m scripts.chat_web


# # ===================== 报告生成 =====================
# echo -e "\n===== 报告生成 =====\n"
# python -m nanochat.report generate