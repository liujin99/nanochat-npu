# NanoChat-NPU

> 基于 [karpathy/nanochat](https://github.com/karpathy/nanochat) 的昇腾 NPU 适配版本

## NPU 适配说明

遵循**最小侵入式适配改造**原则，在昇腾 NPU 上实现完整的 LLM 训练能力。

### 环境

- Ubuntu 22.04.5 LTS
- CANN 8.3.RC2
- NPU-SMI 24.1.0.3
- 8×910B3 NPU

### 性能

| 版本 | 设备 | 预训练耗时 | CORE 得分 |
|------|------|------------|-----------|
| NanoChat 官方 | 8×H100 | 1.8h | 0.2690 |
| NPU 适配版 | 8×910B3 | 13.8h | 0.2668 |

> 详细适配说明见 **[doc.md](doc.md)**

## 新增功能

### 中训练模块（`scripts/mid_train.py`）

原版仅有预训练和 SFT，本项目新增中训练阶段——位于两者之间，使用高质量领域数据进行退火训练。

- 自动继承预训练权重与优化器状态，支持学习率缩放（`--lr-scale`）
- 中训练 checkpoint 独立保存至 `mid_checkpoints/`，与预训练权重隔离
- 新增中训练报告模块（`mid-model-training.md` / `mid-model-evaluation.md`）

### 中训练数据集处理（`nanochat/dataset.py`）

原版仅支持 ClimbMix 单源下载，本项目新增多源数据混合能力：

- 新增 GSM8K、AQUA-RAT 数学推理数据集的下载与格式转换
- 流式混合：70% 通用语料 + 30% 数学语料，输出至 `mid_train_data/`
- 下载优化：多线程 + 连接池复用 + 断点续传（替代原版多进程 + 无续传）

```bash
# 下载并生成中训练混合数据集
python -m nanochat.dataset -n 30 -d mid_train

# 运行中训练（8卡分布式）
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train \
    --target-param-data-ratio=0.5 --device-batch-size=8
```

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/liujin99/nanochat-npu.git
cd nanochat-npu

# 运行训练（已适配 NPU）
bash runs/speedrun.sh
```

其余用法与原项目一致，见 [karpathy/nanochat](https://github.com/karpathy/nanochat)。

## License

MIT