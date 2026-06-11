# NanoChat-NPU 适配文档

> **策略**：由于 NanoChat 在持续更新，无法保证实现对每个版本的适配，仅公布特定版本的适配代码。本文总结如何以最少的修改，完成基于 NPU 的改造与跑通。
>
> **概述**：所有适配版本均能在 8×910B3 NPU 下跑通，环境依赖 Ubuntu 22.04.5 LTS、CANN 8.3.RC2、npu-smi 24.1.0.3

---

## 适配总结

### 核心适配

| # | 适配内容 | 说明 |
|---|---------|------|
| 1 | **FlashAttention 降级** | NPU 不支持 FA-3，降级为 [Ascend FlashAttention](https://arxiv.org/abs/2407.08608)。前向传播速度下降 33%～50%，反向传播速度下降 33%～43% |
| 2 | **NPU 分布式初始化** | 新增昇腾 NPU 分布式初始化模块、NPU 峰值 FLOPS 计算、修改设备检测优先级、精度使用 bf16 |
| 3 | **基础模型训练适配** | 禁用 fp8、禁用 `torch.compile`、修复优化器状态迁移逻辑、新增 `total_batch_size` 自动调整到可整除值、张量异步传输数据到 NPU |
| 4 | **新增中训练模块** | 详见下方 [中训练模块](#中训练模块) 章节 |

### 优化器修复

修复 4 个优化器相关函数：

- **`adamw_step_fused` / `muon_step_fused`**：
  - 禁用 `torch.compile` 装饰器
  - 0-D 张量移到 NPU（原代码强制 CPU，导致设备不匹配）
  - 修复 bfloat16 不支持 `lerp_` 的问题（临时转 float32，兼容 NPU 算子）
- **`MuonAdamW` / `DistMuonAdamW`**：
  - 优化器状态（`exp_avg` / `exp_avg_sq`）张量初始化时绑定参数设备（NPU）

### 数据与工具

| # | 适配内容 | 说明 |
|---|---------|------|
| 5 | **HuggingFace Token 支持** | 新增 token 获取函数，解决数据集下载限流问题 |
| 6 | **文件描述符优化** | 新增自动调整系统文件描述符上限，避免高并发时 "too many open files" 错误 |
| 7 | **数据下载优化** | 多进程 → 多线程，全局复用 `requests.Session`，新增断点续传 |
| 8 | **中训练数据处理** | 详见下方 [中训练数据集处理](#中训练数据集处理) 章节 |
| 9 | **Tokenizer 评估修复** | 修复数据集下载的网络问题 |

### 评估与报告

| # | 适配内容 | 说明 |
|---|---------|------|
| 10 | **报告生成模块** | 新增 NPU 支持（获取 NPU 硬件信息、NPU 成本估算等）、新增中训练报告模块 |
| 11 | **模型评估增强** | 基础/SFT 模型评估新增可选中训练/基础模型（`model-tag`）功能 |

### Bug 修复

- 修复保存新模型权重时未清理原有权重的问题，避免权重版本混淆导致评估得分错误

---

## 新增功能详解

### 中训练模块

> **原版对比**：karpathy/nanochat 官方仓库仅包含预训练（`base_train.py`）和 SFT（`chat_sft.py`），**不存在中训练脚本**。本项目新增完整的 `scripts/mid_train.py`（438 行），填补预训练与 SFT 之间的能力空白。

**中训练（Mid Training）** 是位于预训练和 SFT 之间的退火训练阶段，使用高质量领域数据（如数学推理语料）对预训练模型进行进一步训练，以提升模型在特定领域的能力。

#### 核心设计

**1. 预训练状态继承**
- 从预训练 checkpoint 加载模型权重和优化器状态（`load_optimizer_state`）
- 自动继承预训练超参数：`max_seq_len`、`device_batch_size`、`total_batch_size`、各级学习率（`embedding_lr`、`unembedding_lr`、`matrix_lr`）
- 支持通过 `--lr-scale` 参数缩放预训练学习率，实现平滑接续或探索更优起点

**2. 训练调度策略**
- **迭代次数计算**：支持三种模式——手动指定（`--num-iterations`）、目标 FLOPs（`--target-flops`）、参数-数据比例（`--target-param-data-ratio`，类 Chinchilla 缩放）
- **学习率调度**：warmup → 恒定 → warmdown 三段式，`warmdown_ratio=0.9` 确保充分退火
- **权重衰减**：cosine 衰减策略，随训练进度平滑下降
- **Muon 动量**：前 300 步从 0.85 线性升温至 0.95

**3. 评估与监控**
- 验证集 BPB（bits-per-byte）评估，每 `--eval-every` 步执行
- CORE 基准指标评估，每 `--core-metric-every` 步执行
- 实时采样（`--sample-every`）观察模型生成质量
- WandB 日志集成，记录 loss、MFU、tok/sec 等训练指标
- 训练结束后生成中训练报告（`mid-model-training.md`）

**4. Checkpoint 管理**
- 中训练权重独立保存至 `mid_checkpoints/` 目录，与预训练权重隔离
- 支持通过 `--model-tag` 指定加载特定版本的预训练模型
- 保存完整元信息（模型配置、用户配置、batch size 等）供后续阶段继承

#### 使用方式

```bash
# 下载中训练数据集（详见下方数据集处理章节）
python -m nanochat.dataset -n 30 -d mid_train

# 启动中训练（8卡分布式）
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train \
    --target-param-data-ratio=0.5 \
    --device-batch-size=8 \
    --core-metric-every=500
```

---

### 中训练数据集处理

> **原版对比**：karpathy/nanochat 官方 `dataset.py` 仅支持 ClimbMix 预训练数据的单源下载，使用 `multiprocessing.Pool` 多进程下载，无断点续传，无数据混合格能。本项目对 `nanochat/dataset.py` 进行了大幅重写与扩展。

#### 多源数据集支持

| 数据集 | 类型 | 说明 |
|--------|------|------|
| **ClimbMix-400B** | 通用语料 | 从 HuggingFace 下载，取最后 20 个 shard 作为中训练通用语料 |
| **GSM8K** | 数学推理 | OpenAI 小学数学应用题，含完整解题步骤 |
| **AQUA-RAT** | 数学推理 | DeepMind 代数数学题，含推理过程（rationale）和正确选项 |

#### 格式转换

原版数据集仅有 `text` 列，而 GSM8K 和 AQUA-RAT 的原始格式为 `question`/`answer`/`options` 等字段。新增两个转换函数：

- **`convert_gsm8k_to_text`**：将 `question` + `answer` 拼接为 `Question: ...\nAnswer: ...` 格式
- **`convert_aqua_rat_to_text`**：将 `question` + `options` + `rationale` + `correct` 拼接为结构化文本，保留完整推理链

#### 流式混合策略

新增 `stream_mix` 函数实现通用语料与数学语料的**按比例流式混合**：

- **混合比例**：70% 通用语料（ClimbMix）+ 30% 数学语料（GSM8K + AQUA-RAT）
- **无限循环生成器**：`endless_generator` 包装流式读取器，数据耗尽时自动循环，避免短数据集提前终止
- **均匀采样**：`stream_texts_uniform` 从多个 parquet 文件的 row group 中随机均匀采样，避免数据偏斜
- **输出格式**：混合后写入 `mid_train_data/mixed_XXXX.parquet`，每文件 10000 条记录

#### 下载优化（对比原版改进）

| 改进项 | 原版 | 本项目 |
|--------|------|--------|
| 并发模型 | `multiprocessing.Pool`（多进程） | `ThreadPool`（多线程，IO 密集型更高效） |
| HTTP 会话 | 每次请求新建连接 | 全局复用 `requests.Session`，连接池大小 64 |
| 重试策略 | 手动指数退避，最多 5 次 | `urllib3.Retry`，自动重试 429/500/502/503/504 |
| 断点续传 | 不支持，失败后删除临时文件 | 支持，通过 `Range` 头从断点处继续下载 |
| 文件描述符 | 无处理 | 自动调整 `RLIMIT_NOFILE` 上限至 4096 |

#### Dataloader 适配

`nanochat/dataloader.py` 中的 `_document_batches` 和 `tokenizing_distributed_data_loader_with_state_bos_bestfit` 新增 `data_dir` 参数，使中训练可使用独立的数据目录（`mid_train_data/`），与预训练数据（`base_data_climbmix/`）完全隔离。

#### 使用方式

```bash
# 下载并生成中训练混合数据集（30 个 shard）
python -m nanochat.dataset -n 30 -d mid_train

# 输出目录结构
# mid_train_data/
# ├── mixed_0000.parquet
# ├── mixed_0001.parquet
# └── ...
```

---

## 性能对比

### 性能下降因素

| 因素 | 影响 | 参考 |
|------|------|------|
| NPU 不支持 FlashAttention-3 | 前向传播速度下降 33%～50%，反向传播速度下降 33%～43% | [Ascend FA 论文](https://arxiv.org/abs/2407.08608) |
| NPU 不支持 fp8，降级为 fp16 | 显存占用更多，训练速度下降约 20% | [NVIDIA FP8 博客](https://developer.nvidia.com/blog/faster-training-throughput-in-fp8-precision-with-nvidia-nemo/) |
| NPU 默认不支持 `torch.compile` | 模型执行速度下降约 17%～27% | [PyTorch 2.0 博客](https://pytorch.org/blog/pytorch-2-0-release/) |
| NPU 部分算子对数据格式要求严格 | 须进行显式格式转换 | — |

### 预训练效果对比

| 版本 | 设备 | FP16 算力 | 预训练耗时 | 模型基准得分 |
|------|------|-----------|-----------|-------------|
| NanoChat 官方 | 8×H100 GPU | 8×989 TFLOPS | 1.8h | **0.2690** |
| NPU 适配版 | 8×910B3 NPU | 8×320 TFLOPS | 13.8h | **0.2668** |

> 算力对比参考：[华为昇腾 vs NVIDIA 算力对比](https://developer.huawei.com/consumer/cn/blog/topic/03202360837318320)
