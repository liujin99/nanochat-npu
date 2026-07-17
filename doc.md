# NanoChat-NPU 适配文档

> **策略**：由于 NanoChat 在持续更新，无法保证实现对每个版本的适配，仅公布特定版本的适配代码。本文总结如何以最少的修改，完成基于 NPU 的改造与跑通。
>
> **概述**：所有适配版本均能在 8×910B3 NPU 下跑通，环境依赖 Ubuntu 22.04.5 LTS、CANN 8.3.RC2、npu-smi 24.1.0.3

---

## 适配总结

| # | 适配内容 | 说明 |
|---|---------|------|
| 1 | **FlashAttention 降级** | NPU 不支持 FA-3，降级为 [Ascend FlashAttention](https://arxiv.org/abs/2407.08608)。前向传播速度下降 33%～50%，反向传播速度下降 33%～43% |
| 2 | **NPU 分布式初始化** | 新增昇腾 NPU 分布式初始化模块、NPU 峰值 FLOPS 计算、修改设备检测优先级、精度使用 bf16 |
| 3 | **基础模型训练适配** | 禁用 fp8、禁用 `torch.compile`、修复优化器状态迁移逻辑、新增 `total_batch_size` 自动调整到可整除值、张量异步传输数据到 NPU |
| 4 | **优化器修复** | `adamw_step_fused` / `muon_step_fused`：禁用 `torch.compile` 装饰器、0-D 张量移到 NPU（原代码强制 CPU）、修复 bfloat16 不支持 `lerp_` 的问题（临时转 float32）。`MuonAdamW` / `DistMuonAdamW`：优化器状态张量初始化时绑定 NPU 设备 |
| 5 | **HuggingFace Token 支持** | 新增 token 获取函数，解决数据集下载限流问题 |
| 6 | **文件描述符优化** | 自动调整系统文件描述符上限至 4096，避免高并发时 "too many open files" 错误 |
| 7 | **数据下载优化** | 多进程 → 多线程，全局复用 `requests.Session`（连接池 64），新增断点续传，替代原版每次新建连接 + 无续传 |
| 8 | **中训练数据集处理** | 详见下方 [中训练数据集处理](#中训练数据集处理) 章节 |
| 9 | **Tokenizer 评估修复** | 修复数据集下载的网络问题 |
| 10 | **报告生成模块** | 新增 NPU 硬件信息获取与成本估算、新增中训练报告模块（`mid-model-training.md` / `mid-model-evaluation.md`） |
| 11 | **模型评估增强** | 基础/SFT 模型评估新增可选中训练/基础模型（`model-tag`）功能 |
| 12 | **新增中训练模块** | 详见下方 [中训练模块](#中训练模块) 章节 |
| 13 | **Bug 修复** | 修复保存新模型权重时未清理原有权重的问题，避免权重版本混淆导致评估得分错误 |
| 14 | **STEM 评测基准集** | 详见下方 [STEM 评测基准集](#stem-评测基准集) 章节 |

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
- **重要说明**：预训练默认 `warmdown_ratio=0.0`（恒定 LR），退火仅在 mid_train 发生。这是 2024-2025 业界主流做法（LLaMA 3、MiniCPM、OLMo 2 等）。**如果取消中训练阶段、仅做预训练**，需手动传 `--warmdown-ratio=0.65` 让 LR 收敛，否则模型不会自动退火。

**3. Checkpoint 管理**
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

> **原版对比**：karpathy/nanochat 官方 `dataset.py` 仅支持 ClimbMix 预训练数据的单源下载，使用 `multiprocessing.Pool` 多进程下载，无断点续传，无数据混合能力。本项目对 `nanochat/dataset.py` 进行了大幅重写与扩展。

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

### STEM 评测基准集

> **原版对比**：karpathy/nanochat 官方仅支持 DCLM CORE 22 个基准的固定评测，无选择性评测、无生成式任务评测。本项目新增可配置的 STEM 基准集评测功能。

#### 设计目标

训练过程中频繁跑全量 22 个 DCLM core 基准耗时过长，新增 `--eval-benchmarks` 参数支持选择性评测，快速验证模型在 STEM 相关能力上的进展。

#### `--eval-benchmarks` 参数

| 值 | 说明 |
|----|------|
| `all`（默认） | 测全部 core + STEM 基准 |
| `core` | 只测 DCLM 22 个 core 基准 |
| `stem` | 只测 STEM 6 个基准 |
| 逗号分隔 | 指定具体基准标签（如 `arc_challenge,gsm8k_cot`） |

#### STEM 基准集（6个）

| 基准 | 来源 | 类型 | shot数 | 说明 |
|------|------|------|--------|------|
| `arc_easy` | core.yaml | 多选题 | 10-shot | ARC Easy 科学推理 |
| `arc_challenge` | core.yaml | 多选题 | 10-shot | ARC Challenge 科学推理 |
| `mmlu_stem` | eval_stem (HF) | 多选题 | 0-shot | MMLU STEM 子集（22学科，3545题） |
| `gpqa_diamond` | eval_stem (HF) | 多选题 | 0-shot | GPQA Diamond 博士级科学QA（198题） |
| `gsm8k_cot` | eval_stem (HF) | 生成式 | 8-shot | GSM8K 数学应用题（1319题） |
| `math_cot` | eval_stem (HF) | 生成式 | 4-shot | MATH-500 数学竞赛（500题） |

#### MMLU STEM 子集学科（22个，3545题）

| 学科 | 题数 |
|------|------|
| abstract_algebra | 100 |
| anatomy | 135 |
| astronomy | 152 |
| college_biology | 144 |
| college_chemistry | 100 |
| college_computer_science | 100 |
| college_mathematics | 100 |
| college_physics | 102 |
| computer_security | 100 |
| conceptual_physics | 235 |
| electrical_engineering | 145 |
| elementary_mathematics | 378 |
| formal_logic | 126 |
| high_school_biology | 310 |
| high_school_chemistry | 203 |
| high_school_computer_science | 100 |
| high_school_mathematics | 270 |
| high_school_physics | 151 |
| high_school_statistics | 216 |
| machine_learning | 112 |
| medical_genetics | 100 |
| virology | 166 |

#### 评测流程

**多选题评测**（arc, mmlu_stem, gpqa）：ICL few-shot prompt → 模型前向传播 → 各选项 loss 对比 → 选择最小 loss 选项 → 判断正确性

**生成式评测**（gsm8k_cot, math_cot）：ICL few-shot prompt → 自回归生成 → 正则提取答案 → 与 gold 对比

- GSM8K：提取 `#### N` / `The answer is N` / 最后一个数字
- MATH：提取 `\boxed{}` / `The answer is`
- 数值比较：GSM8K 用 float 近似（tol=1e-6），MATH 用字符串归一化 + float 比较

#### Metric 计算

- **core_metric**：仅当跑全 22 个 DCLM core 任务时计算，否则为 None
- **stem_metric**：STEM_BENCHMARK_LABELS 中已评测基准的 centered 平均值
- **centered**：`(accuracy - random_baseline) / (1 - random_baseline)`

#### 数据管理

- DCLM core 数据：`~/.cache/nanochat/eval_bundle/`（从 karpathy S3 下载 eval_bundle.zip，只读）
- STEM 数据：`~/.cache/nanochat/eval_stem/`（从 HuggingFace 下载 eval_stem.zip，3 MB）
- 优先下载预处理打包数据（eval_stem.zip），下载失败时回退到逐个从 HuggingFace 下载并转换
- 首次运行自动下载，后续缓存复用
- 生成式任务 prompt 超长时自动从尾部逐个删减 few-shot 示例，确保不超过 `max_seq_len - max_gen_tokens`

#### 可选额外基准

| 基准 | 说明 | 用法 |
|------|------|------|
| `mmlu_zeroshot` | MMLU 全量 57 学科（0-shot） | `--eval-benchmarks mmlu_zeroshot` |

---

## 性能对比

| 版本 | 设备 | FP16 算力 | 预训练耗时 | 模型基准得分 |
|------|------|-----------|-----------|-------------|
| NanoChat 官方 | 8×H100 GPU | 8×989 TFLOPS | 1.8h | **0.2690** |
| NPU 适配版 | 8×910B3 NPU | 8×320 TFLOPS | 13.8h | **0.2668** |

> 算力对比参考：[华为昇腾 vs NVIDIA 算力对比](https://developer.huawei.com/consumer/cn/blog/topic/03202360837318320)
