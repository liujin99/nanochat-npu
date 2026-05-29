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