策略：由于NanoChat在持续更新，因此无法保证实现对每个版本的适配，仅公布特定版本的适配代码。本文会总结如何以最少的修改，完成基于NPU的改造与跑通。

概述：所有适配版本均能在 8 x 910B3 NPU下跑通，环境依赖Ubuntu 22.04.5 LTS、CANN-8.3.RC2、npu-smi 24.1.0.3 

适配总结：
1、NPU不支持FlashAttention-3，降级为Ascend FlashAttention。前向传播速度下降33%～50%，反向传播速度下降 33%～43%         https://arxiv.org/abs/2407.08608
2、修复bug：保存新模型权重时，未清理原有权重。导致权重版本混淆，评估时获得错误得分
3、新增昇腾 NPU 分布式初始化模块、NPU 峰值 FLOPS 计算、修改设备检测优先级、精度使用bf16
4、新增 HuggingFace token 获取函数，解决数据集下载限流问题
5、新增自动调整系统文件描述符上限，避免高并发时"too many open files"错误
6、数据下载速度优化：多进程 → 多线程，全局复用 requests.Session，新增断点续传
7、新增中训练数据处理：包括中训练数据集下载、自动格式转换、基础配比混合（通用语料 + 数学语料按比例流式混合）
8、修复4个优化器相关函数：adamw_step_fused、muon_step_fused函数修改如下：禁用torch.compile装饰器、0-D张量移到NPU（原代码强制CPU，导致设备不匹配）、 修复bfloat16不支持lerp_的问题（临时转float32，兼容NPU算子）；MuonAdamW、DistMuonAdamW函数修改如下：优化器状态（exp_avg/exp_avg_sq）张量初始化时绑定参数设备（NPU）
9、报告生成模块修改：新增NPU支持（获取NPU硬件信息功能、 NPU 成本估算等）、新增中训练报告模块
10、基础/sft模型评估：新增可选中训练/基础模型（model-tag）功能
11、基础模型训练：禁用fp8、禁用torch.compile、修复优化器状态迁移逻辑、新增total_batch_size自动调整到可整除值、张量异步传输数据到NPU
12、新增中训练模块代码
13、tokenizer评估模块修复：修复数据集下载的网络问题

对比总结：
1、NPU不支持FlashAttention-3，需降级为Ascend FlashAttention。前向传播速度下降33%～50%，反向传播速度下降 33%～43%              https://arxiv.org/abs/2407.08608
2、NPU不支持fp8，需降级为fp16，显存占用更多，训练速度下降约 20%              https://developer.nvidia.com/blog/faster-training-throughput-in-fp8-precision-with-nvidia-nemo/
3、NPU默认不支持 torch.compile，模型执行速度下降约 17%～27%              https://pytorch.org/blog/pytorch-2-0-release/
4、NPU部分算子对数据格式要求严格，须进行显式转换
5、效果对比（预训练）：
版本
设备
FP16 算力
预训练耗时
模型基准得分
NanoChat官方
8 x H100 GPU
8 x 989 TFLOPS
1.8h
0.2690
NPU适配版（我们）
8 x 910B3 NPU
8 x 320 TFLOPS
13.8h
0.2668
注：算力对比参考（https://developer.huawei.com/consumer/cn/blog/topic/03202360837318320）