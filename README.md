# 4 步推理：31 token、650 万条、8×A100 80GB

本分支 `four-step-len31-32m-a100` 以原实验 `paper_4step_chain_mixed_batch1000_lr1e4_prenorm_kaiming_gamma1_seed2029` 为基准。实现一个 **3 层、d_m=1024、d_ff=2048** 的 4 步推理模型，输入 **31 token**，固定训练集 **6,500,000 条**，**8 张 A100 80GB 共同训练这个模型**。分支名沿用原名，当前训练条数已更新为 650 万。

原实验配置保存在 [reference_config.json](reference_config.json)。本次明确修改序列长度、训练条数、d_ff、GPU 数和 batch；**保留原词典**（101 个输出类别，节点取值 20–100）、学习率 1e-4 及其余模型/优化超参数。main 分支保留此前独立的 7–13 步实验。

## 启动

在目标 8×A100 80GB 服务器上：

```bash
git clone --branch four-step-len31-32m-a100 https://github.com/qt336/multi-reasoning.git
cd multi-reasoning
python3 -m pip install -r requirements.txt
bash run.sh
```

默认全局 **batch=64,000**，每卡 **8,000**，无梯度累积。每轮完整遍历 650 万条：101 个完整 batch，最后一个 batch 为 36,000 条（每卡 4,500 条），共 **102 次优化更新**；2000 轮总计 **204,000 次更新**，20 轮 warmup 对应 **2,040 次更新**。学习率不随 batch 放大，峰值始终 **1e-4=0.0001**。训练 loss 按实际样本数加权汇总。

这个 batch 是针对当前尺寸的起始配置，**未在目标 A100 上做吞吐实测，不能声称已经达到最快**。不同 batch 的实际速度可用下文脚本比较；较大 batch 会改变优化轨迹，最终准确率需要正式训练评估。

```bash
# 仅生成数据，不训练、不访问 GPU
bash run.sh prepare

# 指定磁盘目录或显式选择另一个全局 batch
DATA_ROOT=/fast_disk/data RUN_ROOT=/fast_disk/runs bash run.sh
GLOBAL_BATCH=32000 bash run.sh

# 如服务器不支持编译环境，可以使用 eager 实现
COMPILE_MODEL=0 bash run.sh
```

默认显示设备为 `0,1,2,3,4,5,6,7`，启动时只读检查可见设备确为 8 张 A100、每张至少 75 GiB 显存容量。全局 batch 必须为正、是 8 的倍数且不超过 6,500,000；最后不足一个 batch 时，8 张卡共同处理剩余样本，不丢弃或补齐样本。`PYTHON` 可指定解释器。

数据目录为 `data/chain_4step_6p5m_len31_vocab101_eval10000`。run 目录包含 d_ff、长度、条数、batch、编译开关和每组测试条数，使用 `6p5m` 区分之前 3200 万条的实验。重复同一命令会复用完整数据集、固定训练评估行号，恢复 `latest.pt` 中的模型、AdamW 状态与已完成 epoch；不兼容配置会被拒绝。更新后默认从头训练 650 万条的新实验，保留旧数据与 checkpoint。

## 吞吐优化与目标机器实测

- 每张卡预载自己的 **812,500 条 uint8 数据**，约 **26 MB**；每轮 GPU 上的完整随机排列约 **6.5 MB**。训练 batch 在 GPU 上直接索引，固定评估子集也缓存到 GPU。
- 仍使用 **bf16 autocast、TF32、fused AdamW**，保留 float32 logits 的交叉熵；大 batch 减少每轮优化器更新与跨卡梯度同步次数。
- DDP 使用 `gradient_as_bucket_view=True` 和 `static_graph=True`，减少梯度复制及重复图分析；仍为每卡一进程、8 卡协同。
- 默认启用 `torch.compile(..., mode="max-autotune-no-cudagraphs", dynamic=False)`，对固定形状进行内核调优与算子融合。首次遇到完整 batch 和末尾小 batch 时各有编译开销；该模式保留 PyTorch 2.5 的 DDP 分图兼容性。模型结构、head 数、归一化和损失不变，编译不保证逐位数值相同。
- `status.json` 记录每轮耗时和 examples/second。训练原始数据总计约 **208 MB**，另有 20,000 条测试数据及 checkpoint 空间。

若优先选择这台 A100 服务器上的实测高吞吐 batch，可先运行：

```bash
python3 benchmark.py
```

**benchmark.py 会执行短暂的合成数据优化更新，应仅由用户在目标 A100 上手动运行。** 默认比较全局 batch **16,000 / 32,000 / 64,000 / 128,000**，每个候选都是 8 卡 DDP，默认排除 5 个 warmup step（含编译），测量其后 20 step。基准只测完整 batch 的吞吐，不包含正式训练末尾小 batch 的编译耗时。报告吞吐、单步耗时和峰值显存，并输出最快候选的启动命令：

```text
GLOBAL_BATCH=<实测最快的候选> bash run.sh
```

每个候选在独立的 torchrun 进程组中运行，OOM 时记录并跳过，其他错误报错停止。测量目录位于 `runs/batch_benchmark_<timestamp>/`，不会读取或覆盖正式实验数据、训练 run 或 checkpoint。合成数据只用于吞吐测量，不报告推理准确率；测量结果只代表列出的候选，不能证明所有可能配置中的全局最优。

可以通过 `--candidates`、`--warmup`、`--measure` 改变测量范围，或用 `--no-compile-model` 对比 eager。运行 `bash run.sh` 不会自动执行该基准脚本。

## 与原实验保持一致的设置

| 项目 | 本分支设置 |
| --- | --- |
| 推理任务 | **4 步** |
| 层数 / d_m / d_ff | **3 / 1024 / 2048**；其中 d_ff 按本次要求修改 |
| 序列长度 | **31**：15 个事实对，共 30 token，最后 1 个查询 token |
| 训练数据 | **6,500,000 条**固定数据；每轮完整遍历 |
| GPU / 默认 batch | **8×A100 80GB / 全局 64,000 / 每卡 8,000** |
| 输出词典 / 数据节点 | 原来的 **101 类，token ID 0–100**；数据节点只使用 **20–100**，不做减 1 映射 |
| 最大 LR / warmup / epochs | **1e-4 / 20 epoch / 2000 epoch**，warmup 后 cosine |
| 优化器 | AdamW，betas=(0.9,0.999)，eps=1e-8，weight decay=0.1，包含所有参数 |
| 归一化 | PreNorm RMSNorm；eps=1e-6；固定 gain=1，无可学习 affine；输出头前 final RMSNorm |
| 初始化 | `kaiming_uniform_relu_gamma1`；线性权重 U(−√6/fan_in,+√6/fan_in)，bias U(−1/fan_in,+1/fan_in) |
| Embedding | token 和可学习绝对位置嵌入均 N(0,1)；无 RoPE |
| Attention / FFN | 单头因果 attention；每层独立 FFN；ReLU；无 dropout、无梯度裁剪 |
| 输出头 | 最后 token 分类；独立线性层，不与输入嵌入共享权重 |
| Train/Test 事实对 | train `(y−x) mod 5 ∈ {0,1,4}`，test 属于 `{2,3}` |
| 随机种子 | 数据 2027、训练 2029、训练评估抽样 2027 |
| 测试与保存 | epoch 0、每 5 轮及最后一轮评估并覆盖 `latest.pt`；eval batch=250 |
| 训练准确率 | 每轮固定 10,000 条实际训练样本；每 GPU 1250 条；不放回抽样 |
| 测试准确率 | canonical 和 noncanonical **各 10,000 条**；每 GPU 每组 1250 条；合计 20,000 条 |
| OMP_NUM_THREADS | 2 |

每条数据的 15 条事实组成一条连续、节点互异的有向链；查询起点在能向后走 4 步的位置中均匀选择，答案为第 4 步终点，事实对顺序随机。原数据生成算法保留，仅将链长由 12 条改为 15 条。完整数据写完后才生成 `dataset.json`；未完成的数据目录不能误当作可复用数据集。

## 准确率与 canonical 序关系

每个 run 输出 `accuracy.csv`、自动更新的 `accuracy.png`、`config.json`、`status.json`、`train.log`、`latest.pt` 和 `train_accuracy_sample_ids.npy`。

训练准确率沿用原来的 **固定 1 万条训练样本抽样**口径，不是对整个 650 万条重评，也不是在线 batch 准确率。测试使用 **canonical 10,000 条 + noncanonical 10,000 条配对样本**，两组事实集、查询和标签一致，只改变事实顺序。每次测试完整评估这两组固定样本，各组准确率的分母均为 10,000，总体测试准确率的分母为 20,000。图同时画训练准确率、平衡测试集总体准确率和两种序关系各自的测试准确率。总体测试准确率是两组的等权平均，不代表随机排列分布的加权准确率；未评估的 epoch 对应测试列留空。

数据目录与运行目录增加 `_eval10000` 后缀，避免复用之前每组 1000 条的测试数据或混合不同评估口径的曲线。

令 F1–F4 为查询所需的四条事实，p_i 为 Fi 在输入中的位置，**canonical 条件保持原实验不变**：

```text
p3 > p2 且 p3 > p4
```

F1 与其余 11 条事实没有额外位置限制。随机排列中 canonical 占比为 1/3。代码通过对四条事实全部 24 个排列，以及有 15 条事实上下文的独立 token 级传播模拟核验这一条件。

## 本机验证（不训练）

```bash
bash validate.sh
```

只运行数据、标签、序关系、模型形状、原超参数对照、日志/绘图、8 卡启动参数和语法检查。启动参数检查用替身程序记录命令，**不执行训练、反向传播、优化器更新或 GPU 访问**。不包含目标 A100 的训练性能/编译实测。本分支没有生成正式实验结果，实际准确率曲线在用户启动训练后产生。
