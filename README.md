# 8×A100：7–13 步推理，4 层 Transformer

分别训练 **7、8、9、10、11、12、13 步共 7 个独立模型**。**8 张卡共同训练一个模型，完成后才开始下一个**，不是每张卡独立运行不同任务。

每个任务使用固定 **200,000,000 条训练样本**，模型 **4 层、单头、d_m=2048、d_ff=4096**，输入 **53 token**，词典 **1–200（恰好 200 个类别）**。最大学习率明确为 **1e-4 = 0.0001 = 10⁻⁴**，不随 GPU 数或 batch 放大；沿用 20 epoch warmup 和随后 cosine。

参考原实验 `paper_4step_chain_mixed_batch1000_lr1e4_prenorm_kaiming_gamma1_seed2029`，原配置存于 [reference_config.json](reference_config.json)。按本次要求修改层数、宽度、任务、数据量、长度、词典、GPU 数与 batch，其余模型和优化超参数保持原设置。

## 在目标 A100 服务器启动

```bash
python3 -m pip install -r requirements.txt
bash run.sh all             # 8 卡依次训练 7、8、9、10、11、12、13 步
bash run.sh 13              # 8 卡只训练 13 步
bash run.sh all prepare     # 只准备数据，不训练、不访问 GPU
```

默认 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`，固定 8 个 NCCL/DDP 进程。启动时检查可见设备必须为 8 张 A100，读取显存容量；**检测过程不执行试训练或 batch 扫描**。

| 自动检测到的硬件 | 全局 batch | 每 GPU batch | 每 epoch 优化步数 |
| --- | ---: | ---: | ---: |
| 8×A100 40GB | 16,000 | 2,000 | 12,500 |
| 8×A100 80GB | 32,000 | 4,000 | 6,250 |

混合显存配置按容量最小的一张卡选择。两个 batch 都整除 2 亿，每轮无丢样本、无补齐、无梯度累积。这个选择是**容量估算下的起始配置，未在目标 A100 上做吞吐实测，不能声称是最优 batch 或保证特定加速倍数**。增加 batch 减少每轮优化器更新和梯度同步次数；显存占满不等于吞吐最高，大 batch 也会改变优化轨迹。学习率仍严格使用 1e-4。

如需要调整，可显式覆盖全局 batch，必须为 8 的倍数且整除 200,000,000：

```bash
GLOBAL_BATCH=16000 bash run.sh all
DATA_ROOT=/fast_disk/data RUN_ROOT=/fast_disk/runs bash run.sh all
python3 hardware.py  # 只打印设备、batch 与训练计划，不训练
```

`PYTHON` 可指定解释器。run 目录包含步数、词典与 batch，避免覆盖其他实验。重复原命令会复用完整数据集与固定评估样本，从 `latest.pt` 恢复模型、AdamW 状态、epoch 和学习率计划。配置变化时拒绝错误恢复。若一次运行中断，重新 `bash run.sh all` 会检查已完成模型，然后继续后续模型。

## 吞吐与数据存储

- 数据按 50,000 条一块生成，存为 uint8；每个任务原始训练数组约 **10.8 GB**，七个约 **75.6 GB**，另需 checkpoint 空间。
- 每张卡仅预载自己的 **25,000,000 条**训练样本，约 **1.35 GB**，加上约 0.20 GB 的本轮随机排列。每步在 GPU 上索引样本，减少 CPU、磁盘和 PCIe 的反复搬运。
- 数据在生成后固定，**每轮完整遍历 2 亿条一次**；GPU 上每轮重新随机排列 shard，种子仍为 `2029 + epoch×10007 + rank`。
- 使用 bf16 autocast、TF32、fused AdamW；DDP 开启 `gradient_as_bucket_view=True` 和 `static_graph=True`，减少梯度复制及重复图分析；不改变单头注意力结构、损失或优化器设置。
- `status.json` 记录每轮耗时和实际 examples/second，便于在目标机器判断吞吐。手工运行 `train.py --data-residency mmap` 可改为从磁盘映射取 batch；默认启动脚本使用 GPU 常驻 shard。

A100 容量与性能背景参考 [NVIDIA 官方说明](https://developer.nvidia.com/blog/supercharging-worlds-fastest-ai-supercomputing-platform-on-hgx-a100-80gb-gpus/)，DDP 与数据移动优化参考 [PyTorch 性能指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)。这里的 batch 数值是本实验的选择，不是这些文档对本模型的测量结果。

数据生成中断后的未完成目录应更换目录后重试；`dataset.json` 是完成标记。数据 manifest 含词典范围与格式版本，拒绝复用旧词典数据。

## 模型和训练配置

| 项目 | 设置 |
| --- | --- |
| 层数 / d_m / d_ff | **4 / 2048 / 4096** |
| 每个任务训练集 | **200,000,000 条** |
| 任务 | **7–13 步分别训练，顺序执行** |
| 输入长度 | **53**：26 条事实对 + 1 个查询 token |
| 词典 | **1–200**；内部 embedding/分类标签映射为 0–199；无额外 0 类 |
| GPU / batch | **8 卡 DDP；自动选择全局 16,000 或 32,000** |
| 学习率 | **1e-4**；20 epoch 线性 warmup，随后 cosine |
| 训练轮数 | **2000**；每个模型独立训练、从头初始化 |
| 优化器 | fused AdamW，betas=(0.9, 0.999)，eps=1e-8，weight decay=0.1，作用于所有参数 |
| 归一化 | PreNorm RMSNorm，eps=1e-6，无可学习 gain；分类头前最终 RMSNorm |
| 初始化 | `kaiming_uniform_relu_gamma1`：线性权重 U(−√6/fan_in, +√6/fan_in)，bias U(−1/fan_in, +1/fan_in) |
| 嵌入 | token 与绝对位置嵌入均 N(0,1)；无 RoPE |
| FFN / attention | ReLU；每层独立 FFN；单头 causal attention；无 dropout、无梯度裁剪 |
| 输出 | 最后 token 处独立线性分类头，200 类，不共享输入嵌入权重 |
| 事实对划分 | train `(y−x) mod 5 ∈ {0,1,4}`，test 属于 `{2,3}` |
| 随机种子 | 数据 2027、训练 2029、训练准确率抽样 2027 |
| 精度 | CUDA bfloat16 autocast，cross entropy 用 float32 logits，允许 TF32 |
| 评估与保存 | 每轮训练准确率；epoch 0、每 5 轮及最终轮测试；测试轮覆盖 latest.pt |
| 线程 | OMP_NUM_THREADS=2 |

全局 batch 16,000 时，每个模型共 25,000,000 次更新，warmup 250,000 次；batch 32,000 时分别为 12,500,000 和 125,000 次。这里保持 **epoch 数和 warmup epoch 数**，不会固定旧 batch 对应的优化步数。

每条样本是一条节点互异的完整 26 边有向链，查询起点在能向后走 k 步的位置中均匀选择，标签为第 k 步终点。26 个事实对随机排列；每个任务独立训练，不需要额外任务 token。

## 准确率与曲线

每个 run 输出 `accuracy.csv`、自动更新的 `accuracy.png`、`config.json`、`status.json`、`train.log` 和 `latest.pt`。13 步额外画 canonical 诊断面板。

**训练准确率沿用原实验固定抽样口径**：从实际训练集均匀地按 8 个 shard 固定抽取总共 **10,000 条**（每卡 1250），每轮评估；不是全 2 亿条重评，也不是在线 batch 准确率。保存原始行号到 `train_accuracy_sample_ids.npy`。

13 步另从实际训练集的全部 canonical 行中，均匀、不放回抽取固定 **10,000 条 canonical 训练样本**，每轮记录 `train_canonical_accuracy`，保存 `canonical_train_sample_ids.npy`。采样使用独立 RNG，不影响训练数据；不是另外生成的诊断数据。全训练集 canonical 行数写入 manifest，预期约 823,045 条。

主测试集每个任务为 **2,000 条随机顺序的固定 held-out 样本**。13 步另保留 **canonical 1,000 条 + noncanonical 1,000 条配对诊断集**，两组共享事实集、查询和答案，仅排列不同；分别记录 `test_canonical_accuracy` 和 `test_noncanonical_accuracy`，不与主测试集混合计算。7 步在该 4 层传播规则下全部可达，因此不强行定义 noncanonical 组。

所有指标同时记录 correct/n；没有执行测试的 epoch 测试列留空。可以单独重画已有日志：

```bash
python3 plot.py /path/to/run/accuracy.csv /path/to/run/accuracy.png
```

## 13 步 canonical 序关系

F_i 表示查询链上第 i 条事实 `v_(i−1) → v_i`，p_i 表示它在输入中的事实槽位；**i 是推理顺序，不是输入位置，也不是 token ID 的数值大小**。采用：

```text
p3  > max(p2,  p4)
p6  > max(p5,  p7)
p9  > max(p8,  p10)
p12 > max(p11, p13)
p9  > max(p6,  p12)
```

F1 不受位置约束，查询末尾固定；查询链外的 13 条事实位置任意。上述关系是偏序，没有额外要求各分支连续出现或整个分支先后排列。例如所需事实按 `F1,F2,F4,F3,F5,F7,F6,F11,F13,F12,F8,F10,F9` 出现满足条件，事实之间可插入其余事实。

它是原来三层四步条件 `p3 > p2 且 p3 > p4` 的递归扩展：

1. 第 1 层在事实对内完成配对，查询仍持有 v0。
2. 第 2 层查询取得 v1；F3 汇合 F2–F4，F6 汇合 F5–F7，F9 汇合 F8–F10，F12 汇合 F11–F13。
3. 第 3 层查询从 F3 取得 v4；F9 从 F6、F12 汇合 F5–F13 的九条事实，持有 v4–v13。
4. 第 4 层查询借助共同值 v4 从 F9 取得 v13。

在随机排列中，F2–F4 的偏序概率为 1/3，F5–F13 的递归偏序概率为 1/81，因此总概率为 **1/243**。独立的 token 级同步集合传播模拟器核验了这一条件，并枚举全部 2^13 个子集精确计算合法排列数。该条件描述论文传播规则下的信息可达性，不预先保证训练出的神经网络准确率。

## 检查（不训练）

```bash
bash validate.sh
```

只做数据/标签/词典边界、canonical 独立传播与组合计数、8 卡 batch 选择、模型形状和语法检查；不调用反向传播或优化器，不访问 GPU。完整训练仅由用户在 A100 服务器执行 `run.sh` 启动。本仓库没有 2 亿条实验的实测准确率结果。
