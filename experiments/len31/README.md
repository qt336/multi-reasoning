# 长度 31 的四步推理实验

两个模型均有 3 层单头因果注意力。输入包含 15 对事实和末尾的查询词元。训练集有 650 万条样本，固定抽取 1 万条记录训练准确率。测试集包含 1 万对配对样本，每对共享事实、查询和答案，顺序分别为 canonical 与 noncanonical。

| 实验 | GPU 数 | 最后一次测试 | 该轮总测试准确率 | 该轮 Canonical | 该轮 Noncanonical | 最后完整 checkpoint |
|---|---:|---:|---:|---:|---:|---:|
| [dm=1300，dff=2600](dm1300/) | 4 | 1925 | 31.34% | 47.64% | 15.04% | 1925 |
| [dm=1600，dff=3200](dm1600/) | 2 | 360 | 10.405% | 10.09% | 10.72% | 355 |

**2026-11-12 更新**：650 万条 dm1300 已提前停止，训练评估至第 1928 轮，最后 checkpoint 为第 1925 轮；canonical 历史最高 66.90%（第 1825 轮）。dm1600 行仍是此前归档的历史快照，不表示当前续训状态。

两组均使用训练 seed 2031、数据 seed 2027、全局 batch 4000、FP32、AdamW、峰值学习率 0.0001、weight decay 0.15。前 20 轮 warmup，随后按原定 2000 轮计划做余弦衰减。准确率表保留了所有完成的评估，测试 loss 表仅列出实际计算的轮次。

[下载历史 14 个 checkpoint 和 4 份热图归档](https://github.com/qt336/multi-reasoning/releases/tag/len31-four-step-seed2031) · [SHA-256 校验清单](SHA256SUMS)

[新增：dm1300 第 1925 轮 checkpoint 与完整结果](https://github.com/qt336/multi-reasoning/releases/tag/dm1300-len31-6p5m-stopped-20261112)

## 复现

在 Linux 和 CUDA 环境下进入这个目录，选择一个实验运行。dm1300 使用 4 张可见 GPU，dm1600 使用 2 张。可用 `CUDA_VISIBLE_DEVICES` 指定设备。

```bash
python3 -m pip install -r dm1300/source_at_launch/requirements.txt
bash dm1300/train.sh
# 另一个实验
bash dm1600/train.sh
```

脚本按原种子生成数据，默认在两个实验共用的 `data/` 目录中保存。每个实验的输出写入自己的 `runs/reproduction/`。通过 `DATA_DIR` 和 `RUN_DIR` 可指定其他位置。若要续训，下载与模型匹配的 checkpoint，以 `latest.pt` 存入新的 `RUN_DIR` 后执行同一启动脚本。历史评估保存在各自的 `results/` 目录。可用 `results/dataset.sha256` 校验重新生成的数据。

下载单份权重后可用 `sha256sum 文件名.pt` 与该实验的 `checkpoints.csv` 核对。每个 Release 附带的 `SHA256SUMS` 只校验该次发布的附件；本目录的清单汇总了历史附件和新增的第 1925 轮权重。下载对应清单中的文件后，可运行 `sha256sum -c SHA256SUMS`。
