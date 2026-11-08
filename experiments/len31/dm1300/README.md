# dm=1300，dff=2600

3 层单头 Transformer，序列长度 31。原计划训练 2000 轮，准确率记录至第 1576 轮，最后一份完整 checkpoint 为第 1575 轮。

| 评估口径 | 轮次 | 固定训练样本 | 总测试集 | Canonical | Noncanonical |
|---|---:|---:|---:|---:|---:|
| 最后完整 checkpoint | 1575 | 99.93% | 33.835% | 57.45% | 10.22% |
| 最后一次测试 | 1575 | 99.93% | 33.835% | 57.45% | 10.22% |
| Canonical 最高的一次 | 1350 | 99.82% | 34.83% | 60.02% | 9.64% |

![训练与测试准确率](results/accuracy.png)

[逐轮准确率和训练 loss](results/accuracy.csv) · [已测测试 loss](results/test_loss.csv) · [训练日志](results/train.log) · [完整配置](results/config.json)

## 已保存的 checkpoint

| 轮次 | 原始 PyTorch 文件 | 大小 |
|---:|---|---:|
| 30 | [dm1300-epoch-000030.pt](https://github.com/qt336/multi-reasoning/releases/download/len31-four-step-seed2031/dm1300-epoch-000030.pt) | 468.0 MiB |
| 31 | [dm1300-epoch-000031.pt](https://github.com/qt336/multi-reasoning/releases/download/len31-four-step-seed2031/dm1300-epoch-000031.pt) | 468.0 MiB |
| 32 | [dm1300-epoch-000032.pt](https://github.com/qt336/multi-reasoning/releases/download/len31-four-step-seed2031/dm1300-epoch-000032.pt) | 468.0 MiB |
| 33 | [dm1300-epoch-000033.pt](https://github.com/qt336/multi-reasoning/releases/download/len31-four-step-seed2031/dm1300-epoch-000033.pt) | 468.0 MiB |
| 35 | [dm1300-epoch-000035.pt](https://github.com/qt336/multi-reasoning/releases/download/len31-four-step-seed2031/dm1300-epoch-000035.pt) | 468.0 MiB |
| 380 | [dm1300-epoch-000380.pt](https://github.com/qt336/multi-reasoning/releases/download/len31-four-step-seed2031/dm1300-epoch-000380.pt) | 468.0 MiB |
| 1575 | [dm1300-epoch-001575.pt](https://github.com/qt336/multi-reasoning/releases/download/len31-four-step-seed2031/dm1300-epoch-001575.pt) | 468.0 MiB |

第 380 轮为迁移时保存的 checkpoint。第 1576 轮保留了训练评估。 每个下载包含 epoch、模型参数、优化器状态和训练配置。[文件大小与 SHA-256](checkpoints.csv)

## 注意力与评估

[Canonical 单序列热图](diagnostics/latest_attention/attention_single_can_all.png) · [Noncanonical 单序列热图](diagnostics/latest_attention/attention_single_noncan_all.png) · [历次热图索引](diagnostics/attention_index.csv) · [测试 loss 原始记录](diagnostics/test_loss_history.json)

热图索引中的 `archive` 列对应 Release 附件名。解压后按 `attention/<snapshot>/` 查找同一轮的图片与指标。`initial` 和 `final` 分别对应迁移前后的保存位置，两份归档在交接处可能有相同轮次。

[自然逻辑顺序测试（第 1575 轮）](diagnostics/supplementary/dm1300_natural_order_epoch1575_20261107/result.json) · [值匹配分析（第 105 轮）](diagnostics/supplementary/value_matching_dm1300_20261028_1234/metrics.json) · [值匹配分析（第 145 轮）](diagnostics/supplementary/value_matching_dm1300_20261028_202611/metrics.json)

[训练源代码](source_at_launch/) · [可移植启动脚本](train.sh) · [原始训练机启动脚本](original_launch/)
