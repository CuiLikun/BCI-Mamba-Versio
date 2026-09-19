# 历史实验归档

[返回项目首页](../README.md) · [新版实验指南](../docs/EXPERIMENTS.md)

本目录保存合并前的课程实验资料，包括原始 README、训练日志、图表和模型权重。合并日期为 **2026-09-19**。

## 来源与代码追溯

| 资料 | 来源 | 对应代码 |
| --- | --- | --- |
| [ATCNet 基线](atcnet/README.original.md) | 原 `CuiLikun/BCI-Motor-Imagery-Project` | [原始提交 25edb65](https://github.com/CuiLikun/BCI-Mamba-Versio/tree/25edb658ac6068f71edfdb04e3426c0eb2ff6c4c) |
| [Mamba 旧实验](mamba/README.original.md) | 本仓库合并前版本 | [原始提交 d2a9330](https://github.com/CuiLikun/BCI-Mamba-Versio/tree/d2a93300cbcaa2abbfe2f4477b40f5a07821bccf) |

原 ATCNet 仓库已删除，其完整 Git 历史保留在本仓库的 [archive/atcnet-baseline 分支](https://github.com/CuiLikun/BCI-Mamba-Versio/tree/archive/atcnet-baseline)。上述代码链接均指向保留仓库。

## 资料导航

| 内容 | ATCNet | Mamba |
| --- | --- | --- |
| 原始介绍与逐受试者成绩 | [README](atcnet/README.original.md) | [README](mamba/README.original.md) |
| 训练与评估日志 | [log.txt](atcnet/results/log.txt) | [log.txt](mamba/results/log.txt) |
| 图表 | [result_picutures](atcnet/results/result_picutures/) | [pictures](mamba/results/pictures/) |
| 模型权重 | [saved models](atcnet/results/saved%20models/) | [saved models](mamba/results/saved%20models/) |

原始文件名及拼写保持不变。`README.original.md` 描述的是当时的项目状态，其中命令、目录和链接可能已过时；当前运行方式以根目录 README 为准。

## 如何理解历史成绩

| 原 README 报告 | 平均 Accuracy | 平均 Kappa |
| --- | ---: | ---: |
| ATCNet | 77.16% | 0.695 |
| Mamba 旧实验 | 73.69% | 0.649 |

这些结果用于保存实验记录，未按新版协议复测。合并前代码中发现的问题包括：

- `main_TrainTest.py` 将测试集用于训练期间验证。
- `main_TrainValTest.py` 在多个训练运行中按测试准确率选择最佳运行。
- 标准化发生在训练 / 验证划分之前。
- Mamba 训练脚本从第 6 位受试者开始迭代。

仅凭现有资料无法确认所有旧权重与日志对应的完整配置。因此，历史表格不能作为严格控制变量后的模型优劣结论。新版时间轴 Mamba 结构也尚无完整数据集成绩。

## 完整性与使用边界

两组 `results/` 与 `README.original.md` 均保留来源提交中的原始字节。[sha256.json](sha256.json) 记录全部 64 个归档文件的 SHA-256；此说明文档不属于该清单。

在仓库根目录执行以下命令可核对清单：

```bash
python -c "import hashlib,json; from pathlib import Path; root=Path('historical'); records=json.loads((root/'sha256.json').read_text(encoding='utf-8')); bad=[name for name,digest in records.items() if not (root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest]; print('FAIL: '+', '.join(bad) if bad else 'OK: '+str(len(records))+' files'); raise SystemExit(bool(bad))"
```

旧 `best models.txt` 中的路径属于原运行环境。旧权重的格式、模型结构与依赖环境也可能不同，不应直接作为新版 `evaluate` 的输入。需要追溯时请查看来源提交；开展新实验时请使用[统一入口](../docs/EXPERIMENTS.md)。
