# EEG ATCNet + Mamba：运动想象脑电分类

基于 TensorFlow/Keras 的 BCI Competition IV-2a 课程与探索性实验项目：复现 **ATCNet**，并比较以 **Mamba 风格状态空间模块**替换注意力模块的变体。

本仓库合并了原 `BCI-Motor-Imagery-Project` 的基线资料。ATCNet 原始工作属于 Hamdi Altaheri 等作者；本项目提供实验整理、Mamba 变体和统一评估入口，不将原论文模型作为原创成果。

## 模型与实验状态

| 命令行模型 | 模块 | 扫描维度 | 状态 |
| --- | --- | --- | --- |
| `atcnet` | 多头自注意力 + TCN | 时间序列 | 基线 |
| `mamba` | Mamba 风格模块 + TCN | 时间轴，输入 `(batch, time, features)` | 新版结构，尚无完整数据集成绩 |
| `mamba-channel` | Mamba 风格模块 + TCN | 卷积特征轴 | 保留旧实现的轴语义 |

Mamba 模块使用 TensorFlow `tf.scan`，没有官方融合 CUDA 内核，使用 `same` 卷积，不是严格因果流式模型。此处研究的是离线试次分类，不宣称在线推理能力或相对 ATCNet 的性能提升。时间轴版与旧版权重不能互换。

## 安装

验证环境：Python 3.12、TensorFlow 2.16.2、Keras 3.4.1。依赖版本见 `requirements.txt`。

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux/macOS 则执行 source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 数据准备

使用 BCI Competition IV-2a 的 MATLAB 文件（BNCI 2014-001），不是 GDF 文件。请自行获取数据并遵守其使用条款；本仓库不分发数据。

```text
data/
  A01T.mat
  A01E.mat
  ...
  A09T.mat
  A09E.mat
```

解析器期望 MAT 文件包含 `data` 结构，读取 22 个 EEG 通道和每试次 1125 个采样点；保持原预处理的试次窗口及包含伪迹试次的默认行为。统一 CLI 当前仅支持 BCI IV-2a 的受试者内评估。遗留 HGD/CS2R 辅助函数不属于已验证的入口。

## 训练与评估

```bash
python experiment.py train --model atcnet --data-dir data --output runs/atcnet --seeds 1 2 3
python experiment.py train --model mamba --data-dir data --output runs/mamba --seeds 1 2 3
python experiment.py evaluate --data-dir data --output runs/atcnet
```

默认处理全部 9 位受试者。小规模运行可以添加 `--subjects 1 --epochs 1`。旧脚本 `main_TrainTest.py` 与 `main_TrainValTest.py` 现在转发到同一 CLI，需要相同的子命令和参数。

协议：

1. 将训练会话 T 分层划分为 80% 训练、20% 验证，默认划分种子 42。
2. 仅用训练子集拟合每通道、每时间点的标准化参数，再用于验证与测试。
3. 每个随机种子按验证损失保存权重；多个种子也仅按验证损失选择。
4. 所有受试者完成选择后，使用独立 E 会话评估所选模型。测试准确率不参与选择。

两个模型对比时必须使用相同数据、受试者、划分种子、种子集合和训练预算。默认未强制跨设备确定性，不承诺跨硬件逐位一致。

每次运行要求全新的输出目录，保存配置、数据 SHA-256、划分索引、标准化参数、每轮训练历史、权重和模型选择记录。`metrics.json` 包含各受试者准确率、Kappa、混淆矩阵及均值。独立评估复用保存的标准化参数，并检查测试文件指纹。当前不支持断点续训。

## 历史结果

| 历史实验 | README 报告的平均准确率 | Kappa |
| --- | ---: | ---: |
| ATCNet | 77.16% | 0.695 |
| Mamba 仓库旧实验 | 73.69% | 0.649 |

这些数字来自原仓库 README，**不是新版严格协议的复测结果，也不是经过控制变量验证的模型对比**。原脚本存在测试集参与模型选择、验证划分前标准化，以及 Mamba 脚本从第 6 位受试者开始训练等问题；旧权重与日志对应的完整实验配置无法仅凭现有资料确认。

- [ATCNet 原始表格](historical/atcnet/README.original.md)
- [Mamba 原始表格](historical/mamba/README.original.md)
- [归档来源与完整性清单](historical/README.md)

历史文件仅作为课程成果记录。新协议下的准确率有待在真实数据上完整重训，不能用旧表格替代。

## 测试

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

测试覆盖数据隔离、验证集选择、全部受试者默认配置、三种模型前向与梯度、权重恢复、自定义层序列化，以及合成数据上的训练与独立评估。GitHub Actions 使用 CPU，不下载真实 EEG 数据。

## 来源与许可

- ATCNet：Hamdi Altaheri, Ghulam Muhammad, Mansour Alsulaiman, *Physics-informed attention temporal convolutional network for EEG-based motor imagery classification*, IEEE TII。DOI: https://doi.org/10.1109/TII.2022.3197419
- Mamba：Albert Gu, Tri Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*。本仓库为 TensorFlow 探索性实现，非官方实现。
- 保留原代码的 King Saud University 版权声明，使用 [Apache-2.0](LICENSE)。
