# Multi-modal Temporal Emotion

MMTED（Multi-Modal Temporal Emotion Detector）项目用于从 EDA/GSR 和眼动时序信号中识别 VR 场景下的情绪状态。

项目包含：

- EDA/GSR 时序编码器：一维卷积 + Transformer。
- 眼动时序编码器：Transformer。
- 模态注意力融合和 4/8 类情绪分类。
- 基线校准、训练集标准化、参与者隔离划分和五折交叉验证。
- Accuracy、Balanced Accuracy、Macro-F1、AUC 和混淆矩阵评估。
- TorchScript 边缘推理导出。
- 确定性的合成数据生成器，用于检查完整工程流程。

项目数据协议使用 NPZ 文件：

```text
eda_gsr: (N, T, C)
eye:     (N, T, F)
labels:  (N,)
subject: (N,)
```

其中 `N` 是样本数，`T` 是时间长度，`C` 是皮肤电通道数，`F` 是眼动特征数。

## 安装

CPU 环境可以使用以下命令：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.2.2+cpu'
python -m pip install 'numpy>=1.24,<2.0' 'scipy>=1.10,<1.14' 'scikit-learn>=1.3,<1.6' pillow pyyaml tqdm pytest ruff
python -m pip install --no-deps -e '.[dev]'
```

## 运行流程

生成演示数据：

```bash
python scripts/generate_mmted_synthetic.py --out data/mmted.npz --n 96 --classes 4
```

训练和评估：

```bash
python -m mmted.train --data data/mmted.npz --classes 4 --epochs 2 --output runs/mmted
python -m mmted.evaluate --data data/mmted.npz --checkpoint runs/mmted/last.pt
```

五折交叉验证：

```bash
python -m mmted.cross_validate --data data/mmted.npz --folds 5 --epochs 1
```

加上 `--grouped` 可以按参与者进行隔离交叉验证。`--no-attention` 可用于模态融合消融。

导出边缘推理文件：

```bash
python -m mmted.export --checkpoint runs/mmted/last.pt --output runs/mmted/model.ts
```

项目数据、预处理、模型结构和实验说明见 [docs/methods.md](docs/methods.md)。
