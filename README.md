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

## 3D 打印视觉与 TinyViT

项目还提供面向 3D 打印过程的视觉状态识别模块。`printvision` 将打印图像切分为 patch，使用轻量 TinyViT 编码视觉特征，再与喷嘴温度、热床温度、挤出速率和层高等遥测信号融合。模型同时输出缺陷类别和连续工艺状态值，可用于打印过程监测、缺陷识别以及闭环控制接口。

生成演示帧、训练和导出：

```bash
python scripts/generate_synthetic.py --out data/synthetic --n 40 --size 64
PYTHONPATH=src python -m printvision.train --config configs/smoke.yaml
PYTHONPATH=src python -m printvision.infer \
  --checkpoint runs/printvision_smoke/last.pt \
  --image data/synthetic/images/frame_00000.png \
  --telemetry 205 60 0.8 0.2 \
  --export runs/printvision_smoke/model.ts
```

真实数据使用 CSV 清单，至少包含 `image` 和 `label` 两列；增加 `sequence_id` 后，可用 `scripts/prepare_manifest.py` 按打印序列生成 train/val/test 划分。TinyViT 的结构、输入字段和边缘部署说明见 [docs/tinyvit.md](docs/tinyvit.md)。
