# TinyViT 3D 打印视觉项目

## 目标

该模块面向打印过程中的视觉感知和状态监测。每个样本由一帧相机图像、可选的机器遥测和一个缺陷标签组成。模型输出分类 logits、连续状态值和融合后的 embedding，方便接入告警、质量追踪或闭环控制程序。

## 数据清单

CSV 至少包含以下字段：

```text
image,label
frame_00001.png,normal
frame_00002.png,defect
```

建议同时记录 `sequence_id` 和 `split`。遥测字段可以使用 `nozzle_temperature`、`bed_temperature`、`extrusion_rate`、`layer_height`，并在配置文件的 `telemetry_columns` 中保持相同顺序。图像文件相对于 `image_root` 解析。

如果原始清单还没有划分字段，可以运行：

```bash
python scripts/prepare_manifest.py \
  --input data/frames.csv \
  --output data/frames_split.csv \
  --seed 42
```

脚本以打印序列为单位计算稳定哈希，同一序列不会同时进入不同数据集。

## TinyViT 与遥测融合

`TinyViT` 使用 patch convolution、可学习的分类 token、位置编码和 Transformer Encoder。默认配置为 128×128 输入、16×16 patch、192 维 token、4 层 Encoder 和 6 个注意力头。它只依赖 PyTorch，不需要下载预训练权重，适合先在工作站训练，再导出到边缘设备。

`PrintStateModel` 把 TinyViT 的视觉 embedding 与经过 LayerNorm 和线性层映射的遥测向量拼接，经过融合层后提供三个输出：

- `logits`：缺陷或打印状态分类结果；
- `state`：连续工艺状态值，可作为控制器的风险或质量输入；
- `embedding`：用于检索、聚类或后续多模态适配的特征。

## 训练与部署

`configs/smoke.yaml` 是 CPU 冒烟配置，使用较小模型和合成帧；`configs/default.yaml` 用于真实数据接入。训练会把模型权重、配置和标签映射写入 `last.pt`。`printvision.infer` 支持单帧预测，并可通过 `--export` 生成 TorchScript 文件：

```bash
PYTHONPATH=src python -m printvision.train --config configs/default.yaml
PYTHONPATH=src python -m printvision.infer \
  --checkpoint runs/default/last.pt \
  --image data/images/frame.png \
  --telemetry 205 60 0.8 0.2 \
  --export runs/default/model.ts
```

部署到嵌入式设备时，保持训练和推理的图像尺寸、归一化参数、遥测顺序一致，并在目标硬件上测量吞吐、延迟和内存占用。后续可在视觉 token 上接入 VLM 适配器，或将 `state` 输出连接到打印机控制服务。
