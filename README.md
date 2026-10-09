# Simple CNN（卷积 + GAP + 全连接）

这个模型按照示意图设计：

```text
输入图像 [N, 3, H, W]
    ↓
4 个卷积块（Conv + GroupNorm + ReLU，其中前三层下采样）
    ↓
特征图 [N, 256, H/8, W/8]
    ↓
全局平均池化 AdaptiveAvgPool2d(1)
    ↓
[N, 256, 1, 1] → 展平为 [N, 256]
    ↓
全连接层 Linear(256, K)
    ↓
类别分数 [N, K]
```

## 使用

```python
import torch
from model import SimpleCNN

model = SimpleCNN(num_classes=10)
images = torch.randn(8, 3, 224, 224)
logits = model(images)

# 训练：CrossEntropyLoss 内部已经包含 softmax
labels = torch.randint(0, 10, (8,))
loss = torch.nn.CrossEntropyLoss()(logits, labels)
loss.backward()

# 推理：需要概率时再使用 softmax
model.eval()
with torch.no_grad():
    probabilities = torch.softmax(model(images), dim=1)
    predicted_classes = probabilities.argmax(dim=1)
```

运行结构检查：

```bash
python3 model.py
```

`AdaptiveAvgPool2d((1, 1))` 会把任意大小的最后一层特征图变成每通道一个数，
因此模型不依赖固定的输入宽高（但输入宽高至少应为 8 像素）。

## 训练“是否为鸟”分类器

准备数据时同时提供鸟和非鸟图片，并划分训练集与验证集：

```text
data/
├── train/
│   ├── bird/       # 鸟图片
│   └── not_bird/   # 猫、狗、车、风景、人等非鸟图片
└── val/
    ├── bird/
    └── not_bird/
```

安装依赖并训练：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_bird.py --data data --epochs 20
```

识别一张新图片：

```bash
python predict_bird.py /完整路径/待识别图片.jpg
```

## 迁移学习模型选择

当前数据集很小，从零训练 CNN 容易记住训练图片，却无法正确识别没有见过的图片。
更合适的方案是使用在 ImageNet 上预训练过的模型进行迁移学习。

### 推荐 ResNet18

本项目建议优先使用 **ResNet18**：

- 鸟／非鸟二分类任务不需要特别大的模型，ResNet18 已经足够；
- PyTorch 官方支持完善，迁移学习代码简单，便于调试；
- ImageNet 训练集中包含多种鸟类，预训练特征对当前任务有帮助；
- 虽然模型约有 1170 万个参数，但冻结主干后只训练最后的分类层，CPU 也可以训练。

ResNet18 和 MobileNetV3 Small 的主要差异如下：

| 模型 | 参数量 | 计算量 | 适用场景 |
| --- | ---: | ---: | --- |
| ResNet18 | 约 1170 万 | 约 1.81 GFLOPS | 迁移学习稳定、结构经典、便于调试 |
| MobileNetV3 Small | 约 250 万 | 约 0.06 GFLOPS | 模型更小、更快，适合手机或低功耗部署 |

当前项目首先需要改善识别效果，而不是追求部署性能，因此推荐从 ResNet18 开始。

参考资料：

- [PyTorch ResNet18 官方文档](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [PyTorch MobileNetV3 Small 官方文档](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)

### ResNet18 迁移学习示例

加载 ImageNet 预训练权重，然后冻结主干网络，只替换并训练最后的二分类层：

```python
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# 冻结预训练主干，保留已经学到的通用图像特征。
for parameter in model.parameters():
    parameter.requires_grad = False

# 将原来的 ImageNet 1000 分类头替换为鸟／非鸟二分类头。
model.fc = nn.Linear(model.fc.in_features, 2)
```

优化器只更新新分类层：

```python
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

预训练 ResNet18 默认使用 `224 × 224` 输入，并采用 ImageNet 的归一化参数：

```python
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
```

推荐分两个阶段训练：

1. 冻结所有预训练层，只训练新的全连接分类层；
2. 数据增加到每类几十张以上后，解冻最后一个残差层，并使用较低学习率进行微调。

迁移学习可以明显改善小数据集上的效果，但不能完全弥补数据不足。建议训练集每类至少准备
20～50 张图片，验证集每类至少准备 10 张图片，并保持两个类别的数量大致平衡。
