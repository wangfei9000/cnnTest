# Simple CNN（卷积 + GAP + 全连接）

这个模型按照示意图设计：

```text
输入图像 [N, 3, H, W]
    ↓
4 个卷积块（Conv + BatchNorm + ReLU，其中前三层下采样）
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
