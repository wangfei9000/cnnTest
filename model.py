"""一个使用全局平均池化（GAP）的简单图像分类 CNN。"""

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    """卷积 + 组归一化 + ReLU，可选 2 倍下采样。"""

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                # 后接归一化层时不需要卷积层再学习一份偏置。
                bias=False,
            ),
            # GroupNorm 不依赖批次统计量，比 BatchNorm 更适合当前的极小数据集。
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True),
        ]
        if downsample:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        super().__init__(*layers)


class SimpleCNN(nn.Module):
    """卷积特征提取 -> GAP -> 全连接分类。"""

    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes 必须不小于 2")

        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),   # [N, 3, H, W] -> [N, 32, H/2, W/2]
            ConvBlock(32, 64),            # -> [N, 64, H/4, W/4]
            ConvBlock(64, 128),           # -> [N, 128, H/8, W/8]
            ConvBlock(128, 256, False),   # -> [N, 256, H/8, W/8]
        )
        # 每个通道压缩成一个全局特征，使分类头不依赖输入图片的宽高。
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)             # [N, 256, 1, 1]
        x = torch.flatten(x, 1)     # [N, 256]
        # 返回原始 logits；CrossEntropyLoss 会在内部完成 log-softmax。
        return self.classifier(x)   # [N, num_classes]

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """返回每个类别的预测概率。"""
        return torch.softmax(self(x), dim=1)


if __name__ == "__main__":
    model = SimpleCNN(num_classes=10)

    # GAP 允许模型接收不同空间尺寸的图片。
    for image_size in ((224, 224), (128, 160)):
        sample = torch.randn(4, 3, *image_size)
        logits = model(sample)
        probabilities = torch.softmax(logits, dim=1)
        print(
            f"input={tuple(sample.shape)}, logits={tuple(logits.shape)}, "
            f"probability_sum={probabilities[0].sum().item():.4f}"
        )
