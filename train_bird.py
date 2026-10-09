"""训练“鸟 / 非鸟”二分类模型。"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleCNN


def get_device() -> torch.device:
    """按 Apple GPU、NVIDIA GPU、CPU 的优先级选择训练设备。"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loaders(data_dir: Path, batch_size: int):
    """创建训练/验证数据加载器，并返回 ImageFolder 生成的类别映射。"""
    # 训练集使用随机增强来降低小数据集上的过拟合；归一化参数取自 ImageNet。
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(128, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    # 验证集必须保持确定性，否则每轮准确率之间不具备可比性。
    val_transform = transforms.Compose(
        [
            transforms.Resize(144),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_set = datasets.ImageFolder(data_dir / "train", train_transform)
    val_set = datasets.ImageFolder(data_dir / "val", val_transform)
    # ImageFolder 按文件夹名称排序分配标签，因此还需确认训练、验证映射完全相同。
    if set(train_set.classes) != {"bird", "not_bird"}:
        raise ValueError("训练目录必须只包含 bird 和 not_bird 两个类别文件夹")
    if train_set.class_to_idx != val_set.class_to_idx:
        raise ValueError("train 和 val 的类别文件夹必须一致")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, train_set.class_to_idx


def evaluate(model, loader, criterion, device):
    """在不记录梯度的情况下计算整个数据集的平均损失和准确率。"""
    # eval() 会冻结 BatchNorm 的统计量，并关闭可能存在的 Dropout。
    model.eval()
    total_loss = correct = count = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            # criterion 默认返回批次均值，乘以批大小后才能正确汇总不等长批次。
            total_loss += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
    return total_loss / count, correct / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("bird_cnn.pt"))
    args = parser.parse_args()

    device = get_device()
    train_loader, val_loader, class_to_idx = make_loaders(args.data, args.batch_size)
    model = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_accuracy = 0.0

    print(f"设备: {device}; 类别: {class_to_idx}")
    print(f"训练图片: {len(train_loader.dataset)}; 验证图片: {len(val_loader.dataset)}")

    for epoch in range(1, args.epochs + 1):
        # evaluate() 会切到 eval 模式，所以每个新 epoch 都要显式切回训练模式。
        model.train()
        train_loss = train_correct = train_count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # PyTorch 默认累积梯度，每个批次反向传播前必须清空上一次的梯度。
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_count += labels.size(0)

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss / train_count:.4f}, "
            f"acc {train_correct / train_count:.2%} | "
            f"val loss {val_loss:.4f}, acc {val_accuracy:.2%}"
        )

        # 只保存验证集表现最好的权重，避免最后一轮过拟合后覆盖更好的模型。
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    # 推理时复用标签映射和输入尺寸，避免类别索引或预处理不一致。
                    "class_to_idx": class_to_idx,
                    "image_size": 128,
                },
                args.output,
            )

    print(f"最佳验证准确率: {best_accuracy:.2%}; 模型已保存到 {args.output}")


if __name__ == "__main__":
    main()
