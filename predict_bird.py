"""用训练好的模型判断一张图片是否为鸟。"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import SimpleCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="待识别图片路径")
    parser.add_argument("--model", type=Path, default=Path("bird_cnn.pt"))
    args = parser.parse_args()

    # 映射到 CPU，保证在没有训练所用 GPU 的机器上也能加载模型。
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    # 模型输出的是类别索引，反向映射用于把索引还原成可读标签。
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    image_size = checkpoint.get("image_size", 128)

    model = SimpleCNN(num_classes=2)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 推理阶段的裁剪尺寸和归一化必须与验证阶段一致，否则输入分布会漂移。
    transform = transforms.Compose(
        [
            transforms.Resize(image_size + 16),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    image = Image.open(args.image).convert("RGB")
    # 网络期望 [N, C, H, W]，单张图片需要补上大小为 1 的批次维度。
    batch = transform(image).unsqueeze(0)

    # 关闭梯度记录可减少推理时的内存和计算开销。
    with torch.no_grad():
        probabilities = torch.softmax(model(batch), dim=1)[0]

    bird_probability = probabilities[class_to_idx["bird"]].item()
    prediction = idx_to_class[probabilities.argmax().item()]
    result = "是鸟" if prediction == "bird" else "不是鸟"
    print(f"判断: {result}")
    print(f"是鸟概率: {bird_probability:.2%}")


if __name__ == "__main__":
    main()
