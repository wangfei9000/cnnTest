# CNN 鸟类二分类项目逐行代码分析

本文分析项目中实际参与训练和预测的三个 Python 文件：

- `model.py`：定义 CNN 模型；
- `train_bird.py`：读取数据、训练、验证并保存模型；
- `predict_bird.py`：加载模型并预测单张图片。

空行仅用于分隔逻辑段落，不执行任何操作。`model-1.py` 是 `model.py` 的副本，因此不重复分析。

## 一、`model.py`

### 第 1～10 行：导入依赖并定义卷积块

| 行号 | 代码 | 分析 |
| ---: | --- | --- |
| 1 | `"""一个使用全局平均池化（GAP）的简单图像分类 CNN。"""` | 模块文档字符串，说明该文件定义的是使用全局平均池化的卷积神经网络。 |
| 3 | `import torch` | 导入 PyTorch，后面用于张量操作、关闭梯度和生成测试输入。 |
| 4 | `from torch import nn` | 导入神经网络模块，后面通过 `nn.Module`、`nn.Conv2d` 等组件搭建网络。 |
| 7 | `class ConvBlock(nn.Sequential):` | 定义一个卷积块，并继承 `nn.Sequential`，使内部各层按顺序执行。 |
| 8 | `"""卷积 + 组归一化 + ReLU，可选 2 倍下采样。"""` | 类文档字符串，概括卷积块的组成。 |
| 10 | `def __init__(...)` | 定义构造函数。`in_channels` 是输入通道数，`out_channels` 是输出通道数，`downsample` 控制是否缩小特征图。 |

### 第 11～26 行：组装卷积块

| 行号 | 代码 | 分析 |
| ---: | --- | --- |
| 11 | `layers = [` | 创建层列表，稍后统一交给 `nn.Sequential`。 |
| 12 | `nn.Conv2d(` | 创建二维卷积层，用卷积核从图片或上一层特征图中提取局部特征。 |
| 13 | `in_channels,` | 指定输入特征图的通道数。第一层通常为 RGB 的 3 通道。 |
| 14 | `out_channels,` | 指定卷积后生成多少个特征通道。通道越多，可学习的特征类型通常越丰富。 |
| 15 | `kernel_size=3,` | 使用 `3×3` 卷积核，是图像 CNN 中常见的尺寸。 |
| 16 | `padding=1,` | 在四周填充 1 个像素，使步长为 1 时卷积前后宽高不变。 |
| 17 | 注释 | 解释归一化层已经提供可学习偏移，因此卷积层可以不使用偏置。 |
| 18 | `bias=False,` | 关闭卷积层偏置，减少少量冗余参数。 |
| 19 | `),` | 结束 `Conv2d` 的构造。 |
| 20 | 注释 | 说明选择 GroupNorm 是因为它不依赖批次大小，更适合当前小数据集。 |
| 21 | `nn.GroupNorm(...)` | 对通道分组归一化。最多使用 8 组；`min` 可避免输出通道少于 8 时组数过大。当前各层通道数都能被 8 整除。 |
| 22 | `nn.ReLU(inplace=True),` | 加入 ReLU 非线性激活；`inplace=True` 复用输入内存。 |
| 23 | `]` | 结束基础层列表。 |
| 24 | `if downsample:` | 根据参数决定是否执行空间下采样。 |
| 25 | `layers.append(nn.MaxPool2d(...))` | 添加 `2×2`、步长 2 的最大池化，使特征图宽高各缩小一半。 |
| 26 | `super().__init__(*layers)` | 将列表展开后传给父类，使 `ConvBlock` 成为可以直接调用的顺序网络。 |

### 第 29～45 行：定义完整 CNN

| 行号 | 代码 | 分析 |
| ---: | --- | --- |
| 29 | `class SimpleCNN(nn.Module):` | 定义完整分类模型，继承所有 PyTorch 模型的基类 `nn.Module`。 |
| 30 | 文档字符串 | 说明模型由特征提取、全局平均池化和全连接分类三部分构成。 |
| 32 | `def __init__(...)` | `num_classes` 指类别数量，`in_channels` 默认是 RGB 图片的 3 通道。 |
| 33 | `super().__init__()` | 初始化 `nn.Module`，确保子模块和参数能够被 PyTorch 正确注册。 |
| 34 | `if num_classes < 2:` | 检查类别数是否合理；分类模型至少应有两个类别。 |
| 35 | `raise ValueError(...)` | 参数错误时立即报错，避免构造出无意义的模型。 |
| 37 | `self.features = nn.Sequential(` | 创建按顺序执行的卷积特征提取器。赋给 `self` 后，其参数会被模型注册。 |
| 38 | `ConvBlock(in_channels, 32)` | 第一块将 RGB 通道扩展到 32，并把宽高缩小为原来的二分之一。 |
| 39 | `ConvBlock(32, 64)` | 第二块将通道增加到 64，宽高缩小到原来的四分之一。 |
| 40 | `ConvBlock(64, 128)` | 第三块将通道增加到 128，宽高缩小到原来的八分之一。 |
| 41 | `ConvBlock(128, 256, False)` | 第四块将通道增加到 256，但不再下采样，以保留更多空间信息。 |
| 42 | `)` | 结束特征提取器定义。 |
| 43 | 注释 | 解释 GAP 会把每个通道压缩成一个数，因此分类头不依赖固定图片宽高。 |
| 44 | `self.gap = nn.AdaptiveAvgPool2d((1, 1))` | 对每个通道的全部空间位置求平均，输出形状固定为 `[N, 256, 1, 1]`。 |
| 45 | `self.classifier = nn.Linear(256, num_classes)` | 将 256 个全局特征映射为每个类别的原始分数。 |

### 第 47～57 行：前向传播和概率预测

| 行号 | 代码 | 分析 |
| ---: | --- | --- |
| 47 | `def forward(...)` | 定义模型被调用时执行的前向传播过程，并用类型标注说明输入输出都是张量。 |
| 48 | `x = self.features(x)` | 依次通过四个卷积块，将原始图片转换为高级特征图。 |
| 49 | `x = self.gap(x)` | 将每个通道的空间特征汇总成一个数，输出 `[N, 256, 1, 1]`。 |
| 50 | `x = torch.flatten(x, 1)` | 从第 1 维开始展平，保留批次维，得到 `[N, 256]`。 |
| 51 | 注释 | 说明训练所需的是 logits，而不是提前计算的概率。 |
| 52 | `return self.classifier(x)` | 通过全连接层返回 `[N, num_classes]` 的 logits。 |
| 54 | `@torch.no_grad()` | 装饰器关闭该方法内的梯度记录，减少推理的内存和计算开销。 |
| 55 | `def predict_proba(...)` | 定义一个直接返回类别概率的辅助方法。 |
| 56 | 文档字符串 | 说明方法的返回值含义。 |
| 57 | `return torch.softmax(self(x), dim=1)` | 先调用模型得到 logits，再沿类别维执行 softmax，使每张图片的类别概率之和为 1。 |

### 第 60～71 行：直接运行时进行结构测试

| 行号 | 代码 | 分析 |
| ---: | --- | --- |
| 60 | `if __name__ == "__main__":` | 只有直接运行 `python model.py` 时才执行测试；被其他文件导入时不会执行。 |
| 61 | `model = SimpleCNN(num_classes=10)` | 创建一个用于测试的 10 分类模型。 |
| 63 | 注释 | 说明下面用不同图片尺寸检查 GAP 的自适应能力。 |
| 64 | `for image_size in (...)` | 分别测试正方形输入和非正方形输入。 |
| 65 | `sample = torch.randn(...)` | 创建 4 张随机 RGB 图片，`*image_size` 将元组展开为高和宽。 |
| 66 | `logits = model(sample)` | 执行一次前向传播。 |
| 67 | `probabilities = torch.softmax(...)` | 将测试输出转换为概率。 |
| 68～71 | `print(...)` | 输出输入形状、logits 形状和第一张图片的概率总和，用于确认结构和 softmax 正常。 |

## 二、`train_bird.py`

### 第 1～20 行：导入依赖和选择设备

| 行号 | 代码 | 分析 |
| ---: | --- | --- |
| 1 | 模块文档字符串 | 说明该脚本用于训练鸟／非鸟二分类模型。 |
| 3 | `import argparse` | 用于解析命令行参数。 |
| 4 | `from pathlib import Path` | 用面向对象的方式处理数据集和模型文件路径。 |
| 6 | `import torch` | 导入 PyTorch 核心功能。 |
| 7 | `from torch import nn` | 导入损失函数等神经网络组件。 |
| 8 | `from torch.utils.data import DataLoader` | 导入数据加载器，负责分批、打乱和迭代数据。 |
| 9 | `from torchvision import datasets, transforms` | 导入文件夹数据集和图像预处理工具。 |
| 11 | `from model import SimpleCNN` | 从 `model.py` 导入当前自定义 CNN。 |
| 14 | `def get_device()` | 定义自动选择计算设备的函数。 |
| 15 | 文档字符串 | 说明设备选择顺序。 |
| 16～17 | `if ... mps ...` | 如果 Apple Silicon 的 MPS 可用，优先使用 Apple GPU。 |
| 18～19 | `if ... cuda ...` | 否则检查 NVIDIA CUDA GPU。 |
| 20 | `return torch.device("cpu")` | 没有可用 GPU 时回退到 CPU。 |

### 第 23～43 行：定义训练和验证预处理

| 行号 | 代码 | 分析 |
| ---: | --- | --- |
| 23 | `def make_loaders(...)` | 定义创建训练集、验证集和 DataLoader 的函数。 |
| 24 | 文档字符串 | 说明函数还会返回类别到索引的映射。 |
| 25 | 注释 | 解释训练增强和 ImageNet 归一化的目的。 |
| 26 | `train_transform = transforms.Compose(` | 将多个训练预处理步骤串联起来。 |
| 27、33～34 | 列表及结束符 | 包裹并结束训练预处理流水线，没有独立计算含义。 |
| 28 | `RandomResizedCrop(128, scale=(0.7, 1.0))` | 随机裁剪原图 70%～100% 的区域并缩放到 `128×128`，增加视角变化。 |
| 29 | `RandomHorizontalFlip()` | 默认以 50% 概率水平翻转图片，扩充训练样本变化。 |
| 30 | `ColorJitter(...)` | 随机调整亮度和对比度，降低模型对固定光照的依赖。 |
| 31 | `ToTensor()` | 把 PIL 图片转换为 `[C,H,W]` 张量，并把像素缩放到 `[0,1]`。 |
| 32 | `Normalize(...)` | 使用 ImageNet 的 RGB 均值和标准差标准化三个通道。 |
| 35 | 注释 | 说明验证预处理不能随机变化，否则各轮指标不可直接比较。 |
| 36 | `val_transform = transforms.Compose(` | 创建确定性的验证预处理流水线。 |
| 37、42～43 | 列表及结束符 | 包裹并结束验证预处理步骤。 |
| 38 | `Resize(144)` | 将图片较短边缩放到 144，保持宽高比例。 |
| 39 | `CenterCrop(128)` | 从中心裁剪 `128×128` 区域，与训练输入尺寸一致。 |
| 40 | `ToTensor()` | 将验证图片转换为张量。 |
| 41 | `Normalize(...)` | 使用与训练阶段相同的归一化，防止输入分布不一致。 |

### 第 45～59 行：读取数据并创建加载器

| 行号 | 代码 | 分析 |
| ---: | --- | --- |
| 45 | `train_set = datasets.ImageFolder(...)` | 读取 `data/train/类别名/图片` 结构，并应用训练预处理。 |
| 46 | `val_set = datasets.ImageFolder(...)` | 读取验证目录并应用验证预处理。 |
| 47 | 注释 | 提醒 `ImageFolder` 会按文件夹名字典序生成类别编号。 |
| 48 | `if set(train_set.classes) != ...` | 确保训练目录恰好包含 `bird` 和 `not_bird`。 |
| 49 | `raise ValueError(...)` | 类别目录错误时停止训练并给出明确提示。 |
| 50 | `if train_set.class_to_idx != ...` | 确保训练集和验证集对类别编号的理解完全一致。 |
| 51 | `raise ValueError(...)` | 映射不一致时停止，避免标签含义错位。 |
| 53～55 | `train_loader = DataLoader(...)` | 创建训练加载器。使用指定批大小、每轮打乱数据；`num_workers=0` 表示主进程读取。 |
| 56～58 | `val_loader = DataLoader(...)` | 创建验证加载器。验证时不打乱，方便复现和排查。 |
| 59 | `return ...` | 返回两个加载器和训练集的类别映射。 |

### 第 62～75 行：验证函数

| 行号 | 代码 | 分析 |
| ---: | --- | --- |
| 62 | `def evaluate(...)` | 定义通用验证函数，接收模型、数据、损失函数和设备。 |
| 63 | 文档字符串 | 说明函数计算全数据集平均损失和准确率。 |
| 64 | 注释 | 解释验证模式对 Dropout 等训练专用行为的影响。 |
| 65 | `model.eval()` | 把模型切换到评估模式；GroupNorm 在训练和评估模式下计算方式相同。 |
| 66 | `total_loss = correct = count = 0` | 同时初始化累计损失、正确数和样本总数。 |
| 67 | `with torch.no_grad():` | 验证时关闭梯度记录，节省资源且防止无意构建计算图。 |
| 68 | `for images, labels in loader:` | 逐批读取验证图片和真实标签。 |
| 69 | `images, labels = ...to(device)` | 把数据移动到模型所在设备。 |
| 70 | `logits = model(images)` | 前向传播得到类别分数。 |
| 71 | 注释 | 解释为什么批平均损失需要乘批大小后再累计。 |
| 72 | `total_loss += ...` | 把当前批次的平均损失还原为总损失并累加。 |
| 73 | `correct += ...` | 取最高分的类别，与真实标签比较并累计正确数量。 |
| 74 | `count += labels.size(0)` | 累计当前批次的样本数。 |
| 75 | `return total_loss / count, correct / count` | 返回全验证集的平均损失和准确率。 |

### 第 78～96 行：解析参数并初始化训练

| 行号 | 代码 | 分析 |
| ---: | --- | --- |
| 78 | `def main():` | 定义训练脚本主流程。 |
| 79 | `parser = argparse.ArgumentParser()` | 创建命令行参数解析器。 |
| 80 | `--data` | 设置数据根目录，默认是当前目录下的 `data`。 |
| 81 | `--epochs` | 设置完整遍历训练集的次数，默认 20。 |
| 82 | `--batch-size` | 设置每批图片数量，默认 32；当前只有 6 张训练图，所以实际一批就是 6 张。 |
| 83 | `--lr` | 设置学习率，默认 `0.001`。 |
| 84 | `--output` | 设置最佳模型检查点路径，默认 `bird_cnn.pt`。 |
| 85 | `args = parser.parse_args()` | 解析用户在终端输入的参数。 |
| 87 | `device = get_device()` | 选择训练设备。 |
| 88 | `... = make_loaders(...)` | 创建数据加载器并取得类别映射。 |
| 89 | `model = SimpleCNN(num_classes=2).to(device)` | 创建二分类 CNN 并移动到训练设备。 |
| 90 | `criterion = nn.CrossEntropyLoss()` | 创建交叉熵损失；它直接接收 logits 和整数类别标签。 |
| 91 | `optimizer = torch.optim.Adam(...)` | 创建 Adam 优化器，用指定学习率更新模型全部可训练参数。 |
| 92 | `best_accuracy = 0.0` | 保存目前最佳验证准确率，用于选择检查点。 |
| 93 | `best_loss = float("inf")` | 保存最佳模型的验证损失，初始为正无穷。 |
| 95 | 第一条 `print` | 输出设备和类别映射，便于确认训练配置。 |
| 96 | 第二条 `print` | 输出训练集和验证集样本数量。 |

### 第 98～113 行：训练循环

| 行号 | 代码 | 分析 |
| ---: | --- | --- |
| 98 | `for epoch in range(1, args.epochs + 1):` | 从 1 开始执行指定数量的训练轮次。上限加 1 是因为 `range` 不包含末值。 |
| 99 | 注释 | 解释验证结束后为何必须重新切回训练模式。 |
| 100 | `model.train()` | 将模型切换到训练模式。 |
| 101 | `train_loss = train_correct = train_count = 0` | 重置本轮训练统计量。 |
| 102 | `for images, labels in train_loader:` | 逐批读取训练数据。 |
| 103 | `images, labels = ...to(device)` | 将图片和标签移动到训练设备。 |
| 104 | 注释 | 说明 PyTorch 梯度默认会累积。 |
| 105 | `optimizer.zero_grad()` | 清除上一批次保存在参数上的梯度。 |
| 106 | `logits = model(images)` | 执行前向传播，得到每张图片的两个类别分数。 |
| 107 | `loss = criterion(logits, labels)` | 比较预测与真实标签，计算当前批次的交叉熵损失。 |
| 108 | `loss.backward()` | 反向传播，计算损失对每个可训练参数的梯度。 |
| 109 | `optimizer.step()` | Adam 根据梯度更新模型参数，模型在这一步真正学习。 |
| 111 | `train_loss += ...` | 累计当前批次所有样本的损失。 |
| 112 | `train_correct += ...` | 累计预测正确的训练样本数。 |
| 113 | `train_count += ...` | 累计已处理的训练样本数。 |

### 第 115～141 行：验证、选择并保存最佳模型

| 行号 | 代码 | 分析 |
| ---: | --- | --- |
| 115 | `val_loss, val_accuracy = evaluate(...)` | 完成本轮训练后，在验证集上计算损失和准确率。 |
| 116～121 | `print(...)` | 格式化输出轮次、训练平均损失、训练准确率、验证损失和验证准确率。 |
| 123～124 | 注释 | 说明最佳模型的比较规则及其对小验证集的意义。 |
| 125 | `is_better = val_accuracy > ... or (` | 如果准确率严格提高，则当前模型更好。 |
| 126 | `val_accuracy == ... and val_loss < ...` | 准确率相同时，用更低的验证损失作为第二判断标准。 |
| 127 | `)` | 结束布尔表达式。 |
| 128 | `if is_better:` | 只有当前模型更好时才更新最佳记录并保存。 |
| 129 | `best_accuracy = val_accuracy` | 记录新的最佳准确率。 |
| 130 | `best_loss = val_loss` | 记录该检查点对应的验证损失。 |
| 131 | `torch.save(` | 开始把检查点序列化到磁盘。 |
| 132 | `{` | 创建包含模型及推理元数据的字典。 |
| 133 | `"model_state": model.state_dict(),` | 保存所有可学习权重及模块状态，不保存完整 Python 模型对象。 |
| 134 | 注释 | 解释为什么标签映射和输入尺寸也需要保存在检查点中。 |
| 135 | `"class_to_idx": class_to_idx,` | 保存类别名称到数字标签的映射。 |
| 136 | `"image_size": 128,` | 保存训练时使用的目标图片尺寸，供推理复用。 |
| 137 | `},` | 结束检查点字典。 |
| 138 | `args.output,` | 指定用户设置或默认的模型输出路径。 |
| 139 | `)` | 完成模型保存。 |
| 141 | `print(...)` | 所有轮次结束后，输出最佳准确率和模型文件路径。 |
| 144～145 | 主程序入口 | 直接运行文件时调用 `main()`；被导入时不自动训练。 |

## 三、`predict_bird.py`

### 第 1～17 行：导入依赖并解析参数

| 行号 | 代码 | 分析 |
| ---: | --- | --- |
| 1 | 模块文档字符串 | 说明脚本用于判断单张图片是否为鸟。 |
| 3 | `import argparse` | 用于解析待识别图片和模型路径。 |
| 4 | `from pathlib import Path` | 将命令行路径转换为 `Path` 对象。 |
| 6 | `import torch` | 用于加载检查点、执行模型和计算概率。 |
| 7 | `from PIL import Image` | 使用 Pillow 打开图片并转换颜色模式。 |
| 8 | `from torchvision import transforms` | 导入与训练阶段一致的图片预处理工具。 |
| 10 | `from model import SimpleCNN` | 导入模型结构；检查点只包含参数，因此加载前必须先创建相同结构。 |
| 13 | `def main():` | 定义预测主流程。 |
| 14 | `parser = argparse.ArgumentParser()` | 创建命令行参数解析器。 |
| 15 | `image` 参数 | 定义必填的位置参数，即待识别图片路径。 |
| 16 | `--model` 参数 | 定义可选模型路径，默认使用 `bird_cnn.pt`。 |
| 17 | `args = parser.parse_args()` | 读取并保存命令行参数。 |

### 第 19～28 行：恢复检查点和模型

| 行号 | 代码 | 分析 |
| ---: | --- | --- |
| 19 | 注释 | 说明将权重映射到 CPU 可以避免训练设备与预测设备不一致。 |
| 20 | `checkpoint = torch.load(...)` | 从磁盘读取检查点；`map_location="cpu"` 强制放在 CPU，`weights_only=False` 允许读取整个字典。 |
| 21 | `class_to_idx = ...` | 读取训练时保存的类别名称到数字索引映射。 |
| 22 | 注释 | 解释反向映射的用途。 |
| 23 | `idx_to_class = {...}` | 交换字典的键和值，用于将预测数字转回 `bird` 或 `not_bird`。 |
| 24 | `image_size = checkpoint.get(...)` | 读取训练输入尺寸；旧检查点没有该字段时使用 128。 |
| 26 | `model = SimpleCNN(num_classes=2)` | 在 CPU 上创建与训练时相同的二分类模型结构。 |
| 27 | `model.load_state_dict(...)` | 将检查点中的已训练参数复制到模型。结构不匹配时会报错。 |
| 28 | `model.eval()` | 切换到评估模式，确保推理行为稳定。 |

### 第 30～41 行：预处理单张图片

| 行号 | 代码 | 分析 |
| ---: | --- | --- |
| 30 | 注释 | 强调预测预处理必须与验证预处理一致。 |
| 31 | `transform = transforms.Compose(` | 创建预测预处理流水线。 |
| 32、37～38 | 列表及结束符 | 包裹并结束各项预处理步骤。 |
| 33 | `Resize(image_size + 16)` | 将图片短边放大到目标尺寸加 16；当目标为 128 时即缩放到 144。 |
| 34 | `CenterCrop(image_size)` | 从中心裁剪出训练所需大小。 |
| 35 | `ToTensor()` | 将图片转换为浮点张量。 |
| 36 | `Normalize(...)` | 使用与训练和验证相同的 ImageNet 均值及标准差。 |
| 39 | `Image.open(...).convert("RGB")` | 打开图片并强制转换为三通道 RGB，兼容灰度图和带透明通道图片。 |
| 40 | 注释 | 解释模型需要批次维。 |
| 41 | `batch = transform(image).unsqueeze(0)` | 完成预处理，再在最前面增加批次维，得到 `[1,3,H,W]`。 |

### 第 43～55 行：计算并输出预测

| 行号 | 代码 | 分析 |
| ---: | --- | --- |
| 43 | 注释 | 说明推理时关闭梯度的原因。 |
| 44 | `with torch.no_grad():` | 在该代码块内禁止构建反向传播计算图。 |
| 45 | `probabilities = ...` | 模型先输出 `[1,2]` logits，softmax 转成概率，`[0]` 取出批次中的第一张图片。 |
| 47 | `bird_probability = ...` | 根据保存的映射找到 `bird` 对应位置，并提取 Python 浮点数。没有硬编码类别顺序。 |
| 48 | `prediction = ...` | 找到概率最大的类别索引，再转换为类别名称。 |
| 49 | `result = ...` | 将英文内部标签转换为中文输出。默认以两类中概率较大者为结果，等价于二分类中约 50% 的边界。 |
| 50 | `print(f"判断: {result}")` | 输出最终类别判断。 |
| 51 | `print(f"是鸟概率: ...")` | 以百分比形式输出鸟类别概率，并保留两位小数。 |
| 54～55 | 主程序入口 | 直接运行脚本时调用 `main()`，开始预测。 |

## 四、三个文件如何协作

```text
data/ 中的图片
      │
      ▼
train_bird.py ──导入──▶ model.py / SimpleCNN
      │
      │ 训练并保存权重、类别映射、图片尺寸
      ▼
bird_cnn.pt
      │
      ▼
predict_bird.py ──创建相同模型并加载权重──▶ 预测结果
```

训练阶段最关键的五行是：

```python
optimizer.zero_grad()       # 清除旧梯度
logits = model(images)      # 前向传播
loss = criterion(logits, labels)  # 计算误差
loss.backward()             # 反向传播计算梯度
optimizer.step()            # 根据梯度更新参数
```

预测阶段最关键的三步是：

```python
model.load_state_dict(checkpoint["model_state"])
model.eval()
probabilities = torch.softmax(model(batch), dim=1)[0]
```

> 注意：README 中的 ResNet18 迁移学习内容目前只是建议和示例，尚未写入上述 Python 源码；当前训练和预测仍使用 `SimpleCNN`。
