# Softmax 原理与 PyTorch 使用指南

## 1. Softmax 是什么

Softmax 是一种常用于多分类模型输出层的激活函数。它接收一组任意实数分数，将其转换为一组非负且总和为 1 的数值，因此输出可以被解释为各类别的概率分布。

在 PyTorch 中，可以使用：

```python
torch.nn.Softmax(dim=...)
```

或者函数式接口：

```python
torch.softmax(input, dim=...)
```

模型在 Softmax 之前输出的原始分数通常称为 **logits**。Logit 可以是任意实数，不要求在 0 到 1 之间，也不要求总和为 1。

例如，一个三分类模型可能输出：

```text
[1.0, 2.0, 3.0]
```

这些值表示模型对三个类别的相对偏好，但它们还不是概率。经过 Softmax 后得到：

```text
[0.0900, 0.2447, 0.6652]
```

此时三个数都大于 0，而且总和为 1。

---

## 2. Softmax 公式

假设模型输出一个包含 $K$ 个分数的向量：

$$
\mathbf{x}=(x_1,x_2,\ldots,x_K)
$$

Softmax 将第 $i$ 个分数转换为：

$$
p_i=\mathrm{Softmax}(x_i)
=\frac{e^{x_i}}{\sum_{j=1}^{K}e^{x_j}}
$$

其中：

- $x_i$：第 $i$ 个类别的原始分数，即 logit；
- $e$：自然常数，约等于 2.71828；
- $e^{x_i}$：第 $i$ 个类别的正权重；
- $\sum_{j=1}^{K}e^{x_j}$：所有类别权重的总和；
- $p_i$：第 $i$ 个类别经过归一化后的概率。

这个公式可以分为两步理解：

1. 对每个 logit 取指数，把任意实数变成正数；
2. 用每个指数值除以所有指数值之和，完成归一化。

---

## 3. 逐步计算示例

给定 logits：

$$
\mathbf{x}=(1,2,3)
$$

### 3.1 计算指数

$$
e^1\approx2.718
$$

$$
e^2\approx7.389
$$

$$
e^3\approx20.086
$$

指数值之和为：

$$
2.718+7.389+20.086=30.193
$$

### 3.2 分别归一化

第一个类别：

$$
p_1=\frac{e^1}{e^1+e^2+e^3}
\approx\frac{2.718}{30.193}
\approx0.0900
$$

第二个类别：

$$
p_2\approx\frac{7.389}{30.193}\approx0.2447
$$

第三个类别：

$$
p_3\approx\frac{20.086}{30.193}\approx0.6652
$$

最终结果为：

$$
\mathrm{Softmax}(1,2,3)
\approx(0.0900,0.2447,0.6652)
$$

可以解释为：模型认为三个类别的概率分别约为 9.00%、24.47% 和 66.52%。

---

## 4. Softmax 为什么能产生概率分布

### 4.1 输出一定为正数

指数函数对任何有限实数都有：

$$
e^{x_i}>0
$$

分母也是正数，所以：

$$
p_i>0
$$

因此 Softmax 不会产生负概率。对于有限输入，数学上的输出严格位于 0 和 1 之间，但在浮点计算中，特别小的概率可能因为精度限制显示为 0。

### 4.2 所有输出之和为 1

将所有概率相加：

$$
\sum_{i=1}^{K}p_i
=\sum_{i=1}^{K}\frac{e^{x_i}}{\sum_{j=1}^{K}e^{x_j}}
$$

由于每一项的分母相同，可以写为：

$$
\sum_{i=1}^{K}p_i
=\frac{\sum_{i=1}^{K}e^{x_i}}
{\sum_{j=1}^{K}e^{x_j}}
=1
$$

因此 Softmax 的输出满足概率分布的基本条件。

---

## 5. Softmax 的核心原理

### 5.1 保留类别的大小顺序

指数函数是严格递增函数。如果：

$$
x_a>x_b
$$

那么：

$$
e^{x_a}>e^{x_b}
$$

由于各类别使用相同的分母，因此：

$$
p_a>p_b
$$

Softmax 不会改变类别排名。Logit 最大的类别，经过 Softmax 后仍然拥有最大概率。因此：

```python
logits.argmax(dim=1)
```

和：

```python
torch.softmax(logits, dim=1).argmax(dim=1)
```

得到的预测类别相同。如果只需要最终类别，而不需要概率，通常没有必要额外计算 Softmax。

### 5.2 放大相对差异

指数函数会放大 logits 之间的差距。两个类别的概率之比为：

$$
\frac{p_a}{p_b}
=\frac{e^{x_a}}{e^{x_b}}
=e^{x_a-x_b}
$$

因此，概率比只由两个 logits 的差决定。

例如，当：

$$
x_a-x_b=2
$$

有：

$$
\frac{p_a}{p_b}=e^2\approx7.39
$$

也就是说，类别 $a$ 的概率大约是类别 $b$ 的 7.39 倍。

### 5.3 只关心相对差值

给所有 logits 同时加上同一个常数 $c$，Softmax 结果不会改变：

$$
\frac{e^{x_i+c}}{\sum_j e^{x_j+c}}
=\frac{e^{x_i}e^c}{e^c\sum_j e^{x_j}}
=\frac{e^{x_i}}{\sum_j e^{x_j}}
$$

因此：

$$
\mathrm{Softmax}(\mathbf{x}+c)
=\mathrm{Softmax}(\mathbf{x})
$$

例如：

```text
Softmax([1, 2, 3])
```

与：

```text
Softmax([101, 102, 103])
```

结果完全相同。Softmax 关注的是各类别分数之间的相对差异，而不是分数的绝对大小。

### 5.4 类别之间相互竞争

Softmax 的每个输出都依赖所有输入。某一个类别的 logit 增大时，它的概率通常会上升，同时其他类别的概率会下降，因为所有概率之和必须保持为 1。

这使 Softmax 适合“一个样本只属于一个类别”的互斥分类问题。

---

## 6. 数值稳定性

如果直接计算 $e^{x_i}$，当 logit 很大时可能发生浮点数溢出。例如，$e^{1000}$ 大到无法用常见浮点类型表示。

利用“同时加减常数不改变结果”的性质，可以先减去最大 logit：

$$
m=\max_j x_j
$$

再计算：

$$
p_i=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}
$$

例如：

$$
(1000,1001,1002)-(1002,1002,1002)=(-2,-1,0)
$$

于是只需要计算 $e^{-2}$、$e^{-1}$ 和 $e^0$，避免计算巨大的指数。PyTorch 会在内部使用数值稳定的实现，通常不需要手动减最大值。

当需要对数概率时，应直接使用 `torch.log_softmax` 或 `nn.LogSoftmax`，而不是先计算 Softmax 再取对数：

```python
log_probs = torch.log_softmax(logits, dim=1)
```

这种写法更稳定，也更高效。

---

## 7. PyTorch 中的 `dim` 参数

`dim` 指定沿哪个维度执行 Softmax，也就是“哪些元素需要相互竞争，并且归一化后总和为 1”。

假设张量形状为：

```text
[batch_size, num_classes]
```

例如：

```python
import torch
from torch import nn

logits = torch.tensor([
    [1.0, 2.0, 3.0],
    [2.0, 1.0, 0.0],
])

softmax = nn.Softmax(dim=1)
probs = softmax(logits)

print(probs)
print(probs.sum(dim=1))
```

输出：

```text
tensor([[0.0900, 0.2447, 0.6652],
        [0.6652, 0.2447, 0.0900]])
tensor([1., 1.])
```

`dim=1` 表示对每个样本的类别维度进行 Softmax，所以每一行的概率之和为 1。

如果使用 `dim=0`，则会在批次维度上归一化，使每一列之和为 1。这通常不是普通多分类任务需要的行为。

常见张量结构及典型选择如下：

| 输出形状 | 含义 | 常见 Softmax 维度 |
|---|---|---|
| `[num_classes]` | 单个样本的类别分数 | `dim=0` |
| `[batch_size, num_classes]` | 一批样本的类别分数 | `dim=1` 或 `dim=-1` |
| `[batch_size, sequence_length, vocabulary_size]` | 每个位置的词表分数 | `dim=-1` |
| `[batch_size, heads, query_length, key_length]` | 注意力分数 | 通常 `dim=-1` |

使用 `dim=-1` 表示沿最后一个维度计算，面对维度可能变化的张量时通常更直观。

---

## 8. Softmax 的主要使用场景

### 8.1 单标签多分类

这是 Softmax 最典型的使用场景。一个样本只能属于多个候选类别中的一个，例如：

- 图片属于猫、狗、鸟中的一个类别；
- 新闻属于体育、科技、财经中的一个类别；
- 手写数字属于 0 到 9 中的一个类别；
- 用户意图属于预先定义的某一个意图。

模型输出每个类别的 logit，推理时通过 Softmax 转换为类别概率：

```python
import torch

logits = model(inputs)                  # [batch_size, num_classes]
probs = torch.softmax(logits, dim=-1)  # 类别概率
predictions = probs.argmax(dim=-1)     # 概率最高的类别
```

### 8.2 Transformer 中的注意力机制

注意力机制先计算 query 与 key 的相似度分数，然后使用 Softmax 将这些分数转换为注意力权重：

$$
\mathrm{Attention}(Q,K,V)
=\mathrm{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Softmax 使每个 query 对所有 key 的注意力权重之和为 1，再用这些权重对 value 加权求和。

### 8.3 语言模型的下一个词预测

语言模型会为词表中的每个 token 输出一个 logit。Softmax 将整个词表上的 logits 转换为概率分布，从而可以：

- 选择概率最大的 token；
- 按概率随机采样 token；
- 使用 top-k、top-p 或温度调节控制生成结果。

### 8.4 知识蒸馏

知识蒸馏中，教师模型的 Softmax 分布不仅表示正确类别，还包含其他类别之间的相似关系。学生模型可以学习这个“软标签”分布。

### 8.5 强化学习中的离散动作策略

当智能体需要从有限个离散动作中选择一个动作时，可以对动作 logits 使用 Softmax，得到动作概率分布，再根据该分布采样动作。

---

## 9. 训练时与 `CrossEntropyLoss` 的关系

在 PyTorch 的单标签多分类训练中，通常使用：

```python
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(logits, targets)
```

这里应该将模型的原始 logits 直接传给 `CrossEntropyLoss`，不要提前调用 Softmax：

```python
# 正确
logits = model(inputs)
loss = criterion(logits, targets)
```

不要写成：

```python
# 不推荐
probs = torch.softmax(model(inputs), dim=1)
loss = criterion(probs, targets)
```

原因是 `CrossEntropyLoss` 在数学上等价于 `LogSoftmax` 加 `NLLLoss`，内部会直接从 logits 进行数值稳定的计算。提前执行 Softmax 会导致重复转换，可能降低数值稳定性，并使优化行为不符合损失函数的设计。

一个完整示例：

```python
import torch
from torch import nn

model = nn.Linear(10, 3)
criterion = nn.CrossEntropyLoss()

inputs = torch.randn(4, 10)
targets = torch.tensor([0, 2, 1, 0])

logits = model(inputs)          # 训练时保留原始 logits
loss = criterion(logits, targets)

# 推理或展示结果时再转换为概率
probs = torch.softmax(logits, dim=1)
predictions = logits.argmax(dim=1)
```

---

## 10. Softmax、Sigmoid 和 LogSoftmax 的区别

| 函数 | 输出特点 | 类别是否相互竞争 | 典型场景 |
|---|---|---|---|
| Softmax | 所有类别概率之和为 1 | 是 | 单标签多分类 |
| Sigmoid | 每个输出独立位于 0 到 1 | 否 | 二分类、多标签分类 |
| LogSoftmax | 输出 Softmax 概率的对数 | 是 | 对数概率、NLLLoss、数值稳定计算 |

### 单标签多分类

一个样本只能有一个类别，例如一张图片只能被标为“猫”“狗”或“鸟”之一。此时类别互斥，适合使用 Softmax。

### 多标签分类

一个样本可以同时具有多个标签，例如一张图片可以同时包含“人”“汽车”和“道路”。这些标签并不互斥，通常应对每个标签独立使用 Sigmoid，而不是 Softmax。

### 二分类

二分类既可以使用两个输出配合 Softmax，也可以使用一个输出配合 Sigmoid。实践中通常使用单个 logit 和 `BCEWithLogitsLoss`，因为它更简洁。

---

## 11. 温度参数 Temperature

Softmax 可以加入温度参数 $T$：

$$
p_i=\frac{e^{x_i/T}}{\sum_j e^{x_j/T}}
$$

温度控制概率分布的尖锐程度：

- $T=1$：标准 Softmax；
- $T>1$：缩小 logits 差距，分布更平滑、更均匀；
- $0<T<1$：放大 logits 差距，分布更尖锐、更集中；
- 当 $T\to0^+$ 时，概率逐渐集中到最大 logit 对应的类别；
- 当 $T\to\infty$ 时，概率逐渐趋向均匀分布。

PyTorch 示例：

```python
temperature = 2.0
probs = torch.softmax(logits / temperature, dim=-1)
```

温度 Softmax 常用于语言模型采样、知识蒸馏和概率校准。

---

## 12. 梯度与可训练性

Softmax 是可微函数，因此可以参与神经网络的反向传播。其偏导数为：

$$
\frac{\partial p_i}{\partial x_j}
=p_i(\delta_{ij}-p_j)
$$

其中，$\delta_{ij}$ 是 Kronecker delta：当 $i=j$ 时等于 1，否则等于 0。

分两种情况：

当 $i=j$ 时：

$$
\frac{\partial p_i}{\partial x_i}=p_i(1-p_i)
$$

这通常为正，说明提高自身 logit 会提高自身概率。

当 $i\ne j$ 时：

$$
\frac{\partial p_i}{\partial x_j}=-p_ip_j
$$

这通常为负，说明提高其他类别的 logit 会降低当前类别的概率。这也体现了 Softmax 中类别之间的竞争关系。

---

## 13. 常见误区与注意事项

### 13.1 Softmax 概率高，不代表模型一定可靠

Softmax 只根据 logits 的相对差距进行归一化。即使模型遇到了训练分布之外的数据，也可能产生非常集中的高概率。因此 Softmax 输出是模型的相对置信度表达，不自动等于真实正确率。

如果应用对概率可信度要求很高，可以考虑概率校准、温度缩放和不确定性估计等方法。

### 13.2 `dim` 选错会得到错误含义

Softmax 会沿指定维度归一化。对于 `[batch_size, num_classes]`，普通分类一般使用 `dim=1` 或 `dim=-1`。如果错误地使用 `dim=0`，不同样本会在批次维度上相互归一化。

可以通过下面的方式检查：

```python
probs = torch.softmax(logits, dim=-1)
print(probs.sum(dim=-1))
```

结果应在浮点误差范围内接近 1。

### 13.3 训练时不要在 `CrossEntropyLoss` 前使用 Softmax

`CrossEntropyLoss` 需要原始 logits，并在内部完成稳定的 LogSoftmax 计算。

### 13.4 多标签任务不要使用普通 Softmax

Softmax 强制所有类别概率之和为 1，适用于互斥类别。多标签任务中每个标签应独立判断，通常使用 Sigmoid 和 `BCEWithLogitsLoss`。

### 13.5 极端 logits 可能造成饱和

当一个 logit 远大于其他值时，Softmax 输出会非常接近 one-hot 分布。此时部分梯度可能变得很小，模型学习速度可能受到影响。合理的初始化、归一化、损失函数和训练策略有助于缓解问题。

---

## 14. 常用 PyTorch 写法

### 使用模块

```python
import torch
from torch import nn

x = torch.tensor([1.0, 2.0, 3.0])
softmax = nn.Softmax(dim=0)
probs = softmax(x)

print(probs)
print(probs.sum())
```

### 使用函数式接口

```python
probs = torch.softmax(x, dim=0)
```

### 批量分类推理

```python
with torch.no_grad():
    logits = model(inputs)
    probs = torch.softmax(logits, dim=-1)
    confidence, predicted_class = probs.max(dim=-1)
```

### 只获取预测类别

```python
predicted_class = logits.argmax(dim=-1)
```

因为 Softmax 不改变大小顺序，所以不需要先计算概率。

### 获取稳定的对数概率

```python
log_probs = torch.log_softmax(logits, dim=-1)
```

---

## 15. 总结

Softmax 的核心公式是：

$$
\boxed{
p_i=\frac{e^{x_i}}{\sum_j e^{x_j}}
}
$$

它的主要作用可以概括为：

1. 通过指数运算将任意实数转成正权重；
2. 通过除以总和，将权重归一化为总和为 1 的分布；
3. 保留原始 logits 的大小顺序；
4. 用指数关系放大类别之间的相对差异；
5. 让各类别形成竞争关系，适合互斥的单标签多分类；
6. 广泛应用于分类模型、注意力机制、语言模型、知识蒸馏和离散动作策略。

在 PyTorch 中使用 Softmax 时，最重要的实践原则是：正确选择 `dim`；训练多分类模型时把原始 logits 直接交给 `CrossEntropyLoss`；仅在推理、展示概率或算法明确需要概率分布时显式调用 Softmax。
