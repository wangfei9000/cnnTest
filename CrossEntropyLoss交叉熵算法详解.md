# CrossEntropyLoss 交叉熵算法详解

## 1. 一句话理解

`CrossEntropyLoss` 用于衡量模型给出的预测概率分布与真实标签之间的差异。

对于普通单标签分类，它最核心的含义是：

> 取出模型给正确类别分配的概率，计算这个概率的负对数。

正确类别的预测概率越大，损失越小；概率越小，损失越大。

---

## 2. 交叉熵数学公式

真实概率分布记为：

$$
q=(q_1,q_2,\ldots,q_C)
$$

模型预测的概率分布记为：

$$
p=(p_1,p_2,\ldots,p_C)
$$

交叉熵定义为：

$$
\boxed{H(q,p)=-\sum_{i=1}^{C}q_i\log p_i}
$$

其中：

- $C$：类别数量；
- $q_i$：真实分布认为样本属于第 $i$ 类的概率；
- $p_i$：模型预测样本属于第 $i$ 类的概率；
- $\log$：深度学习中通常指自然对数 $\ln$。

两个分布都满足：

$$
q_i\ge 0,\qquad p_i\ge 0
$$

$$
\sum_{i=1}^{C}q_i=1,\qquad \sum_{i=1}^{C}p_i=1
$$

### 2.1 普通分类中的简化

普通单标签分类通常使用 one-hot 真实分布。假设正确类别是 $y$：

$$
q_i=
\begin{cases}
1,&i=y\\
0,&i\ne y
\end{cases}
$$

代入交叉熵公式：

$$
H(q,p)=-(0\log p_1+\cdots+1\log p_y+\cdots+0\log p_C)
$$

因此：

$$
\boxed{L=-\log p_y}
$$

$p_y$ 就是模型分配给正确类别的概率。

---

## 3. 为什么使用负对数

因为概率 $p$ 位于 $(0,1]$，所以 $\log p\le 0$。加上负号后，损失成为非负数。

| 正确类别概率 $p_y$ | 损失 $-\ln p_y$ |
|---:|---:|
| $1.00$ | $0$ |
| $0.90$ | $0.105$ |
| $0.50$ | $0.693$ |
| $0.10$ | $2.303$ |
| $0.01$ | $4.605$ |

因此：

- $p_y\to1$ 时，$L\to0$；
- $p_y\to0$ 时，$L\to+\infty$；
- 模型越自信地预测错误，惩罚越严重。

计算损失不只是为了判断结果好坏，更重要的是得到可求导的目标函数。反向传播可以利用它的梯度，判断每个模型参数应该向什么方向调整、调整多少。

---

## 4. 从 logits 到概率

神经网络最后一层通常输出的不是概率，而是原始分数 logits：

$$
z=(z_1,z_2,\ldots,z_C)
$$

Softmax 将 logits 转换为概率：

$$
\boxed{p_i=\frac{e^{z_i}}{\sum_{j=1}^{C}e^{z_j}}}
$$

Softmax 保证：

$$
0<p_i<1,\qquad \sum_i p_i=1
$$

把 Softmax 代入单标签交叉熵：

$$
L=-\log\left(\frac{e^{z_y}}{\sum_{j=1}^{C}e^{z_j}}\right)
$$

根据对数运算规则，可改写为：

$$
\boxed{L=\log\left(\sum_{j=1}^{C}e^{z_j}\right)-z_y}
$$

因此 `CrossEntropyLoss` 可以理解为：

$$
\boxed{\text{CrossEntropyLoss}=\text{LogSoftmax}+\text{NLLLoss}}
$$

PyTorch 会采用数值更稳定的方式完成这些运算，所以应直接传入 logits，不要提前调用 `softmax`。

---

## 5. 三分类完整手算

假设任务有三个类别：

- 类别 0：猫；
- 类别 1：狗；
- 类别 2：鸟。

模型对一个样本输出 logits：

$$
z=[2,1,0]
$$

真实类别是“猫”，类别编号为 0，所以：

$$
y=0,\qquad q=[1,0,0]
$$

### 第一步：计算指数

$$
e^2\approx7.389056
$$

$$
e^1\approx2.718282
$$

$$
e^0=1
$$

### 第二步：计算 Softmax 分母

$$
S=e^2+e^1+e^0
$$

$$
S\approx7.389056+2.718282+1=11.107338
$$

### 第三步：计算每个类别的概率

$$
p_0=\frac{e^2}{S}=\frac{7.389056}{11.107338}\approx0.665241
$$

$$
p_1=\frac{e^1}{S}=\frac{2.718282}{11.107338}\approx0.244728
$$

$$
p_2=\frac{e^0}{S}=\frac{1}{11.107338}\approx0.090031
$$

所以模型的预测概率分布为：

$$
p\approx[0.665241,0.244728,0.090031]
$$

检查概率之和：

$$
0.665241+0.244728+0.090031=1
$$

### 第四步：计算交叉熵

完整公式为：

$$
H(q,p)=-\sum_{i=1}^{3}q_i\log p_i
$$

代入 $q=[1,0,0]$：

$$
H(q,p)
=-[1\ln(0.665241)+0\ln(0.244728)+0\ln(0.090031)]
$$

只有正确类别对应的项被保留：

$$
H(q,p)=-\ln(0.665241)
$$

$$
\boxed{H(q,p)\approx0.407606}
$$

也可以直接从 logits 计算：

$$
L=\ln(e^2+e^1+e^0)-2
$$

$$
L=\ln(11.107338)-2
$$

$$
L\approx2.407606-2=0.407606
$$

两种计算方法结果完全一致。

### 如果真实类别是鸟

若真实类别改为类别 2，则：

$$
q=[0,0,1]
$$

正确类别的预测概率只有：

$$
p_2\approx0.090031
$$

损失为：

$$
L=-\ln(0.090031)\approx2.407606
$$

对比可见：

| 真实类别 | 正确类别概率 | 损失 |
|---|---:|---:|
| 猫 | $0.665241$ | $0.407606$ |
| 鸟 | $0.090031$ | $2.407606$ |

---

## 6. 多个样本时如何计算

假设 batch 中有 $N$ 个样本，第 $n$ 个样本的正确类别为 $y_n$，则每个样本的损失是：

$$
L_n=-\log\frac{e^{z_{n,y_n}}}{\sum_{c=1}^{C}e^{z_{n,c}}}
$$

PyTorch 默认使用平均损失：

$$
\boxed{L=\frac{1}{N}\sum_{n=1}^{N}L_n}
$$

对应：

```python
nn.CrossEntropyLoss(reduction="mean")
```

也可以选择：

```python
# 返回每个样本各自的损失
nn.CrossEntropyLoss(reduction="none")

# 将所有样本的损失相加
nn.CrossEntropyLoss(reduction="sum")
```

---

## 7. Python 手动计算

### 7.1 使用 Python 标准库手算

```python
import math

logits = [2.0, 1.0, 0.0]
target = 0

# 1. 计算 exp(logit)
exp_values = [math.exp(x) for x in logits]

# 2. 计算 Softmax 分母
denominator = sum(exp_values)

# 3. 得到概率分布
probabilities = [x / denominator for x in exp_values]

# 4. 取正确类别概率并计算负对数
correct_probability = probabilities[target]
loss = -math.log(correct_probability)

print("exp:", exp_values)
print("probabilities:", probabilities)
print("correct probability:", correct_probability)
print("loss:", loss)
```

输出约为：

```text
exp: [7.389056, 2.718282, 1.0]
probabilities: [0.665241, 0.244728, 0.090031]
correct probability: 0.665241
loss: 0.407606
```

### 7.2 使用 PyTorch 分步骤验证

```python
import torch

logits = torch.tensor([[2.0, 1.0, 0.0]])
target = torch.tensor([0])

probabilities = torch.softmax(logits, dim=1)
correct_probability = probabilities[0, target[0]]
manual_loss = -torch.log(correct_probability)

print("概率:", probabilities)
print("正确类别概率:", correct_probability)
print("手算损失:", manual_loss)
```

### 7.3 使用 `nn.CrossEntropyLoss`

```python
import torch
import torch.nn as nn

logits = torch.tensor([[2.0, 1.0, 0.0]])
target = torch.tensor([0])

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, target)

print(loss)
# tensor(0.4076)
```

注意：传入 `CrossEntropyLoss` 的是原始 logits，而不是 Softmax 后的概率。

```python
# 正确
loss = loss_fn(logits, target)

# 不推荐：CrossEntropyLoss 内部已经包含相关转换
probabilities = torch.softmax(logits, dim=1)
loss = loss_fn(probabilities, target)
```

### 7.4 验证等价关系

```python
import torch
import torch.nn as nn

logits = torch.tensor([[2.0, 1.0, 0.0]])
target = torch.tensor([0])

loss1 = nn.CrossEntropyLoss()(logits, target)

log_probabilities = torch.log_softmax(logits, dim=1)
loss2 = nn.NLLLoss()(log_probabilities, target)

print(loss1)  # tensor(0.4076)
print(loss2)  # tensor(0.4076)
```

---

## 8. 为什么损失可以训练模型

“正确类别概率越高越好”只是目标描述，不能直接告诉神经网络的每个参数应该如何变化。损失函数把这个目标变成一个连续、可求导的数值。

Softmax 与交叉熵组合后，对第 $k$ 个 logit 的导数非常简洁：

$$
\boxed{\frac{\partial L}{\partial z_k}=p_k-q_k}
$$

在手算例子中：

$$
p=[0.665241,0.244728,0.090031]
$$

$$
q=[1,0,0]
$$

所以：

$$
\frac{\partial L}{\partial z}=p-q
$$

$$
=[0.665241-1,0.244728-0,0.090031-0]
$$

$$
=\boxed{[-0.334759,0.244728,0.090031]}
$$

梯度下降使用：

$$
z_k^{\text{new}}=z_k-\eta\frac{\partial L}{\partial z_k}
$$

因此：

- 正确类别“猫”的梯度为负，更新后它的 logit 会升高；
- 错误类别“狗”和“鸟”的梯度为正，更新后它们的 logits 会降低；
- 最终正确类别的 Softmax 概率会上升。

这就是计算损失的真正用途：不仅衡量错得多严重，还通过梯度指导参数更新。

---

## 9. 信息论解释

### 9.1 信息量

一个概率为 $p_i$ 的事件发生时，它携带的信息量定义为：

$$
\boxed{I(i)=-\log p_i}
$$

直觉上：

- 经常发生的事件不令人意外，信息量小；
- 很少发生的事件一旦发生，令人意外，信息量大。

例如：

$$
-\ln(0.9)\approx0.105
$$

$$
-\ln(0.01)\approx4.605
$$

### 9.2 信息熵

如果事件真实按照分布 $q$ 出现，其平均信息量称为信息熵：

$$
\boxed{H(q)=-\sum_i q_i\log q_i}
$$

它描述真实分布自身的不确定程度。

### 9.3 交叉熵

数据实际上按照 $q$ 出现，但我们使用模型分布 $p$ 来估计每个事件的信息量，此时平均信息量就是：

$$
\boxed{H(q,p)=-\sum_iq_i\log p_i}
$$

从编码角度理解：

- $q$ 是数据真实出现的频率；
- $p$ 是我们设计编码时采用的概率；
- $-\log p_i$ 是按照 $p$ 编码事件 $i$ 所需的长度；
- 交叉熵是使用错误分布 $p$ 编码真实数据时的平均编码长度。

预测分布越接近真实分布，编码越有效，交叉熵越小。

### 9.4 交叉熵、信息熵和 KL 散度

三者满足：

$$
\boxed{H(q,p)=H(q)+D_{\mathrm{KL}}(q\|p)}
$$

其中 KL 散度为：

$$
D_{\mathrm{KL}}(q\|p)
=\sum_iq_i\log\frac{q_i}{p_i}
$$

并且：

$$
D_{\mathrm{KL}}(q\|p)\ge0
$$

因为训练数据的真实分布 $q$ 固定，所以 $H(q)$ 也是常量。因此：

$$
\boxed{
\text{最小化交叉熵}
\iff
\text{最小化 }D_{\mathrm{KL}}(q\|p)
}
$$

也就是说，最小化交叉熵是在推动模型预测分布 $p$ 接近真实分布 $q$。

### 9.5 为什么 one-hot 情况的最小值是 0

若真实分布为：

$$
q=[1,0,0]
$$

模型完美预测时：

$$
p=[1,0,0]
$$

于是：

$$
H(q,p)=-\ln1=0
$$

但如果真实标签是软分布，例如 $q=[0.7,0.2,0.1]$，即使模型做到 $p=q$，交叉熵也等于 $H(q)$，通常不会是 0，因为真实分布本身包含不确定性。

---

## 10. PyTorch 中的输入要求

普通分类通常使用：

```text
logits.shape = (N, C)
target.shape = (N,)
```

其中：

- $N$ 是 batch size；
- $C$ 是类别数；
- `logits` 是浮点数；
- 类别索引形式的 `target` 通常必须是 `torch.long`；
- target 的类别编号范围是 $[0,C)$。

例如：

```python
logits = torch.randn(32, 10)                  # 32 个样本，10 类
target = torch.randint(0, 10, (32,)).long()  # 每个样本一个类别编号

loss = nn.CrossEntropyLoss()(logits, target)
```

语义分割通常使用：

```text
logits.shape = (N, C, H, W)
target.shape = (N, H, W)
```

`target` 不包含类别维度，每个像素直接保存正确类别编号。

---

## 11. CrossEntropyLoss 与 BCEWithLogitsLoss

### 单标签多分类

一个样本只能属于一个类别，例如“猫、狗、鸟”三选一：

```python
nn.CrossEntropyLoss()
```

Softmax 使所有类别概率之和为 1。

### 多标签分类

一个样本可以同时具有多个标签，例如一张图中同时存在“人、汽车、道路”：

```python
nn.BCEWithLogitsLoss()
```

它对每个类别分别使用 Sigmoid，各标签之间不互斥。

---

## 12. 常见错误

### 错误一：先做 Softmax

```python
# 错误或不推荐
probabilities = torch.softmax(logits, dim=1)
loss = nn.CrossEntropyLoss()(probabilities, target)
```

应直接写：

```python
loss = nn.CrossEntropyLoss()(logits, target)
```

### 错误二：传入 argmax 结果

```python
# 错误
predicted = logits.argmax(dim=1)
loss = nn.CrossEntropyLoss()(predicted, target)
```

`argmax` 会丢失各类别的分数，而且不能提供正常训练所需的连续梯度。正确做法是用 logits 计算 loss，用 `argmax` 计算准确率：

```python
loss = nn.CrossEntropyLoss()(logits, target)
predicted = logits.argmax(dim=1)
accuracy = (predicted == target).float().mean()
```

### 错误三：标签编号从 1 开始

若类别数是 3，正确编号是：

```text
0, 1, 2
```

不是：

```text
1, 2, 3
```

---

## 13. 总结

交叉熵的一般公式：

$$
\boxed{H(q,p)=-\sum_iq_i\log p_i}
$$

普通 one-hot 分类中的简化公式：

$$
\boxed{L=-\log p_y}
$$

从 logits 直接计算：

$$
\boxed{L=\log\left(\sum_j e^{z_j}\right)-z_y}
$$

梯度：

$$
\boxed{\frac{\partial L}{\partial z_i}=p_i-q_i}
$$

信息论分解：

$$
\boxed{H(q,p)=H(q)+D_{\mathrm{KL}}(q\|p)}
$$

因此，`CrossEntropyLoss` 同时完成了三件事：

1. 衡量模型预测得有多差；
2. 通过梯度告诉模型参数如何调整；
3. 从信息论角度推动预测分布接近真实分布。

