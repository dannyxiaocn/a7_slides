# 第1页

![img-0.jpeg](img-0.jpeg)

数据密集科学哲学硕士

提交时间：2025年6月22日晚11:59

# - 课程作业

A. Biguri博士，P. Cañizares博士，S. Mukherjee博士，M. Roberts博士，A. Breger博士

课程作业包含3个模块。最终报告应该解释你是如何得出答案结论的以及每一步的选择依据。报告应包括你的分析、展示生成的图表、讨论以及你在完成此作业过程中学到的内容。这必须针对每个模块分别完成。报告总分为[60]分。

课程作业将通过为你创建的GitLab仓库提交。你应该编写一份不超过3000字的报告来配合你编写的解决问题的软件。你应该将所有代码和报告放在此仓库中。报告应为PDF格式，放在名为"report"的文件夹中。你将获得仓库访问权限直到上述截止日期。在此之后你将失去访问权限，这将构成你作业的提交。

与课程作业相关的代码应用Python编写，并遵循研究计算模块定义的最佳软件开发实践。这应包括：

- 编写清晰可读的代码，符合通用代码规范并使用合适的构建管理工具。
- 提供适当的文档。
- 项目必须结构良好（合理的文件夹结构、README.md、许可证等），遵循标准最佳实践。

# 第2页

- 使用适当的版本控制最佳实践，包括开发和测试的分支，以及保护"main"分支的提交钩子。

# 第3页

# 模块1：经典图像处理

一位昆虫学家带着一系列蝴蝶图像（随课程作业附上）来找你，请你帮助他们清理在最新旅行中获取的数据。他们需要帮助完成一些任务。

![img-1.jpeg](img-1.jpeg)

他们只向你展示了数据集的一小部分，所以重要的是，无论你如何帮助他们，都必须是完全自动的，不依赖于手动选择。你可以假设任何对这些图像有效的精调值对所有新图像都有效，但过程应该是自动的。然而，这位昆虫学家爸爸曾经被ChatGPT侮辱过，所以严格禁止使用任何类型的机器学习解决方案来帮助他们。你只能使用来自skimage或你自己的经典图像处理技术。

1.1 - 颜色分类。他们拍摄了3种不同颜色的蝴蝶。你能按颜色对它们进行分类吗？每种类型应该有4张图片。

1.2 - 背景移除。他们对背景不感兴趣，所以你能请移除背景只保留蝴蝶吗？

1.3 - 收藏展示。如果我们能制作一个图像，将每种类型的所有蝴蝶用单一背景（你选择）显示，那就太好了。注意昆虫学家非常挑剔，所以他们希望能够放置任意数量的蝴蝶图像（我们现在有4张，但你的函数应该接受每类$N$张图像），还要能够选择最终图像的分辨率（以像素为单位）。你可以选择最合理的排列方式。

1.4 - 异类识别。你观察蝴蝶的丰富经验非常有用！你注意到在每个类别的4张图像中，有一只蝴蝶与其他蝴蝶是不同的种类！你能找到一种方法自动将它与其他3只区分开来吗？

请记住向昆虫学家解释为什么你选择了所使用的方法，而不是其他解决方案。

注意：此练习没有唯一正确答案，也没有用于比较结果的标准答案。这是关于你探索方法、仔细选择它们并展示你的图像处理技能。如果你能说服我们你的方法是解决任务的合理方式，即使结果不完美，你也能获得满分。

# 第4页

# 模块2：逆问题的数据驱动正则化

在坚持之后，你说服了昆虫学家使用一些机器学习并不是那么糟糕。特别是考虑到一些昆虫学家的图像不仅有噪声，而且真的很模糊，是用坏掉的相机拍摄的。昆虫学家同意使用机器学习，但只有在严格的数学框架下适当使用，比如PnP。

因此，你将利用强大的图像去噪器（基于预训练深度神经网络）来解决更具挑战性的逆问题，如图像去模糊和修复，希望能恢复蝴蝶图像。

## 交替方向乘子法（ADMM）

考虑变分问题

$$
\min _{x} \frac{1}{2}\|A x-y\|_{2}^{2}+g(x)
$$

用于图像重建。注意(1)可以重新表述为

$$
\min _{x, v} \frac{1}{2}\|A x-y\|_{2}^{2}+g(v) \text { subject to } x=v
$$

(3)的增广拉格朗日函数为

$$
L_{\eta}(x, v, u):=\frac{1}{2}\|A x-y\|_{2}^{2}+g(v)+u^{\top}(x-v)+\frac{\eta}{2}\|x-v\|_{2}^{2}
$$

其中$\eta>0$，$u$表示对偶变量。然后，回想(1)的ADMM迭代更新包含以下三个步骤：

$$
\begin{aligned}
& x \text{-更新: } x^{k+1}=\underset{x}{\arg \min } L_{\eta}\left(x, v^{k}, u^{k}\right)=\left(A^{\top} A+\eta I\right)^{-1}\left(A^{\top} y+\eta v^{k}-u^{k}\right) \\
& v \text{-更新: } v^{k+1}=\underset{v}{\arg \min } L_{\eta}\left(x^{k+1}, v, u^{k}\right)=\operatorname{prox}_{g}^{\frac{1}{\eta}}\left(x^{k+1}+\frac{1}{\eta} u^{k}\right), \text{和} \\
& u \text{-更新: } u^{k+1}=u^{k}+\eta\left(x^{k+1}-v^{k+1}\right)
\end{aligned}
$$

这里，$\operatorname{prox}_{\gamma}^{g}(z)=\underset{v}{\arg \min } \frac{1}{2}\|v-z\|_{2}^{2}+\gamma g(v)$。在重新定义对偶变量为$\frac{1}{\eta} u$后，我们注意到ADMM算法的缩放形式包含以下迭代更新：

$$
\begin{aligned}
& x \text{-更新: } x^{k+1}=\left(A^{\top} A+\eta I\right)^{-1}\left(A^{\top} y+\eta\left(v^{k}-u^{k}\right)\right) \\
& v \text{-更新: } v^{k+1}=\operatorname{prox}_{g}^{\frac{1}{\eta}}\left(x^{k+1}+u^{k}\right), \text{和} \\
& u \text{-更新: } u^{k+1}=u^{k}+\left(x^{k+1}-v^{k+1}\right)
\end{aligned}
$$

# 第5页

# 即插即用（PnP）-ADMM

在ADMM算法的PnP变体中，我们用现成的图像去噪器$D$替换$v$-更新步骤中的近端算子。我们将使用预训练的深度去噪器（基于U-net架构，可在https://drive.google.com/file/d/1FFuauq-PUjY_kG3iiiHfDpHcG4Srl8mQ/view?usp=sharing获得）来实现此目的。使用与https://github.com/facebookresearch/fastMRI/blob/main/fastmri/models/unet.py中相同的U-net实现，并使用以下代码行加载预训练模型：

```python
model = Unet(3, 3, chans=64).to(device)
model.load_state_dict(torch.load('denoiser.pth', map_location=device))
```

你可以使用模块1中的一张蝴蝶图像作为本模块练习的标准答案，以模拟昆虫学家遇到的图像退化。

PnP-ADMM算法总结如下：

1. 初始化：$x, u, v=0$
2. 迭代直到收敛：

$$
\begin{aligned}
& x \leftarrow\left(A^{\top} A+\eta I\right)^{-1}\left(A^{\top} y+\eta(v-u)\right) \\
& v \leftarrow D(x+u) \\
& u \leftarrow u+(x-v)
\end{aligned}
$$

$x$-更新步骤通过使用共轭梯度（CG）方法求解线性系统$C x=d$来实现，其中$C:=A^{\top} A+\eta I$和$d=A^{\top} y+\eta(v-u)$。实现CG的代码如下。

```python
def conjugate_gradient(A, b, x0, max_iter, tol):
    """CG用于求解Ax=b。
    这里，参数A是一个返回Ax的函数"""
    x = x0
    r = b-A(x)
    d = r
    for _ in range(max_iter):
        z = A(d)
        rr = torch.sum(r**2)
        alpha = rr/torch.sum(d*z)
        x += alpha*d
        r -= alpha*z
        if torch.norm(r)/torch.norm(b) < tol:
            break
        beta = torch.sum(r**2)/rr
        d = r + beta*d
    return x
```

# 第6页

练习2.1（去模糊）在图像去模糊中，我们寻求从模糊测量$y=A x$中恢复清晰图像$x$，其中$A$表示模糊算子。

1. 考虑图像去模糊问题，其中$A$对应于大小为$p \times p$的运动模糊核（即，大小为$p \times p$的核，核的所有条目都等于$\frac{1}{p^{2}}$）。使用PnP-ADMM找到不同$p$值（比如$p=7,13,17$）的重建图像（相对于标准答案图像）的均方误差（MSE）。实现$A$和$A^{\top}$的模块已提供给你。你可以将输入图像归一化到范围$[0,1]$并选择$\eta=10^{-4}$。你必须编写一个实现PnP-ADMM的函数，并在最后将其输出裁剪到范围$[0,1]$。

以下函数实现对应于给定模糊核的前向（$A$）和伴随（$A^{\top}$）算子。

```python
def conv2d_block(kernel, channels, p, device, stride=1):
    """从2D核返回nn.Conv2d和nn.ConvTranspose2d模块，使得
    nn.ConvTranspose2d是nn.Conv2d的伴随算子
    参数：
        kernel: 2D核，p x p是核大小
        channels: 图像通道数"""
    kernel_size = kernel.shape
    kernel = kernel/kernel.sum()
    kernel = kernel.repeat(channels, 1, 1, 1)
    filter = nn.Conv2d(
        in_channels=channels, out_channels=channels,
        kernel_size=kernel_size, groups=channels, bias=False, stride=stride,
        padding=p//2
    )
    filter.weight.data = kernel
    filter.weight.requires_grad = False
    filter_adjoint = nn.ConvTranspose2d(
        in_channels=channels, out_channels=channels,
        kernel_size=kernel_size, groups=channels, bias=False, stride=stride,
        padding=p//2,
    )
    filter_adjoint.weight.data = kernel
    filter_adjoint.weight.requires_grad = False
    return filter.to(device), filter_adjoint.to(device)
```

2. 将核大小设为$p=13$。向模糊图像$y$添加零均值和$\sigma=0.01$的高斯噪声，并运行PnP-ADMM算法进行重建。重建图像与"测量中无噪声"情况在MSE方面如何比较？写出你对结果的解释。

# 第7页

练习2.2（修复）图像修复的任务是恢复图像中的缺失像素。这里，前向算子$A$涉及应用随机二进制掩码$M$，其中零表示缺失像素，损坏的输入图像由$y=M \odot X$给出，其中$\odot$表示掩码$M$与图像$X$的逐元素相乘。注意这个算子可以等价地表示为$y=A x$，其中$A$是对角矩阵，在缺失像素位置为零，在其余位置为一，$x$表示向量化形式的图像。这个前向算子的伴随显然是$A^{\top}=A$，可以通过在参数上应用相同的掩码来实现。你可以使用以下修复问题的前向和伴随算子实现。

```python
# 修复
channels = 3 # 对于彩色图像
mask = torch.rand(1,channels,h,w).to(device) # 3个通道，有h×w像素
mask = mask < 0.4 # 这意味着大约40%的像素将被保留
def forward(x): # 确保图像大小为1×3×h×w
    return x*mask # 与图像逐元素相乘
adjoint = forward
```

1. 应用PnP-ADMM算法解决40%、60%和80%缺失像素的修复问题，并比较相应重建图像相对于标准答案的MSE。

2. 回想在课堂上，我们学习了PnP算法的变体（称为去噪正则化（RED）），其中正则化器$\rho(x)$直接使用预训练去噪器$D(x)$形成为$\rho(x)=\frac{1}{2} x^{\top}(x-D(x))$。如果去噪器是雅可比对称且局部齐次的，则该正则化器的梯度由$\nabla \rho(x)=x-D(x)$给出，即去噪残差。

(a) 假设上述正则化器梯度表达式，实现梯度下降算法来最小化$J(x)=\frac{1}{2}\|y-A x\|_{2}^{2}+\lambda \rho(x)$，其中$\lambda=0.1$，$A$是对应于60%像素随机置零情况的修复掩码。使用步长$\eta=1$。我们将这个版本的PnP称为PnP-RED。

(b) 在重建图像的MSE方面比较PnP-ADMM和PnP-RED的输出。

(c) 对于此练习中选择的去噪器，设置$\nabla \rho(x)=x-D(x)$是否正确？

练习2.3：调查如果运行更多PnP-ADMM迭代，MSE是增加还是减少。绘制重建图像的MSE作为PnP-ADMM迭代的函数。解释/解释随着迭代次数的变化。

# 第8页

迭代。提出一种策略来缓解你在图像质量中看到的影响。

# 模块3：陷阱和挑战

## 练习3.1 图像质量评估

昆虫学家对你迄今为止的结果印象深刻，但他们并不完全相信你说最好的图像（MSE更低的图像）实际上是最好的。他们希望你对哪些图像真正是最好的进行更详细的分析（展览对昆虫学家非常重要！）。

[3.1.a] 讨论为什么PSNR和SSIM可能不足以作为图像质量度量来评估模块2（去模糊和修复）中结果的性能。哪些全参考（FR）和无参考（NR）IQA度量对这些任务有用？用你选择的FR和NR-IQA度量集重新评估你的结果，并将其与PSNR和SSIM的评估进行比较。

[3.1.b] 从模块1的数据集中选择一只蝴蝶，创建退化版本，其中(1) PSNR和(2) SSIM给出相同的评估，尽管退化类型差异很大。当背景像素被移除时，这些值会发生什么，你能推断出什么？

## 练习3.2 构建图像分析ML/DL系统的陷阱和挑战

为了说服昆虫学家在成像中使用机器学习并不是那么糟糕，你将向他们展示你对工具的掌握，使用MNIST数据集。你将向他们展示你对成像中ML挑战的理解程度。可以在这里找到笔记本和数据。打开ex3.2.ipynb并用GPU从头到尾运行。此代码使用MNIST数据集训练神经网络（case=a）进行数字分类。它应该在大约20-40个周期内收敛。

[3.2.a] 解释你看到的内容，对于case=a，关于：

- 这段代码在做什么。
- 如何缓解常见陷阱。
- 如何使用数据分区。
- 所采用方法可能引入的偏差。
- 模型性能和训练效果，特别是什么指标最合适。
- 给出一些改进训练的想法，包括可能修改代码 - 为它们提供解释和动机。

# 第9页

[3.2.b] 为case=a实施你建议的更改并重新训练模型以实现接近零的损失。对于两个表现最差的类别，你能识别性能低的关键原因以及如何使模型对它们具有鲁棒性吗？

[3.2.c] 现在考虑case=b并从头到尾运行笔记本。解释两种情况之间的差异，并详细说明你是否预期case=b模型会比case=a表现更差、更好或相同。描述你认为case=b代码需要的任何更改以提高表现最差类别的性能，并报告你的最佳模型表现如何。

论文结束 