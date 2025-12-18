# 🖼️ CIFAR-10 Image Classification with Advanced ResNet-18

一个使用 **PyTorch** 实现的深度计算机视觉项目，在经典的CIFAR-10数据集上，通过系统性的**数据增强、现代训练技巧和精细调优**，使用修改后的ResNet-18模型达到了**接近95%**的测试准确率。本项目是深度学习工程能力的综合实践。

## 📊 项目概述与成果

本项目目标是在CIFAR-10数据集（包含10个类别的6万张32x32彩色图像）上实现高精度图像分类。

**核心成果**:
*   **最终测试准确率**: **94.99%**
*   **训练策略**: 综合应用了数据增强、混合精度训练、标签平滑、组合学习率调度（线性预热+余弦退火）等现代深度学习技巧。
*   **代码规范**: 包含完整的可复现性设置（固定随机种子）、训练过程可视化、以及最佳模型保存与加载机制。

## ✨ 核心技术亮点

### 1. 系统化的数据增强与预处理
为提升模型泛化能力，精心设计了一组强化的数据增强流程，有效增加了数据的多样性：
```python
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 色彩扰动
    transforms.RandomRotation(degrees=20), # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize(...)
])
```

### 2. 针对性的模型架构修改
没有直接使用原生ResNet-18，根据CIFAR-10图像尺寸较小的特点进行了适配性修改：
*   **修改首层卷积**: 将 `kernel_size=7, stride=2` 改为 `kernel_size=3, stride=1, padding=1`，避免在早期过度下采样丢失信息。
*   **移除早期池化层**: 将 `maxpool` 层替换为 `Identity()`，进一步保留空间细节。
*   **调整全连接层**: 将最终的 `fc` 层输出维度改为10，以匹配CIFAR-10的类别数。

### 3. 先进的训练技巧集成
本项目集成了多项被证明能有效提升模型性能与训练效率的现代技术：
*   **混合精度训练 (AMP)**: 使用 `autocast` 与 `GradScaler`，在保持精度的同时**显著加快训练速度并降低GPU显存占用**。
*   **标签平滑 (Label Smoothing)**: 在交叉熵损失函数中设置 `label_smoothing=0.1`，缓解模型对训练标签的“过度自信”，**提升模型的泛化能力和校准度**。
*   **组合学习率调度**:
    *   **线性预热 (Linear Warm-up)**: 在前20%的epoch中，学习率从初始值的1%线性增长至100%，确保训练初期稳定性。
    *   **余弦退火 (Cosine Annealing)**: 在预热结束后，使用余弦函数将学习率平滑下降至0，有助于模型收敛到更优的局部最小点。
*   **自动最佳模型保存**: 在训练过程中持续监控测试集准确率，并自动保存性能最佳的模型权重。

### 4. 严谨的实验管理与可视化
*   **可复现性**: 通过 `set_seed()` 函数固定所有随机种子(`random`, `numpy`, `torch`, `cuda`)，确保实验结果可完全复现。
*   **全面监控**: 记录并绘制了训练/测试的**损失曲线、准确率曲线以及学习率变化曲线**，并对最后50个epoch进行重点展示，便于分析模型收敛情况。

## 🛠️ 环境依赖与运行

### 环境配置
建议使用Python 3.8+及以下主要库：
```bash
pip install torch torchvision numpy matplotlib pandas scikit-learn
```

### 运行项目
1.  **数据准备**: 确保CIFAR-10数据集已下载并存放在 `./data` 目录下，或修改代码中的 `root` 路径指向正确位置。本项目数据已存在于 `/kaggle/input/cnn-cifar-10`。
2.  **执行训练**: 直接运行完整的Jupyter Notebook或Python脚本。训练过程将自动开始，并打印每个epoch的损失和准确率。
3.  **查看结果**: 训练结束后，最佳模型权重将保存为 `resnet18_cifar10_best.pth`，最终在测试集上的预测结果将保存为 `result.csv`。所有可视化图表将在运行过程中显示。

## 📈 训练过程与结果分析

从提供的日志和可视化图表可以观察到：
1.  **稳定收敛**: 训练损失与测试损失均稳步下降，未出现显著过拟合迹象（训练准确率最终接近100%，测试准确率稳定在95%左右）。
2.  **调度器效果**: 学习率曲线清晰展示了线性预热和后续余弦退火的过程，验证了调度策略按预期工作。
3.  **高性能结果**: 最终 **94.99%** 的测试准确率表明，所采用的模型修改和训练策略组合对于CIFAR-10数据集非常有效。

## 🔮 未来改进方向

虽然当前模型已达到优秀性能，仍有进一步探索的空间：
1.  **模型架构**: 尝试更深的ResNet变体（如ResNet-34, ResNet-50），或更现代的架构如EfficientNet、Vision Transformer (ViT)。
2.  **数据增强**: 引入CutMix、MixUp、AutoAugment等更复杂的数据增强策略。
3.  **优化与正则化**: 测试不同的优化器（如AdamW），调整权重衰减策略，或尝试Stochastic Depth等正则化方法。
4.  **超参数调优**: 使用自动化工具（如Optuna, Ray Tune）对学习率、批大小、增强参数等进行系统调优。
5.  **集成与后处理**: 使用多个模型的预测结果进行集成，或对模型输出进行温度缩放等后校准。

## 👨‍💻 作者

**yunna hua**
*   Kaggle: [@yunnahua](https://www.kaggle.com/yunnahua)
*   GitHub: [huayunna3]
*   专注于通过扎实的工程实践解决计算机视觉与深度学习问题。

## 📄 许可

此项目基于 Apache 2.0 许可证开源。

---
**如果这个项目对您有启发或帮助，欢迎 Star ⭐ 本项目！任何问题或建议，也欢迎提出Issue或Pull Request。**
