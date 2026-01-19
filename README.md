SemEval-2026 POLAR (Task 9) - Project Framework
📑 项目概述 (Project Overview)
本项目旨在解决 SemEval-2026 Task 9: POLAR 任务，即在 22 种语言背景下对文本的极化现象（Polarization）进行检测与分析。

我们采用了 “路径 C：混合增强流水线 (Hybrid Augmentation Pipeline)”。核心策略是：利用大语言模型 (GPT-4) 的推理能力生成高质量的翻译、思维链 (CoT) 和辅助特征，并将这些知识“蒸馏”到以 XLM-RoBERTa-Large 为底座的高性能分类器中，从而在极有限的推理资源下实现跨语言的高精度识别。

🎯 任务分解 (Task Decomposition)
Subtask 1 (Backbone): 极化二分类 (Binary Classification)。

Subtask 2 (Factors): 5 类话题维度多标签分类 (Multi-label)。

Subtask 3 (Strategies): 6 类修辞策略多标签分类 (Multi-label)。