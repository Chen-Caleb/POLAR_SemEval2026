import torch
import torch.nn as nn


class MultiSampleDropoutHead(nn.Module):
    def __init__(self, hidden_size, num_labels, num_dropout=5, dropout_prob=0.1):
        super().__init__()
        self.num_dropout = num_dropout

        # 定义多个并行的 Dropout 层
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_prob) for _ in range(num_dropout)
        ])

        # 共享同一个分类线性层
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        # x 形状: [batch_size, hidden_size]

        # 核心逻辑：将输入通过每一个 Dropout 层，然后通过分类器，最后取平均
        # 这种方式在训练时能提供更稳定的梯度
        logits = torch.stack([
            self.classifier(dropout(x)) for dropout in self.dropouts
        ], dim=0).mean(dim=0)

        return logits