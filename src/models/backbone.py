from transformers import AutoModel, PreTrainedModel
from .multi_task_head import MultiSampleDropoutHead
import torch.nn as nn
import torch


class XLMRobertaForPolarization(nn.Module):
    def __init__(self, model_name, num_labels, task="st1", use_multi_dropout=True, num_dropout=5, dropout_prob=0.1):
        super().__init__()
        # 加载预训练底座
        self.roberta = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.roberta.config.hidden_size
        self.num_labels = num_labels
        self.task = task.lower()

        # 根据配置选择分类头
        if use_multi_dropout:
            self.classifier = MultiSampleDropoutHead(
                self.hidden_size, 
                num_labels, 
                num_dropout=num_dropout, 
                dropout_prob=dropout_prob
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(self.hidden_size, num_labels)
            )

    def forward(self, input_ids, attention_mask=None, labels=None):
        # 1. 提取底座特征
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # 使用 [CLS] 位的特征 (或者通过 mean pooling)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # 2. 通过多重 Dropout 分类头
        logits = self.classifier(pooled_output)

        # 3. 计算损失（如果提供了 labels）
        loss = None
        if labels is not None:
            if self.task == "st1":
                # ST1: 二分类任务，使用 CrossEntropyLoss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())
            else:
                # ST2/ST3: 多标签任务，使用 BCEWithLogitsLoss
                loss_fct = nn.BCEWithLogitsLoss()
                # 确保 labels 是 float 类型且形状匹配 [batch_size, num_labels]
                labels = labels.float()
                # 如果 labels 是 1D，需要 reshape 为 [batch_size, num_labels]
                # 但通常 collator 会处理成正确的形状
                if labels.dim() == 1:
                    # 如果 batch_size=1，需要 unsqueeze；否则需要 reshape
                    if len(labels) == self.num_labels:
                        # 单个样本的多标签列表
                        labels = labels.unsqueeze(0)
                    else:
                        # 多个样本，需要 reshape
                        labels = labels.view(-1, self.num_labels)
                elif labels.dim() == 2 and labels.size(1) != self.num_labels:
                    # 如果形状不匹配，尝试 reshape
                    labels = labels.view(-1, self.num_labels)
                loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}