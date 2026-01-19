from transformers import AutoModel, PreTrainedModel
from .multi_task_head import MultiSampleDropoutHead
import torch.nn as nn


class XLMRobertaForPolarization(nn.Module):
    def __init__(self, model_name, num_labels, use_multi_dropout=True):
        super().__init__()
        # 加载预训练底座
        self.roberta = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.roberta.config.hidden_size

        # 根据配置选择分类头
        if use_multi_dropout:
            self.classifier = MultiSampleDropoutHead(self.hidden_size, num_labels)
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
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
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}