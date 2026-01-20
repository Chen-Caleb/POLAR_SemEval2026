from transformers import AutoModel, PreTrainedModel
from .multi_task_head import MultiSampleDropoutHead
import torch.nn as nn
import torch


class XLMRobertaForPolarization(nn.Module):
    def __init__(self, model_name, num_labels, task="st1", use_multi_dropout=True, num_dropout=5, dropout_prob=0.1):
        super().__init__()
        # Load pretrained backbone
        self.roberta = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.roberta.config.hidden_size
        self.num_labels = num_labels
        self.task = task.lower()

        # Choose classification head according to configuration
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
        # 1. Extract backbone features
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation (or could use mean pooling)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # 2. Forward through multi-sample dropout classification head
        logits = self.classifier(pooled_output)

        # 3. Compute loss (if labels are provided)
        loss = None
        if labels is not None:
            if self.task == "st1":
                # ST1: binary classification, use CrossEntropyLoss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())
            else:
                # ST2/ST3: multi-label classification, use BCEWithLogitsLoss
                loss_fct = nn.BCEWithLogitsLoss()
                # Ensure labels are float and have shape [batch_size, num_labels]
                labels = labels.float()
                # If labels are 1D, reshape to [batch_size, num_labels]
                # (in practice the collator should normally produce the right shape)
                if labels.dim() == 1:
                    # If batch_size=1, unsqueeze; otherwise reshape
                    if len(labels) == self.num_labels:
                        # Single multi-label target
                        labels = labels.unsqueeze(0)
                    else:
                        # Multiple samples, reshape
                        labels = labels.view(-1, self.num_labels)
                elif labels.dim() == 2 and labels.size(1) != self.num_labels:
                    # If shape still mismatched, try reshaping
                    labels = labels.view(-1, self.num_labels)
                loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}