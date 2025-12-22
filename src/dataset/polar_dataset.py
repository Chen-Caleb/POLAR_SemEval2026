import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pathlib import Path


class PolarDataset(Dataset):
    """
    通用极化数据集类，已优化为支持 ST1 分类任务
    """

    def __init__(self, data_path, tokenizer_name, max_length=128, task="st1"):
        self.data = []
        self.task = task.lower()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # 路径解析逻辑
        self.project_root = Path(__file__).resolve().parents[2]
        absolute_path = self.project_root / data_path if not Path(data_path).is_absolute() else Path(data_path)

        with open(absolute_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)

                # ST1: 极化分类 (0/1)
                if self.task == "st1" and item.get("label_st1") != -1:
                    self.data.append({"text": item["text"], "label": item["label_st1"]})

                # ST2: 极化程度回归
                elif self.task == "st2" and item.get("label_st2"):
                    if item.get("label_st1") == 1:
                        self.data.append({"text": item["text"], "label": item["label_st2"]})

                # ST3: 维度识别
                elif self.task == "st3" and item.get("label_st3"):
                    if item.get("label_st1") == 1:
                        self.data.append({"text": item["text"], "label": item["label_st3"]})

        print(f"✅ {self.task.upper()} 数据集加载完成！规模: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(item["text"], max_length=self.max_length,
                                  padding="max_length", truncation=True, return_tensors="pt")

        # 🚀 核心修改逻辑
        if self.task == "st1":
            # 1. 修改为 Long 类型 (分类任务必需)
            # 2. 移除 unsqueeze(0)，CrossEntropyLoss 期望输入形状为 (batch_size,)
            label_tensor = torch.tensor(item["label"], dtype=torch.long)
        else:
            # ST2/ST3 保持 float 类型 (回归或多标签任务)
            label_tensor = torch.tensor(item["label"], dtype=torch.float)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label_tensor
        }