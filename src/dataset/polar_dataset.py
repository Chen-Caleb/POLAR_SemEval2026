import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pathlib import Path


class PolarDataset(Dataset):
    """
    通用极化数据集类，通过 task 参数支持不同子任务
    """

    def __init__(self, data_path, tokenizer_name, max_length=128, task="st1"):
        self.data = []
        self.task = task.lower()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # 路径解析逻辑保持不变...
        self.project_root = Path(__file__).resolve().parents[2]
        absolute_path = self.project_root / data_path if not Path(data_path).is_absolute() else Path(data_path)

        with open(absolute_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)

                # 🚀 根据任务类型“钩出”对应数据
                if self.task == "st1" and item.get("label_st1") != -1:
                    self.data.append({"text": item["text"], "label": item["label_st1"]})

                elif self.task == "st2" and item.get("label_st2"):
                    # Stage 2 逻辑：仅加载已标记极化的样本
                    if item.get("label_st1") == 1:
                        self.data.append({"text": item["text"], "label": item["label_st2"]})

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

        # 将标签转换为张量。注意：ST1 是单个 float，ST2/3 是列表转 float
        label_tensor = torch.tensor(item["label"], dtype=torch.float)
        if self.task == "st1":
            label_tensor = label_tensor.unsqueeze(0)  # 保持 (1,) 形状供二分类损失使用

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label_tensor
        }