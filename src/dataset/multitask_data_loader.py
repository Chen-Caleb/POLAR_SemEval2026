import json
import torch
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pathlib import Path


class MultitaskPolarDataset(Dataset):
    """
    更新版：支持 Augmented 数据的推理注入与多任务标签提取
    - ST1: label_st1 (int)
    - ST2: label_st2 (list of 5)
    - ST3: label_st3 (list of 6)
    """

    def __init__(self, data_path, tokenizer_name, max_length=256, task="st1"):
        self.data = []
        self.task = task.lower()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # --- 1. 智能路径解析 ---
        # 兼容 Colab 相对路径与项目结构
        current_path = Path(os.getcwd())
        absolute_path = current_path / data_path if not Path(data_path).is_absolute() else Path(data_path)

        print(f"🔍 正在加载数据: {absolute_path}")

        with open(absolute_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)

                # 提取共有字段
                text = item["text"]
                analysis = item.get("analysis")  # 可能为 None

                # --- 2. 标签提取逻辑 (适配最新 JSON 格式) ---
                if self.task == "st1":
                    # 直接获取 label_st1 (0 或 1)
                    label = item.get("label_st1")
                    if label is not None and label != -1:
                        self.data.append({"text": text, "label": label, "analysis": analysis})

                elif self.task == "st2":
                    # 直接获取 label_st2 列表 [1, 1, 0, 0, 0]
                    label = item.get("label_st2")
                    if label and item.get("label_st1") == 1:
                        self.data.append({"text": text, "label": label, "analysis": analysis})

                elif self.task == "st3":
                    # 直接获取 label_st3 列表 [1, 1, 0, 1, 1, 0]
                    label = item.get("label_st3")
                    if label and item.get("label_st1") == 1:
                        self.data.append({"text": text, "label": label, "analysis": analysis})

        print(f"✅ {self.task.upper()} 加载完成！样本数: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        analysis = item["analysis"]

        # --- 3. 推理注入 (Reasoning Injection) ---
        # 如果有分析信息，利用模型的 sep_token 进行拼接
        if analysis:
            sep = self.tokenizer.sep_token
            # 拼接格式: [Text] </s> Analysis: [Reasoning]
            text = f"{text} {sep} Analysis: {analysis}"

        # 文本分词
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # --- 4. 标签类型转换 ---
        if self.task == "st1":
            # 单标签二分类用 Long
            label_tensor = torch.tensor(item["label"], dtype=torch.long)
        else:
            # 多标签任务用 Float (对应 BCEWithLogitsLoss)
            label_tensor = torch.tensor(item["label"], dtype=torch.float)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label_tensor
        }