import json
import torch
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pathlib import Path


class MultitaskPolarDataset(Dataset):
    """
    Multitask dataset for polarization detection.

    Features:
    - Reasoning injection (Reasoning Injection)
    - K-Fold index-based splitting
    - Test/inference mode without labels
    """

    def __init__(self, data_path, tokenizer_name, max_length=256, task="st1", is_test=False, indices=None):
        self.all_data = []  # Raw full dataset
        self.data = []  # Final filtered dataset
        self.task = task.lower()
        self.is_test = is_test
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # 1. Resolve data path
        current_path = Path(os.getcwd())
        absolute_path = Path(data_path) if Path(data_path).is_absolute() else current_path / data_path

        if not absolute_path.exists():
            raise FileNotFoundError(f"❌ Data file not found: {absolute_path}")

        # 2. Load data and filter by label
        with open(absolute_path, 'r', encoding='utf-8') as f:
            raw_list = [json.loads(line) for line in f if line.strip()]

        for item in raw_list:
            if self.is_test:
                self.all_data.append(item)
            else:
                label = self._extract_label(item)
                if label is not None:
                    # Only keep samples with valid labels
                    self.all_data.append({"raw_item": item, "label": label})

        # 3. K-Fold index-based splitting (optional)
        if indices is not None:
            self.data = [self.all_data[i] for i in indices if i < len(self.all_data)]
        else:
            self.data = self.all_data

        print(f"✅ {self.task.upper()} [{'TEST' if is_test else 'TRAIN'}] dataset loaded. Samples: {len(self.data)}")

    def _extract_label(self, item):
        """Core logic: extract labels according to task type."""
        if self.task == "st1":
            l = item.get("label_st1")
            return l if l in [0, 1] else None
        elif self.task == "st2":
            # Only polarized samples (label_st1 == 1) are used for ST2/ST3
            return item.get("label_st2") if item.get("label_st1") == 1 else None
        elif self.task == "st3":
            return item.get("label_st3") if item.get("label_st1") == 1 else None
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        raw_item = item if self.is_test else item["raw_item"]

        text = raw_item.get("text", "")
        analysis = raw_item.get("analysis", "")

        # --- 🚀 推理注入 (Reasoning Injection) ---
        if analysis and str(analysis).strip():
            sep = self.tokenizer.sep_token
            text = f"{text} {sep} Analysis: {analysis}"

        # 核心修改：这里不再填充，只做截断
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,  # <--- 关键：改为动态填充的基础
            truncation=True,
            return_tensors="pt"
        )

        # 标签张量转换
        if self.is_test:
            label_tensor = torch.tensor(0)
        elif self.task == "st1":
            label_tensor = torch.tensor(item["label"], dtype=torch.long)
        else:
            label_tensor = torch.tensor(item["label"], dtype=torch.float)

        res = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label_tensor
        }

        # 推理模式返回 ID 用于结果打包
        if self.is_test:
            res["id"] = raw_item.get("id", "unknown")

        return res