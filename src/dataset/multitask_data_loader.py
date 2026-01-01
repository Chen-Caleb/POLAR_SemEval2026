import json
import torch
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pathlib import Path


class MultitaskPolarDataset(Dataset):
    """
    SemEval 2026 POLAR 任务通用数据集类
    支持:
    - ST1: 二分类 (Polarized vs Non-polarized)
    - ST2: 多标签维度识别 (Political, Religious, etc.)
    - ST3: 多标签表现形式识别 (Stereotype, Dehumanization, etc.)
    """

    def __init__(self, data_path, tokenizer_name, max_length=128, task="st1"):
        self.data = []
        self.task = task.lower()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # --- 1. 智能路径解析 ---
        # 无论在 Colab 还是本地，自动定位项目根目录
        try:
            self.project_root = Path(__file__).resolve().parents[2]
        except NameError:
            self.project_root = Path(os.getcwd())

        absolute_path = self.project_root / data_path if not Path(data_path).is_absolute() else Path(data_path)

        # --- 2. 核心“钩取”逻辑 ---
        print(f"🔍 正在从 {absolute_path} 加载数据...")
        with open(absolute_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)

                # Subtask 1: 二分类 (0/1)
                if self.task == "st1":
                    if item.get("label_st1") != -1:
                        self.data.append({
                            "text": item["text"],
                            "label": item["label_st1"]
                        })

                # Subtask 2: 极化维度 (多标签: 5维)
                elif self.task == "st2":
                    # 仅当样本被标记为极化时，才研究其维度
                    if item.get("label_st1") == 1:
                        labels = [
                            item["political"], item["racial/ethnic"],
                            item["religious"], item["gender/sexual"], item["other"]
                        ]
                        self.data.append({"text": item["text"], "label": labels})

                # Subtask 3: 表现形式 (多标签: 6种)
                elif self.task == "st3":
                    # 仅当样本被标记为极化时，才研究其表现形式
                    if item.get("label_st1") == 1:
                        labels = [
                            item["stereotype"], item["vilification"], item["dehumanization"],
                            item["extreme_language"], item["lack_of_empathy"], item["invalidation"]
                        ]
                        self.data.append({"text": item["text"], "label": labels})

        print(f"✅ {self.task.upper()} 数据加载完成！样本规模: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 文本分词处理
        encoding = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # --- 3. 标签张量转换逻辑 ---
        if self.task == "st1":
            # ST1 必须是 Long 类型，用于 CrossEntropyLoss
            label_tensor = torch.tensor(item["label"], dtype=torch.long)
        else:
            # ST2/ST3 多标签任务必须是 Float 类型，用于 BCEWithLogitsLoss
            label_tensor = torch.tensor(item["label"], dtype=torch.float)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label_tensor
        }