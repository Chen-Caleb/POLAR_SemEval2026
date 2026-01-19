import yaml
import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# 从你重构后的工具包中导入打包函数
from src.utils.submission_tools import generate_submission_zip


class InferenceDataset(Dataset):
    """专门用于推理的数据集类，处理无标签的情况"""

    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df['text'].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


def main():
    parser = argparse.ArgumentParser(description="Inference Script for POLAR SemEval 2026")
    parser.add_argument("--config", type=str, default="configs/augmented_st1.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the specific checkpoint (e.g., checkpoints/st1_baseline/final_model)")
    parser.add_argument("--task", type=str, default="st1", help="Task name (st1/st2/st3)")
    args = parser.parse_args()

    # 1. 加载配置与路径
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = Path(args.checkpoint)

    # 2. 加载模型与分词器
    print(f"📦 正在从 {checkpoint_path} 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path).to(device)

    # 3. 加载测试/验证数据 (从配置文件的 data 路径读取)
    test_data_path = Path(config['data']['train_file']).parent / "dev_subtask1.jsonl"
    if not test_data_path.exists():
        # 如果路径不对，尝试寻找 data/processed/dev_subtask1.jsonl
        test_data_path = Path("data/processed/dev_subtask1.jsonl")

    print(f"📊 正在读取数据: {test_data_path}")
    df = pd.read_json(test_data_path, lines=True)

    # 4. 设置推理引擎 (使用 Trainer 的 predict 模式)
    dataset = InferenceDataset(df, tokenizer, max_length=config['model']['max_length'])

    # 只需要最基本的 TrainingArguments 来运行 predict
    training_args = TrainingArguments(
        output_dir="./temp_preds",
        per_device_eval_batch_size=config['train'].get('batch_size', 32),
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(model=model, args=training_args)

    # 5. 执行推理
    print(f"🔮 正在为任务 {args.task} 生成预测...")
    raw_output = trainer.predict(dataset)
    preds = np.argmax(raw_output.predictions, axis=-1)

    # 将结果填回 DataFrame
    df['polarization'] = preds

    # 6. 调用工具函数生成提交包
    # 结果存放在 outputs 目录下
    output_zip_name = f"submission_{args.task}_v1"
    generate_submission_zip(
        df=df,
        output_dir=f"temp_{args.task}",
        zip_name=output_zip_name
    )

    print(f"\n✅ 推理与打包流程结束！")


if __name__ == "__main__":
    main()