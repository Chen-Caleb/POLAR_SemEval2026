import yaml
import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

# 导入自定义模型和数据集类
from src.models.backbone import XLMRobertaForPolarization
from src.utils.submission_tools import generate_submission_zip


class InferenceDataset(Dataset):
    """专门用于推理的数据集类，处理无标签的情况，支持推理注入"""

    def __init__(self, df, tokenizer, max_length=256, task="st1"):
        self.texts = df['text'].astype(str).tolist()
        # 支持推理注入：如果数据中有 analysis 字段，则使用
        self.analyses = df.get('analysis', [None] * len(self.texts))
        if isinstance(self.analyses, pd.Series):
            self.analyses = self.analyses.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task.lower()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        analysis = self.analyses[idx] if idx < len(self.analyses) else None
        
        # 🚀 推理注入：与训练时保持一致
        if analysis and str(analysis).strip():
            sep = self.tokenizer.sep_token
            text = f"{text} {sep} Analysis: {analysis}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,  # 使用动态填充
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

    # 2. 加载模型与分词器（使用自定义模型类）
    print(f"📦 正在从 {checkpoint_path} 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # 从配置中读取模型参数
    num_labels = config['model'].get('num_labels', 2)
    use_multi_dropout = config['model'].get('use_multi_dropout', True)
    num_dropout = config['model'].get('num_dropout', 5)
    dropout_prob = config['model'].get('dropout_prob', 0.1)
    
    # 确定 backbone 模型名称（从配置或 checkpoint 的 config.json 读取）
    backbone_name = config['model'].get('backbone', 'xlm-roberta-base')
    
    # 使用自定义模型类创建模型实例
    model = XLMRobertaForPolarization(
        model_name=backbone_name,  # 使用配置中的 backbone 名称
        num_labels=num_labels,
        task=args.task,
        use_multi_dropout=use_multi_dropout,
        num_dropout=num_dropout,
        dropout_prob=dropout_prob
    )
    
    # 加载保存的模型权重
    model_file = checkpoint_path / "pytorch_model.bin"
    if not model_file.exists():
        # 尝试 safetensors 格式
        from safetensors.torch import load_file
        model_file = checkpoint_path / "model.safetensors"
        if model_file.exists():
            state_dict = load_file(str(model_file))
            model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"❌ 在 {checkpoint_path} 中找不到模型文件 (pytorch_model.bin 或 model.safetensors)")
    else:
        state_dict = torch.load(str(model_file), map_location=device)
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()

    # 3. 加载测试/验证数据 (从配置文件的 data 路径读取)
    # 根据任务选择对应的dev文件
    dev_file_map = {"st1": "dev_subtask1.jsonl", "st2": "dev_subtask2.jsonl", "st3": "dev_subtask3.jsonl"}
    dev_filename = dev_file_map.get(args.task, "dev_subtask1.jsonl")
    
    test_data_path = Path(config['data']['train_file']).parent / dev_filename
    if not test_data_path.exists():
        # 如果路径不对，尝试寻找 data/processed/
        test_data_path = Path("data/processed") / dev_filename

    if not test_data_path.exists():
        raise FileNotFoundError(f"❌ 找不到测试数据文件: {test_data_path}")

    print(f"📊 正在读取数据: {test_data_path}")
    df = pd.read_json(test_data_path, lines=True)

    # 4. 设置推理引擎 (使用 Trainer 的 predict 模式)
    dataset = InferenceDataset(
        df, 
        tokenizer, 
        max_length=config['model'].get('max_length', 256),
        task=args.task
    )
    
    # 需要动态填充，导入 collator
    from src.dataset.data_collator import get_polar_collator
    data_collator = get_polar_collator(tokenizer)

    # 只需要最基本的 TrainingArguments 来运行 predict
    training_args = TrainingArguments(
        output_dir="./temp_preds",
        per_device_eval_batch_size=config['train'].get('batch_size', 32),
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(model=model, args=training_args, data_collator=data_collator)

    # 5. 执行推理
    print(f"🔮 正在为任务 {args.task.upper()} 生成预测...")
    raw_output = trainer.predict(dataset)
    
    # 根据任务类型处理预测结果
    if args.task == "st1":
        # 二分类：取 argmax
        preds = np.argmax(raw_output.predictions, axis=-1)
    else:
        # 多标签：使用 sigmoid + 阈值
        probs = 1 / (1 + np.exp(-raw_output.predictions))  # sigmoid
        preds = (probs > 0.5).astype(int)

    # 将结果填回 DataFrame
    # 根据任务类型设置不同的列名
    if args.task == "st1":
        df['polarization'] = preds
    elif args.task == "st2":
        # ST2: 5个标签列
        label_cols = ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']
        for i, col in enumerate(label_cols):
            df[col] = preds[:, i] if preds.ndim > 1 else preds
    elif args.task == "st3":
        # ST3: 6个标签列
        label_cols = ['stereotype', 'vilification', 'dehumanization', 'extreme_language', 'lack_of_empathy', 'invalidation']
        for i, col in enumerate(label_cols):
            df[col] = preds[:, i] if preds.ndim > 1 else preds

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