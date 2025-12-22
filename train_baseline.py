import yaml
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)
from src.dataset.polar_dataset import PolarDataset


def compute_metrics(eval_pred):
    """计算 Subtask 1 的核心指标：Macro F1"""
    logits, labels = eval_pred
    # 将模型输出的 Logits 转为 0/1 判定
    probs = 1 / (1 + np.exp(-logits))
    predictions = (probs > 0.5).astype(int)

    return {
        "f1_macro": f1_score(labels, predictions, average='macro'),
        "accuracy": accuracy_score(labels, predictions)
    }


def main():
    # 1. 环境准备
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "configs" / "baseline_st1.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    set_seed(config['train']['seed'])

    # 2. 构造数据集
    full_dataset = PolarDataset(
        data_path=config['data']['train_file'],
        tokenizer_name=config['model']['backbone'],
        max_length=config['model']['max_length']
    )

    # 随机划分训练/验证集 (90/10)
    train_size = int((1 - config['data']['val_split']) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # 3. 加载预训练模型
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['backbone'],
        num_labels=1  # Subtask 1 是二分类
    )

    # 4. 配置训练参数
    training_args = TrainingArguments(
        output_dir=config['train']['output_dir'],
        num_train_epochs=config['train']['epochs'],
        per_device_train_batch_size=config['train']['batch_size'],
        per_device_eval_batch_size=config['train']['batch_size'],
        learning_rate=float(config['train']['learning_rate']),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",  # 以 Macro F1 为准保存最佳模型
        fp16=torch.cuda.is_available(),  # 有 GPU 自动开启混合精度加速
        logging_dir="./logs",
        report_to="none"
    )

    # 5. 启动 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("🚀 引擎已启动，正在微调 Subtask 1 Baseline...")
    trainer.train()

    # 6. 持久化存储
    trainer.save_model(config['train']['output_dir'])
    print(f"✅ 训练圆满结束！权重已保存至 {config['train']['output_dir']}")


if __name__ == "__main__":
    main()