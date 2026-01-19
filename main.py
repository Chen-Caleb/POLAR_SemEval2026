import yaml
import torch
import argparse
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback
)
# 统一使用你最新的多任务数据集类
from src.dataset.multitask_data_loader import MultitaskPolarDataset


def compute_metrics(eval_pred):
    """通用二分类指标计算"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "f1_macro": f1_score(labels, predictions, average='macro'),
        "accuracy": accuracy_score(labels, predictions)
    }


def main():
    # --- 1. 命令行参数解析 ---
    parser = argparse.ArgumentParser(description="POLAR SemEval 2026 Training Entry")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file (e.g., configs/augmented_st1.yaml)")
    parser.add_argument("--task", type=str, default="st1", help="Task name: st1, st2, or st3")
    args = parser.parse_args()

    # --- 2. 加载指定配置 ---
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"❌ 找不到配置文件: {args.config}")

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设置随机种子
    set_seed(config['train'].get('seed', 42))
    print(f"🚀 已加载配置: {args.config} | 任务: {args.task}")

    # --- 3. 构造数据集 ---
    # 这里统一使用 MultitaskPolarDataset，通过 args.task 切换任务
    full_dataset = MultitaskPolarDataset(
        data_path=config['data']['train_file'],
        tokenizer_name=config['model']['backbone'],
        max_length=config['model']['max_length'],
        task=args.task
    )

    # 划分训练集和验证集
    train_size = int((1 - config['data']['val_split']) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['train'].get('seed', 42))
    )
    print(f"📊 数据就绪: 训练集 {len(train_ds)}, 验证集 {len(val_ds)}")

    # --- 4. 加载模型 ---
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['backbone'],
        num_labels=config['model'].get('num_labels', 2)
    )

    # --- 5. 定义训练参数 ---
    # 这里的 output_dir 会根据配置文件自动切换路径
    training_args = TrainingArguments(
        output_dir=config['train']['output_dir'],
        num_train_epochs=config['train']['epochs'],
        per_device_train_batch_size=config['train']['batch_size'],
        per_device_eval_batch_size=config['train']['batch_size'],
        learning_rate=float(config['train']['learning_rate']),

        # 策略设置
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=2,  # 限制保存数量，防止磁盘满

        # 硬件优化
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_dir="./logs"
    )

    # --- 6. 实例化 Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # 增加早停保护
    )

    # --- 7. 执行训练 ---
    print(f"🔥 正在启动训练，输出目录: {config['train']['output_dir']}")
    trainer.train()

    # --- 8. 最终保存 ---
    final_save_path = Path(config['train']['output_dir']) / "final_model"
    trainer.save_model(final_save_path)
    full_dataset.tokenizer.save_pretrained(final_save_path)
    print(f"✅ 任务完成！权重已导出至: {final_save_path}")


if __name__ == "__main__":
    main()