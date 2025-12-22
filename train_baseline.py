import yaml
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed, EarlyStoppingCallback
)
from src.dataset.polar_dataset import PolarDataset


def compute_metrics(eval_pred):
    """🚀 修正：适配 2 个输出节点的分类指标计算"""
    logits, labels = eval_pred

    # 以前是 Sigmoid，现在是寻找 2 个输出中得分最大的索引 (0 或 1)
    # logits 形状从 (batch_size, 1) 变为 (batch_size, 2)
    predictions = np.argmax(logits, axis=-1)

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

    # 2. 构造数据集 (确保你已经更新了 PolarDataset.py 中的 dtype=torch.long)
    full_dataset = PolarDataset(
        data_path=config['data']['train_file'],
        tokenizer_name=config['model']['backbone'],
        max_length=config['model']['max_length']
    )

    train_size = int((1 - config['data']['val_split']) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # 3. 加载预训练模型
    # 🚀 核心修改：num_labels 设为 2，这将自动启用 CrossEntropyLoss
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['backbone'],
        num_labels=2
    )

    # 4. 配置训练参数
    # 4. 配置训练参数
    training_args = TrainingArguments(
        # 🚀 改进 1：直接存入挂载的 Google Drive，断线也不怕
        output_dir="/content/drive/MyDrive/POLAR_Checkpoints/st1_baseline",

        # 🚀 改进 2：只保留最重要的 2 个模型包（节省云盘空间，防止撑爆）
        save_total_limit=2,

        # 🚀 改进 3：增加保存频率（可选），比如每 500 步存一次
        # save_strategy="steps",
        # save_steps=500,

        num_train_epochs=config['train']['epochs'],
        per_device_train_batch_size=config['train']['batch_size'],
        per_device_eval_batch_size=config['train']['batch_size'],
        learning_rate=float(config['train']['learning_rate']),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("🚀 引擎已重新启动，正在以【真分类模式】微调 Subtask 1...")
    trainer.train()

    # 6. 持久化存储
    trainer.save_model(config['train']['output_dir'])
    print(f"✅ 训练圆满结束！修正后的权重已保存至 {config['train']['output_dir']}")


if __name__ == "__main__":
    main()