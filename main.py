import os
import yaml
import argparse
import torch
from pathlib import Path
from transformers import TrainingArguments, set_seed, EarlyStoppingCallback, AutoTokenizer

# 1. 导入已拆分并优化的模块
from src.dataset.polar_dataset import MultitaskPolarDataset
from src.dataset.data_collator import get_polar_collator
from src.models.backbone import XLMRobertaForPolarization
from src.engine.trainer import FGMTrainer
from src.engine.evaluator import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="POLAR SemEval 2026 Training Pipeline")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--task", type=str, default="st1", choices=["st1", "st2", "st3"], help="子任务名称")
    return parser.parse_args()


def main():
    # --- 阶段 A: 环境与配置准备 ---
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设置随机种子保证实验可复现
    set_seed(config['train'].get('seed', 42))

    # 确定输出目录并自动创建
    output_dir = Path(config['train']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 阶段 B: 数据流水线 (Dataset + Collator) ---
    print(f"📂 正在加载数据任务: {args.task.upper()}")

    # 实例化支持推理注入和 K-Fold 的 Dataset
    full_dataset = MultitaskPolarDataset(
        data_path=config['data']['train_file'],
        tokenizer_name=config['model']['backbone'],
        max_length=config['model'].get('max_length', 256),
        task=args.task
    )

    # 验证集切分
    val_split = config['data'].get('val_split', 0.1)
    train_size = int((1 - val_split) * len(full_dataset))
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, len(full_dataset) - train_size])

    # 🚀 改进 1: 引入动态填充 (Dynamic Padding)
    # 不再在 Dataset 里强制 padding 到最大长度，而是按 Batch 动态对齐
    tokenizer = AutoTokenizer.from_pretrained(config['model']['backbone'])
    data_collator = get_polar_collator(tokenizer)

    # --- 阶段 C: 模型组装 (Backbone + Multi-Sample Dropout) ---
    # 🚀 改进 2: 接入自定义模型底座
    # 使用集成 5 组并行 Dropout 的定制模型，而非原生分类模型
    print(f"🧠 正在组装自定义模型 (底座: {config['model']['backbone']})")
    model = XLMRobertaForPolarization(
        model_name=config['model']['backbone'],
        num_labels=config['model'].get('num_labels', 2),
        use_multi_dropout=config['model'].get('use_multi_dropout', True),
        num_dropout=config['model'].get('num_dropout', 5)
    )

    # --- 阶段 D: 训练参数与引擎设定 ---
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config['train']['epochs'],
        per_device_train_batch_size=config['train']['batch_size'],
        per_device_eval_batch_size=config['train']['batch_size'],
        learning_rate=float(config['train']['learning_rate']),
        weight_decay=config['train'].get('weight_decay', 0.01),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),  # 自动检测并开启混合精度
        save_total_limit=2,  # 仅保留最近 2 个检查点，节省磁盘
        report_to="none"
    )

    # 🚀 改进 3: 配置 FGM 对抗训练开关
    # 只有当 YAML 中设置 use_fgm 为 True 时才激活扰动
    use_fgm = config['train'].get('use_fgm', True)
    fgm_eps = config['train'].get('fgm_eps', 0.5) if use_fgm else 0.0

    trainer = FGMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,  # 传入动态填充器
        compute_metrics=compute_metrics,  # 统一评价指标
        fgm_epsilon=fgm_eps,  # 对抗扰动系数
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # --- 阶段 E: 执行训练 ---
    print(f"🔥 训练启动！对抗扰动: {'开启 (eps=' + str(fgm_eps) + ')' if use_fgm else '关闭'}")
    trainer.train()

    # 最终模型保存，确保 get_outputs.py 可以直接读取
    final_save_path = output_dir / "final_model"
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"✅ 训练完成，模型保存至: {final_save_path}")


if __name__ == "__main__":
    main()