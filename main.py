import os
import yaml
import argparse
import torch
from pathlib import Path
from transformers import TrainingArguments, set_seed, EarlyStoppingCallback, AutoTokenizer

# 1. Import modularized components
from src.dataset.polar_dataset import MultitaskPolarDataset
from src.dataset.data_collator import get_polar_collator
from src.models.backbone import XLMRobertaForPolarization
from src.engine.trainer import FGMTrainer
from src.engine.evaluator import get_compute_metrics_fn


def parse_args():
    parser = argparse.ArgumentParser(description="POLAR SemEval 2026 Training Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--task",
        type=str,
        default="st1",
        choices=["st1", "st2", "st3"],
        help="Subtask name (st1=polarization, st2=topic, st3=rhetorical strategies)",
    )
    return parser.parse_args()


def main():
    # --- Stage A: Environment & configuration setup ---
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Set random seed for reproducibility
    set_seed(config['train'].get('seed', 42))

    # Ensure output directory exists
    output_dir = Path(config['train']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage B: Data pipeline (Dataset + Collator) ---
    print(f"📂 Loading dataset for task: {args.task.upper()}")

    # Instantiate dataset with reasoning injection and optional K-Fold indices
    full_dataset = MultitaskPolarDataset(
        data_path=config['data']['train_file'],
        tokenizer_name=config['model']['backbone'],
        max_length=config['model'].get('max_length', 256),
        task=args.task
    )

    # Train/validation split
    val_split = config['data'].get('val_split', 0.1)
    train_size = int((1 - val_split) * len(full_dataset))
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, len(full_dataset) - train_size])

    # 🚀 Improvement 1: Dynamic padding
    # Padding is done per-batch by the data collator instead of in the dataset.
    tokenizer = AutoTokenizer.from_pretrained(config['model']['backbone'])
    data_collator = get_polar_collator(tokenizer)

    # --- Stage C: Model assembly (Backbone + Multi-Sample Dropout) ---
    # 🚀 Improvement 2: Custom backbone with multi-sample dropout
    # Use a custom model with 5 parallel dropout heads instead of the vanilla classifier head.
    print(f"🧠 Building custom model (backbone: {config['model']['backbone']}, task: {args.task.upper()})")
    model = XLMRobertaForPolarization(
        model_name=config['model']['backbone'],
        num_labels=config['model'].get('num_labels', 2),
        task=args.task,  # Pass task type to choose the correct loss function
        use_multi_dropout=config['model'].get('use_multi_dropout', True),
        num_dropout=config['model'].get('num_dropout', 5),
        dropout_prob=config['model'].get('dropout_prob', 0.1)
    )

    # --- Stage D: Training arguments & engine configuration ---
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
        fp16=torch.cuda.is_available(),  # Enable mixed precision automatically if CUDA is available
        save_total_limit=2,  # Keep only the last 2 checkpoints to save disk space
        report_to="none"
    )

    # 🚀 Improvement 3: FGM adversarial training toggle
    # Only enable adversarial perturbation when use_fgm is set to True in the YAML config.
    use_fgm = config['train'].get('use_fgm', True)
    fgm_eps = config['train'].get('fgm_eps', 0.5) if use_fgm else 0.0

    # Select evaluation function according to task
    compute_metrics_fn = get_compute_metrics_fn(task=args.task)
    
    trainer = FGMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,  # Use dynamic padding collator
        compute_metrics=compute_metrics_fn,  # Task-aware evaluation metrics
        fgm_epsilon=fgm_eps,  # Adversarial perturbation strength
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # --- Stage E: Run training ---
    print(f"🔥 Training started! Adversarial training: {'ON (eps=' + str(fgm_eps) + ')' if use_fgm else 'OFF'}")
    trainer.train()

    # Save final model so that get_outputs.py can load it directly
    final_save_path = output_dir / "final_model"
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"✅ Training complete. Model saved to: {final_save_path}")


if __name__ == "__main__":
    main()