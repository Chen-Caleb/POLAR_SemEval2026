import yaml
import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

# Import custom model and dataset utilities
from src.models.backbone import XLMRobertaForPolarization
from src.utils.submission_tools import generate_submission_zip


class InferenceDataset(Dataset):
    """Dataset class for inference (no labels), with optional reasoning injection."""

    def __init__(self, df, tokenizer, max_length=256, task="st1"):
        self.texts = df['text'].astype(str).tolist()
        # Support reasoning injection: use 'analysis' field if available
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
        
        # 🚀 Reasoning injection: keep consistent with training
        if analysis and str(analysis).strip():
            sep = self.tokenizer.sep_token
            text = f"{text} {sep} Analysis: {analysis}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,  # Use dynamic padding
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

    # 1. Load config and paths
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = Path(args.checkpoint)

    # 2. Load model and tokenizer (using custom model class)
    print(f"📦 Loading model from {checkpoint_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Read model parameters from config
    num_labels = config['model'].get('num_labels', 2)
    use_multi_dropout = config['model'].get('use_multi_dropout', True)
    num_dropout = config['model'].get('num_dropout', 5)
    dropout_prob = config['model'].get('dropout_prob', 0.1)
    
    # Determine backbone model name (from config or checkpoint config.json)
    backbone_name = config['model'].get('backbone', 'xlm-roberta-base')
    
    # Create model instance using custom class
    model = XLMRobertaForPolarization(
        model_name=backbone_name,  # Use backbone name from config
        num_labels=num_labels,
        task=args.task,
        use_multi_dropout=use_multi_dropout,
        num_dropout=num_dropout,
        dropout_prob=dropout_prob
    )
    
    # Load saved model weights
    model_file = checkpoint_path / "pytorch_model.bin"
    if not model_file.exists():
        # Try safetensors format as a fallback
        from safetensors.torch import load_file
        model_file = checkpoint_path / "model.safetensors"
        if model_file.exists():
            state_dict = load_file(str(model_file))
            model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"❌ Could not find model file in {checkpoint_path} (pytorch_model.bin or model.safetensors)")
    else:
        state_dict = torch.load(str(model_file), map_location=device)
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()

    # 3. Load test/dev data (from config-specified data path)
    # Choose dev file according to task
    dev_file_map = {"st1": "dev_subtask1.jsonl", "st2": "dev_subtask2.jsonl", "st3": "dev_subtask3.jsonl"}
    dev_filename = dev_file_map.get(args.task, "dev_subtask1.jsonl")
    
    test_data_path = Path(config['data']['train_file']).parent / dev_filename
    if not test_data_path.exists():
        # If path is not correct, fall back to data/processed/
        test_data_path = Path("data/processed") / dev_filename

    if not test_data_path.exists():
        raise FileNotFoundError(f"❌ Test data file not found: {test_data_path}")

    print(f"📊 Loading data from: {test_data_path}")
    df = pd.read_json(test_data_path, lines=True)

    # 4. Set up inference engine (Trainer.predict)
    dataset = InferenceDataset(
        df, 
        tokenizer, 
        max_length=config['model'].get('max_length', 256),
        task=args.task
    )
    
    # Import collator for dynamic padding
    from src.dataset.data_collator import get_polar_collator
    data_collator = get_polar_collator(tokenizer)

    # Only minimal TrainingArguments are required for prediction
    training_args = TrainingArguments(
        output_dir="./temp_preds",
        per_device_eval_batch_size=config['train'].get('batch_size', 32),
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(model=model, args=training_args, data_collator=data_collator)

    # 5. Run inference
    print(f"🔮 Generating predictions for task {args.task.upper()} ...")
    raw_output = trainer.predict(dataset)
    
    # Post-process predictions according to task type
    if args.task == "st1":
        # Binary classification: take argmax
        preds = np.argmax(raw_output.predictions, axis=-1)
    else:
        # Multi-label: use sigmoid + threshold
        probs = 1 / (1 + np.exp(-raw_output.predictions))  # sigmoid
        preds = (probs > 0.5).astype(int)

    # Write predictions back into DataFrame
    # Use different column names depending on the task
    if args.task == "st1":
        df['polarization'] = preds
    elif args.task == "st2":
        # ST2: 5 topic label columns
        label_cols = ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']
        for i, col in enumerate(label_cols):
            df[col] = preds[:, i] if preds.ndim > 1 else preds
    elif args.task == "st3":
        # ST3: 6 rhetorical strategy label columns
        label_cols = ['stereotype', 'vilification', 'dehumanization', 'extreme_language', 'lack_of_empathy', 'invalidation']
        for i, col in enumerate(label_cols):
            df[col] = preds[:, i] if preds.ndim > 1 else preds

    # 6. Generate submission archive
    output_zip_name = f"submission_{args.task}_v1"
    generate_submission_zip(
        df=df,
        output_dir=f"temp_{args.task}",
        zip_name=output_zip_name
    )

    print(f"\n✅ Inference and packaging complete!")


if __name__ == "__main__":
    main()