import os
import yaml
import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# 🚀 Import shared project components
from src.dataset.polar_dataset import MultitaskPolarDataset
from src.engine.evaluator import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="POLAR Tier Audit System")
    parser.add_argument("--config", type=str, default="configs/augmented_st1.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--task", type=str, default="st1", help="Task to audit (st1/st2/st3)")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def run_tier_audit():
    args = parse_args()

    # 1. Load environment and configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use a unified output directory at project root: DeepAudit_Results
    output_dir = Path("DeepAudit_Results")
    output_dir.mkdir(exist_ok=True)

    # 2. Load model and tokenizer from the specified checkpoint
    print(f"📦 Loading audited model from: {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device)
    model.eval()

    # 3. Load dataset (using shared MultitaskPolarDataset to keep reasoning logic consistent)
    dataset = MultitaskPolarDataset(
        data_path=config['data']['train_file'],
        tokenizer_name=args.checkpoint,
        max_length=config['model'].get('max_length', 256),
        task=args.task,
        is_test=False  # Need labels for comparison
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    results = []
    print(f"🔍 Starting 5-tier audit: scanning {len(dataset)} samples...")
    
    # Track global index using enumerate
    global_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            confs = torch.max(probs, dim=1).values.cpu().numpy()

            batch_size = len(labels)
            
            # Correctly handle each sample within the batch
            for i in range(batch_size):
                # Compute global index
                current_global_idx = global_idx + i
                
                # Retrieve original data from dataset
                if current_global_idx < len(dataset.data):
                    item = dataset.data[current_global_idx]
                    raw_item = item if dataset.is_test else item.get("raw_item", item)
                    
                    # Extract id and text
                    sample_id = raw_item.get("id", f"unknown_{current_global_idx}")
                    text = raw_item.get("text", "")
                    lang = raw_item.get("lang", str(sample_id).split('_')[0] if '_' in str(sample_id) else "unknown")
                else:
                    # Defensive programming: handle out-of-range indices
                    sample_id = f"unknown_{current_global_idx}"
                    text = ""
                    lang = "unknown"
                
                results.append({
                    'id': sample_id,
                    'lang': lang,
                    'text': text,
                    'label': labels[i].item(),
                    'pred': preds[i],
                    'conf': confs[i],
                    'is_correct': labels[i].item() == preds[i]
                })
            
            # Update global index
            global_idx += batch_size

    df = pd.DataFrame(results)

    # --- 2. Five-tier diagnostic splitting logic ---
    t1_mask = (~df['is_correct']) & (df['conf'] > 0.90)  # Conflict
    t2_mask = (~df['is_correct']) & (df['conf'] > 0.70) & (df['conf'] <= 0.90)  # Misled
    t3_mask = (~df['is_correct']) & (df['conf'] <= 0.70)  # Confusion
    t4_mask = (df['is_correct']) & (df['conf'] <= 0.70)  # Unstable Corrects

    def save_clean_csv(mask, filename):
        sub_df = df[mask].drop(columns=['is_correct'])
        sub_df.to_csv(output_dir / filename, index=False)
        return len(sub_df)

    print(f"💾 Saving tiered error sets to: {output_dir}")
    q1_count = save_clean_csv(t1_mask, f'{args.task}_Conflict_T1.csv')
    save_clean_csv(t2_mask, f'{args.task}_Misled_T2.csv')
    save_clean_csv(t3_mask, f'{args.task}_Confusion_T3.csv')
    save_clean_csv(t4_mask, f'{args.task}_Unstable_Corrects.csv')

    # --- 3. Generate language-level audit report ---
    print("\n📊 Generating audit analysis report...")
    report = []
    for lang, group in df.groupby('lang'):
        y_true, y_pred = group['label'], group['pred']
        total = len(group)
        q1 = len(group[(~group['is_correct']) & (group['conf'] > 0.90)])
        q2 = len(group[(~group['is_correct']) & (group['conf'] > 0.70) & (group['conf'] <= 0.90)])
        q3 = len(group[(~group['is_correct']) & (group['conf'] <= 0.70)])
        q4 = len(group[(group['is_correct']) & (group['conf'] <= 0.70)])

        prob_rate = (q1 + q2 + q3 + q4) / total

        report.append({
            'Language': lang,
            'Total': total,
            'Macro_F1': round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
            'Accuracy': round(accuracy_score(y_true, y_pred), 4),
            'F1_Binary_P': round(f1_score(y_true, y_pred, average='binary', zero_division=0), 4),
            'Precision_P': round(precision_score(y_true, y_pred, zero_division=0), 4),
            'Recall_P': round(recall_score(y_true, y_pred, zero_division=0), 4),
            'Total_Prob_Rate': f"{round(prob_rate * 100, 2)}%",
            'T1_Conflict_Rate': f"{round(q1 / total * 100, 2)}%",
            'T2_Misled_Rate': f"{round(q2 / total * 100, 2)}%",
            'T3_Confusion_Rate': f"{round(q3 / total * 100, 2)}%",
            'Unstable_Rate': f"{round(q4 / total * 100, 2)}%"
        })

    report_df = pd.DataFrame(report).sort_values(by='Macro_F1')

    # Compute global summary
    g_total = len(df)
    g_q1 = len(df[(~df['is_correct']) & (df['conf'] > 0.90)])
    g_q2 = len(df[(~df['is_correct']) & (df['conf'] > 0.70) & (df['conf'] <= 0.90)])
    g_q3 = len(df[(~df['is_correct']) & (df['conf'] <= 0.70)])
    g_q4 = len(df[(df['is_correct']) & (df['conf'] <= 0.70)])

    avg_row = pd.DataFrame([{
        'Language': 'AVERAGE (GLOBAL)',
        'Total': g_total,
        'Macro_F1': round(f1_score(df['label'], df['pred'], average='macro'), 4),
        'Accuracy': round(accuracy_score(df['label'], df['pred']), 4),
        'F1_Binary_P': round(f1_score(df['label'], df['pred'], average='binary'), 4),
        'Precision_P': round(precision_score(df['label'], df['pred']), 4),
        'Recall_P': round(recall_score(df['label'], df['pred']), 4),
        'Total_Prob_Rate': f"{round((g_q1 + g_q2 + g_q3 + g_q4) / g_total * 100, 2)}%",
        'T1_Conflict_Rate': f"{round(g_q1 / g_total * 100, 2)}%",
        'T2_Misled_Rate': f"{round(g_q2 / g_total * 100, 2)}%",
        'T3_Confusion_Rate': f"{round(g_q3 / g_total * 100, 2)}%",
        'Unstable_Rate': f"{round(g_q4 / g_total * 100, 2)}%"
    }])

    final_report = pd.concat([report_df, avg_row], ignore_index=True)
    final_report.to_csv(output_dir / f'TRAIN_{args.task}_Audit_Report.csv', index=False)

    print("\n" + "=" * 130)
    disp_cols = ['Language', 'Macro_F1', 'F1_Binary_P', 'Total_Prob_Rate', 'T2_Misled_Rate', 'Unstable_Rate']
    print(final_report[disp_cols].to_string(index=False))
    print("=" * 130)
    print(f"🎉 Audit completed! Found {q1_count} Tier 1 conflict samples. Results saved to {output_dir}")


if __name__ == "__main__":
    run_tier_audit()