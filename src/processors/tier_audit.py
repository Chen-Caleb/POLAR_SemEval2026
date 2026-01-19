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

# 🚀 导入项目统一的组件
from src.dataset.polar_dataset import MultitaskPolarDataset
from src.engine.evaluator import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="POLAR Tier Audit System")
    parser.add_argument("--config", type=str, default="configs/augmented_st1.yaml", help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练好的模型路径 (checkpoint)")
    parser.add_argument("--task", type=str, default="st1", help="审计的任务 (st1/st2/st3)")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def run_tier_audit():
    args = parse_args()

    # 1. 环境与配置加载
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 统一输出路径到项目根目录下的 DeepAudit_Results
    output_dir = Path("DeepAudit_Results")
    output_dir.mkdir(exist_ok=True)

    # 2. 加载模型与分词器 (从指定的 checkpoint 加载)
    print(f"📦 正在加载受检模型: {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device)
    model.eval()

    # 3. 加载数据集 (使用统一的 MultitaskPolarDataset 保证推理注入逻辑一致)
    dataset = MultitaskPolarDataset(
        data_path=config['data']['train_file'],
        tokenizer_name=args.checkpoint,
        max_length=config['model'].get('max_length', 256),
        task=args.task,
        is_test=False  # 需要加载标签进行对比
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    results = []
    print(f"🔍 启动五层审计：正在扫描 {len(dataset)} 条样本...")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            confs = torch.max(probs, dim=1).values.cpu().numpy()

            for i in range(len(batch['id'])):
                # 注意：从 dataset 原始列表获取 text 以保证对应
                results.append({
                    'id': batch['id'][i],
                    'lang': str(batch['id'][i]).split('_')[0],
                    'text': dataset.data[i]['text'],
                    'label': labels[i].item(),
                    'pred': preds[i],
                    'conf': confs[i],
                    'is_correct': labels[i].item() == preds[i]
                })

    df = pd.DataFrame(results)

    # --- 2. 五层诊断分流逻辑 (保留原始逻辑) ---
    t1_mask = (~df['is_correct']) & (df['conf'] > 0.90)  # Conflict
    t2_mask = (~df['is_correct']) & (df['conf'] > 0.70) & (df['conf'] <= 0.90)  # Misled
    t3_mask = (~df['is_correct']) & (df['conf'] <= 0.70)  # Confusion
    t4_mask = (df['is_correct']) & (df['conf'] <= 0.70)  # Unstable Corrects

    def save_clean_csv(mask, filename):
        sub_df = df[mask].drop(columns=['is_correct'])
        sub_df.to_csv(output_dir / filename, index=False)
        return len(sub_df)

    print(f"💾 正在保存分层错题本至: {output_dir}")
    q1_count = save_clean_csv(t1_mask, f'{args.task}_Conflict_T1.csv')
    save_clean_csv(t2_mask, f'{args.task}_Misled_T2.csv')
    save_clean_csv(t3_mask, f'{args.task}_Confusion_T3.csv')
    save_clean_csv(t4_mask, f'{args.task}_Unstable_Corrects.csv')

    # --- 3. 生成语种多维透视报告 (保留原始逻辑) ---
    print("\n📊 正在生成审计分析报告...")
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

    # 计算全局汇总
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
    print(f"🎉 审计完成！共发现 {q1_count} 个 Tier 1 冲突样本。结果已存入 {output_dir}")


if __name__ == "__main__":
    run_tier_audit()