import pandas as pd
import numpy as np
import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
# from google.colab import drive

# ================= 1. 环境与 Drive 路径配置 =================

BASE_DIR = "/content/POLAR_SemEval2026"
MODEL_PATH = "/content/POLAR_SemEval2026/checkpoints/ST1_baseline"
TRAIN_DATA_PATH = "data/processed/train_joint.jsonl"

# 自动创建审计结果文件夹
OUTPUT_DIR = os.path.join(BASE_DIR, "DeepAudit_Results")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"📂 已创建新文件夹: {OUTPUT_DIR}")

BATCH_SIZE = 64
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================

class PolarDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(item['text'], truncation=True, padding='max_length',
                                  max_length=self.max_length, return_tensors='pt')

        # 对应 ST1 任务，你的键名是 label_st1
        label = item.get('label_st1', None)
        if label is None:
            raise KeyError(f"数据 ID {item.get('id')} 中找不到 'label_st1'。")

        return {
            'id': item['id'],
            'text': item['text'],
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': int(label)
        }


def run_tier_audit():
    print(f"📦 正在加载模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, fix_mistral_regex=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    dataset = PolarDataset(TRAIN_DATA_PATH, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    results = []
    print(f"🔍 启动五层审计：正在扫描 {len(dataset)} 条样本...")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, mask = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids, attention_mask=mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            confs = torch.max(probs, dim=1).values.cpu().numpy()

            for i in range(len(batch['id'])):
                results.append({
                    'id': batch['id'][i],
                    'lang': str(batch['id'][i]).split('_')[0],
                    'text': batch['text'][i],
                    'label': batch['label'][i].item(),
                    'pred': preds[i],
                    'conf': confs[i],
                    'is_correct': batch['label'][i].item() == preds[i]
                })

    df = pd.DataFrame(results)

    # --- 2. 五层诊断分流逻辑 ---
    t1_mask = (~df['is_correct']) & (df['conf'] > 0.90)  # Conflict
    t2_mask = (~df['is_correct']) & (df['conf'] > 0.70) & (df['conf'] <= 0.90)  # Misled
    t3_mask = (~df['is_correct']) & (df['conf'] <= 0.70)  # Confusion
    t4_mask = (df['is_correct']) & (df['conf'] <= 0.70)  # Unstable Corrects

    # 导出时删除 is_correct 列 (根据你的要求)
    def save_clean_csv(mask, filename):
        sub_df = df[mask].drop(columns=['is_correct'])
        sub_df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

    print(f"💾 正在将分层错题本自动保存至 Google Drive...")
    save_clean_csv(t1_mask, 'ST1_Conflict_T1.csv')
    save_clean_csv(t2_mask, 'ST1_Misled_T2.csv')
    save_clean_csv(t3_mask, 'ST1_Confusion_T3.csv')
    save_clean_csv(t4_mask, 'ST1_Unstable_Corrects.csv')

    # --- 3. 生成语种多维透视报告 ---
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
    final_report.to_csv(os.path.join(OUTPUT_DIR, 'TRAIN_Tier_Audit_Report.csv'), index=False)

    print("\n" + "=" * 130)
    # 打印最核心的五个列
    disp_cols = ['Language', 'Macro_F1', 'F1_Binary_P', 'Total_Prob_Rate', 'T2_Misled_Rate', 'Unstable_Rate']
    print(final_report[disp_cols].to_string(index=False))
    print("=" * 130)
    print(f"🎉 任务完成！已过滤掉稳定样本，其余四类错题本已存入 Drive。")


if __name__ == "__main__":
    run_tier_audit()