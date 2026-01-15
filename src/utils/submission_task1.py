import os
import shutil
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

# ==================== 核心配置 ====================
# 1. 解压后的模型路径
PROJECT_ROOT = Path(os.getcwd())

# 2. 基于根目录定义其他子路径 (统一使用 / 运算符，不要混用 os.path.join)
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "st1_baseline"
DEV_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dev_subtask1.jsonl"

# 3. 提交包相关配置
SUBMISSION_DIR = "subtask_1"
OUTPUT_ZIP_NAME = "submission_st1"

# ========================================================

class SimplePolarDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df['text'].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


def run_st1_submission():
    print(f"🚀 开始生成 Subtask 1 提交包...")

    # --- 1. 环境与路径核对 ---
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误：找不到模型路径 {MODEL_PATH}")
        return

    # --- 2. 加载模型与分词器 ---
    print(f"📦 正在从 {MODEL_PATH} 加载权重...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

    # --- 3. 加载测试数据 ---
    if not os.path.exists(DEV_DATA_PATH):
        print(f"❌ 错误：找不到测试集文件 {DEV_DATA_PATH}")
        return
    df = pd.read_json(DEV_DATA_PATH, lines=True)
    print(f"📊 成功加载 {len(df)} 条测试样本")

    # --- 4. 执行预测 (Classification 逻辑) ---
    print("🔮 正在进行模型推理 (真分类模式)...")
    dataset = SimplePolarDataset(df, tokenizer)
    trainer = Trainer(model=model)

    raw_output = trainer.predict(dataset)
    # 核心：使用 argmax 选取得分最高的类别索引 (0 或 1)
    preds = np.argmax(raw_output.predictions, axis=-1)

    df['polarization'] = preds

    # --- 5. 按语种拆分并打包 ---
    print("📦 正在按照 Codabench 规范格式化文件...")
    if os.path.exists(SUBMISSION_DIR):
        shutil.rmtree(SUBMISSION_DIR)
    os.makedirs(SUBMISSION_DIR)

    # 逻辑：从 id 中提取语言前缀 (例如: 'amh_001' -> 'amh')
    df['lang'] = df['id'].apply(lambda x: str(x).split('_')[0])

    for lang in df['lang'].unique():
        lang_df = df[df['lang'] == lang]
        output_file = os.path.join(SUBMISSION_DIR, f"pred_{lang}.csv")
        # 仅保留比赛要求的 id 和预测结果列
        lang_df[['id', 'polarization']].to_csv(output_file, index=False)
        print(f"   ✅ 已生成 pred_{lang}.csv")

    # 打包为 zip
    shutil.make_archive(OUTPUT_ZIP_NAME, 'zip', root_dir=".", base_dir=SUBMISSION_DIR)

    print("\n" + "=" * 50)
    print(f"🎉 提交包制作完成！路径: /content/{OUTPUT_ZIP_NAME}.zip")
    print(f"💡 包含语种: {list(df['lang'].unique())}")
    print("=" * 50)


if __name__ == "__main__":
    run_st1_submission()