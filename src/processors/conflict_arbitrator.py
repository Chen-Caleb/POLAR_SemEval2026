import pandas as pd
import json
import time
import csv
from google import genai
from google.colab import userdata
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. 配置与全英文 Prompt 模板
# ==========================================
MODEL_NAME = "gemini-2.5-flash"

PROMPT_TEMPLATE = """
# Role
You are a senior sociolinguist expert specializing in global subcultures, political contexts, and internet slang.

# Task
Arbitrate "Tier 1 Conflicts" between "Human Labels" and "Model Predictions" for the SemEval 2026 Polarization Detection task.

# Context
- Objective: Identify "Polarization" (inciting conflict, discrimination, dehumanization, or radical stances).
- Conflict: Model is highly confident (conf > 0.9) but contradicts the human label.

# Input Data
- Language: {lang}
- Text: "{text}"
- Human Label: {label} (0=Neutral, 1=Polarized)
- Model Prediction: {pred}

# Reasoning Requirements
Analyze the text based on the linguistic habits of {lang}. 
Identify hidden intents like sarcasm, irony, or phonetic slurs (e.g., "黑乐色" in Chinese).
The analysis MUST be in English regardless of the input text language.

# Output Format (Strict JSON, MUST be in English, Max 150 words)
{{
  "final_label": 0 or 1,
  "category": "Label Error" or "Model Bias",
  "analysis": "[Feature]: Identify key slangs or rhetoric. [Logic]: Explain the polarization logic in one sentence."
}}
"""


# ==========================================
# 2. 核心调用逻辑
# ==========================================
def get_client():
    return genai.Client(api_key=userdata.get('GEMINI_API_KEY'))


def arbitrate_sample(client, row):
    """调用 API 处理单条数据并返回解析后的 JSON"""
    prompt = PROMPT_TEMPLATE.format(
        lang=row['lang'],
        text=row['text'],
        label=row['label'],
        pred=row['pred']
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'temperature': 0.1
            }
        )
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}


# ==========================================
# 3. 多线程并行处理流水线
# ==========================================
def run_pipeline_fast(input_file, output_file, max_workers=5, limit=None):
    # 读取 CSV (增加容错处理)
    try:
        df = pd.read_csv(
            input_file,
            on_bad_lines='skip',  # 跳过格式有问题的行
            quoting=csv.QUOTE_MINIMAL,
            escapechar='\\'
        )
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    if limit:
        df = df.head(limit)

    client = get_client()
    print(f"🚀 启动并行处理 (线程数: {max_workers})")
    print(f"目标文件: {input_file}，预计处理 {len(df)} 条数据...")

    # 使用线程池并发执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 建立任务映射
        future_to_row = {executor.submit(arbitrate_sample, client, row): row for _, row in df.iterrows()}

        with open(output_file, "w", encoding="utf-8") as f:
            # as_completed 保证谁先跑完谁先写入
            for future in tqdm(as_completed(future_to_row), total=len(df)):
                row = future_to_row[future]
                try:
                    res = future.result()

                    if "error" in res:
                        # 如果是频率限制报错，建议在这里增加 time.sleep 或降低 max_workers
                        continue

                    # 构造最终数据条目
                    entry = {
                        "id": row['id'],
                        "lang": row['lang'],
                        "text": row['text'],
                        "final_label": res.get('final_label'),
                        "category": res.get('category'),
                        "analysis": res.get('analysis')  # 确保字段名与 Prompt 一致
                    }

                    # 实时写入 JSONL
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"处理 ID {row.get('id')} 时发生异常: {e}")

    print(f"\n✅ 处理完成！结果保存至: {output_file}")


# ==========================================
# 4. 运行入口
# ==========================================

# 如果你使用的是免费版 API (Free Tier)，建议 max_workers 设为 2 或 3
# 如果你使用的是付费版 API (Pay-as-you-go)，可以设为 10-20 以极速处理
run_pipeline_fast(
    input_file="ST1_Conflict_test.csv",
    output_file="Tier1_Test_Results.jsonl",
    max_workers=5
)