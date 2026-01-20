import pandas as pd
import json
import time
import csv
from google import genai
from google.colab import userdata
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. Configuration and English-only prompt template
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
Identify hidden intents like sarcasm, irony, or phonetic slurs (e.g., abusive phonetic wordplay in Chinese).
The analysis MUST be in English regardless of the input text language.

# Output Format (Strict JSON, MUST be in English, Max 150 words)
{{
  "final_label": 0 or 1,
  "category": "Label Error" or "Model Bias",
  "analysis": "[Feature]: Identify key slangs or rhetoric. [Logic]: Explain the polarization logic in one sentence."
}}
"""


# ==========================================
# 2. Core API call logic
# ==========================================
def get_client():
    return genai.Client(api_key=userdata.get('GEMINI_API_KEY'))


def arbitrate_sample(client, row):
    """Call the API on a single sample and return parsed JSON."""
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
# 3. Multithreaded processing pipeline
# ==========================================
def run_pipeline_fast(input_file, output_file, max_workers=5, limit=None):
    # Read CSV with robust error handling
    try:
        df = pd.read_csv(
            input_file,
            on_bad_lines='skip',  # Skip malformed lines
            quoting=csv.QUOTE_MINIMAL,
            escapechar='\\'
        )
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return

    if limit:
        df = df.head(limit)

    client = get_client()
    print(f"🚀 Starting parallel processing (workers: {max_workers})")
    print(f"Input file: {input_file}, total rows to process: {len(df)}")
    
    # Execute in parallel with a thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 建立任务映射
        future_to_row = {executor.submit(arbitrate_sample, client, row): row for _, row in df.iterrows()}

        with open(output_file, "w", encoding="utf-8") as f:
            # as_completed ensures results are written as soon as they are ready
            for future in tqdm(as_completed(future_to_row), total=len(df)):
                row = future_to_row[future]
                try:
                    res = future.result()

                    if "error" in res:
                        # For rate limit errors, consider adding sleep or reducing max_workers
                        continue

                    # Build final JSONL entry
                    entry = {
                        "id": row['id'],
                        "lang": row['lang'],
                        "text": row['text'],
                        "final_label": res.get('final_label'),
                        "category": res.get('category'),
                        "analysis": res.get('analysis')  # Keep field name aligned with prompt
                    }

                    # Write JSONL entry immediately
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Exception while processing ID {row.get('id')}: {e}")

    print(f"\n✅ Processing complete. Results saved to: {output_file}")


# ==========================================
# 4. Entry point
# ==========================================

# For free-tier API keys, consider using max_workers=2 or 3.
# For paid API keys, max_workers=10-20 can significantly speed up processing.
run_pipeline_fast(
    input_file="ST1_Conflict_test.csv",
    output_file="Tier1_Test_Results.jsonl",
    max_workers=5
)