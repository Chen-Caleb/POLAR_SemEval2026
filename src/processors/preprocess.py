import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm


class PolarPreprocessor:
    def __init__(self):
        # 🚀 路径硬化
        self.project_root = Path(__file__).resolve().parents[2]
        self.raw_path = self.project_root / "data" / "raw"
        self.processed_path = self.project_root / "data" / "processed"
        self.processed_path.mkdir(parents=True, exist_ok=True)

        # 官方 22 语种
        self.languages = [
            'amh', 'arb', 'ben', 'mya', 'zho', 'eng', 'deu', 'hau',
            'hin', 'ita', 'khm', 'nep', 'ori', 'fas', 'pol', 'pan',
            'rus', 'spa', 'swa', 'tel', 'tur', 'urd'
        ]

        # 🚀 定义多标签任务的列名映射
        self.st2_cols = ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']
        self.st3_cols = ['stereotype', 'vilification', 'dehumanization', 'extreme_language', 'lack_of_empathy',
                         'invalidation']

    def process_all(self):
        print(f"📍 项目根目录: {self.project_root}")
        self._process_training_set()
        self._process_dev_set()
        print("✨ 预处理任务圆满完成！所有维度标签已精准入库。")

    def _process_training_set(self):
        combined_data = {}
        print("🚀 正在精准整合 22 语种多任务标签...")

        for st in [1, 2, 3]:
            st_folder = self.raw_path / "train" / f"subtask{st}"
            if not st_folder.exists(): continue

            for lang in tqdm(self.languages, desc=f"Processing ST{st}"):
                file_path = st_folder / f"{lang}.csv"
                if not file_path.exists(): continue

                df = pd.read_csv(file_path, encoding='utf-8')

                # 识别列名变体
                cols = {c.lower(): c for c in df.columns}
                c_id = cols.get('id', 'id')
                c_text = cols.get('text', 'text')

                for _, row in df.iterrows():
                    uid = str(row[c_id])
                    if uid not in combined_data:
                        combined_data[uid] = {
                            "id": uid, "text": str(row[c_text]), "lang": lang,
                            "label_st1": -1, "label_st2": [], "label_st3": []
                        }

                    # --- 核心逻辑切换：根据任务类型提取标签 ---
                    if st == 1:
                        # ST1: 提取 polarization 列
                        c_pol = cols.get('polarization') or cols.get('label')
                        if c_pol: combined_data[uid]["label_st1"] = int(row[c_pol])

                    elif st == 2:
                        # ST2: 提取 5 个话题维度列
                        labels = []
                        for col in self.st2_cols:
                            actual_col = next((c for c in df.columns if c.lower() == col.lower()), None)
                            labels.append(int(row[actual_col]) if actual_col else 0)
                        combined_data[uid]["label_st2"] = labels

                    elif st == 3:
                        # ST3: 提取 6 个修辞策略列
                        labels = []
                        for col in self.st3_cols:
                            actual_col = next((c for c in df.columns if c.lower() == col.lower()), None)
                            labels.append(int(row[actual_col]) if actual_col else 0)
                        combined_data[uid]["label_st3"] = labels

        # 保存 JSONL
        output_file = self.processed_path / "train_joint.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in combined_data.values():
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✅ 成功生成全维度联合训练集: {output_file}")

    def _process_dev_set(self):
        print("🚀 正在转换无标签开发集...")
        for st in [1, 2, 3]:
            dev_folder = self.raw_path / "dev_phase" / f"subtask{st}"
            if not dev_folder.exists(): continue

            st_dev_results = []
            for lang in self.languages:
                file_path = dev_folder / f"{lang}.csv"
                if not file_path.exists(): continue
                df = pd.read_csv(file_path, encoding='utf-8')
                c_id = next((c for c in df.columns if c.lower() == 'id'), 'id')
                c_text = next((c for c in df.columns if c.lower() == 'text'), 'text')

                for _, row in df.iterrows():
                    st_dev_results.append({"id": str(row[c_id]), "text": str(row[c_text]), "lang": lang})

            with open(self.processed_path / f"dev_subtask{st}.jsonl", 'w', encoding='utf-8') as f:
                for item in st_dev_results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    PolarPreprocessor().process_all()