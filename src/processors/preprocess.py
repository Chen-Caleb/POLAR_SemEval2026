import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm


class PolarPreprocessor:
    def __init__(self):
        # 🚀 Path hardening: resolve project root robustly
        self.project_root = Path(__file__).resolve().parents[2]
        self.raw_path = self.project_root / "data" / "raw"
        self.processed_path = self.project_root / "data" / "processed"
        self.processed_path.mkdir(parents=True, exist_ok=True)

        # Official 22 languages
        self.languages = [
            'amh', 'arb', 'ben', 'mya', 'zho', 'eng', 'deu', 'hau',
            'hin', 'ita', 'khm', 'nep', 'ori', 'fas', 'pol', 'pan',
            'rus', 'spa', 'swa', 'tel', 'tur', 'urd'
        ]

        # 🚀 Define column mappings for multi-label tasks
        self.st2_cols = ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']
        self.st3_cols = ['stereotype', 'vilification', 'dehumanization', 'extreme_language', 'lack_of_empathy',
                         'invalidation']

    def process_all(self):
        print(f"📍 Project root: {self.project_root}")
        self._process_training_set()
        self._process_dev_set()
        print("✨ Preprocessing completed. All label dimensions have been merged into unified files.")

    def _process_training_set(self):
        combined_data = {}
        print("🚀 Aggregating multi-task labels across 22 languages...")

        for st in [1, 2, 3]:
            st_folder = self.raw_path / "train" / f"subtask{st}"
            if not st_folder.exists(): continue

            for lang in tqdm(self.languages, desc=f"Processing ST{st}"):
                file_path = st_folder / f"{lang}.csv"
                if not file_path.exists(): continue

                df = pd.read_csv(file_path, encoding='utf-8')

                # Detect column name variants
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

                    # --- Core logic: extract labels according to task type ---
                    if st == 1:
                        # ST1: extract polarization column
                        c_pol = cols.get('polarization') or cols.get('label')
                        if c_pol: combined_data[uid]["label_st1"] = int(row[c_pol])

                    elif st == 2:
                        # ST2: extract 5 topic dimension columns
                        labels = []
                        for col in self.st2_cols:
                            actual_col = next((c for c in df.columns if c.lower() == col.lower()), None)
                            labels.append(int(row[actual_col]) if actual_col else 0)
                        combined_data[uid]["label_st2"] = labels

                    elif st == 3:
                        # ST3: extract 6 rhetorical strategy columns
                        labels = []
                        for col in self.st3_cols:
                            actual_col = next((c for c in df.columns if c.lower() == col.lower()), None)
                            labels.append(int(row[actual_col]) if actual_col else 0)
                        combined_data[uid]["label_st3"] = labels

        # Save merged training set as JSONL
        output_file = self.processed_path / "train_joint.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in combined_data.values():
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✅ Generated joint training set with all label dimensions: {output_file}")

    def _process_dev_set(self):
        print("🚀 Converting unlabeled dev sets to JSONL format...")
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