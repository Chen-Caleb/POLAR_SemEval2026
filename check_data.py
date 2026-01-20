import json
from collections import Counter
import os

# 1. Define data path
data_path = 'data/processed/train_joint.jsonl'


def run_statistics():
    if not os.path.exists(data_path):
        print(f"❌ Error: cannot find file {data_path} in current directory.")
        print("Please make sure you run this script from the POLAR_SemEval2026 project root.")
        return

    labels = []
    total_count = 0
    error_count = 0

    print("🚀 Scanning ~73k samples, please wait...")

    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                # Core fix: use label_st1 as the binary label key
                label = item.get('label_st1')

                if label is not None:
                    labels.append(label)
                else:
                    error_count += 1
                total_count += 1
            except Exception:
                error_count += 1
                continue

    # 2. Compute statistics
    counts = Counter(labels)

    print("\n" + "=" * 40)
    print("📊 POLAR Subtask 1 Data Distribution Report")
    print("=" * 40)
    print(f"✅ Total lines read: {total_count}")
    print(f"⚠️ Invalid / missing lines: {error_count}")
    print("-" * 40)

    if labels:
        for label, count in sorted(counts.items()):
            percentage = (count / len(labels)) * 100
            label_name = "Polarized (1)" if label == 1 else "Non-polarized (0)"
            print(f"📍 {label_name}: {count:6d} samples | Ratio: {percentage:6.2f}%")

        # 3. Ratio diagnosis
        num_0 = counts.get(0, 0)
        num_1 = counts.get(1, 0)
        if num_1 > 0:
            ratio = num_0 / num_1
            print("-" * 40)
            print(f"💡 Class ratio (0:1) = {ratio:.2f} : 1")

            if ratio > 2.0:
                print("\n🚨 Diagnosis: severe class imbalance detected.")
                print(f"The model could achieve {(num_0 / total_count) * 100 :.1f}% accuracy by predicting all zeros.")
                print("This explains why the F1 Macro is stuck around 0.35.")
            else:
                print("\n✅ Diagnosis: data is relatively balanced.")
    else:
        print("❌ No valid labels extracted; please check file format.")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    run_statistics()