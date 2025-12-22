import json
from collections import Counter
import os

# 1. 定义数据路径
data_path = 'data/processed/train_joint.jsonl'


def run_statistics():
    if not os.path.exists(data_path):
        print(f"❌ 错误：在当前目录下找不到文件 {data_path}")
        print("请确保你在 POLAR_SemEval2026 文件夹内运行此脚本。")
        return

    labels = []
    total_count = 0
    error_count = 0

    print("🚀 正在扫描 7.3 万条数据，请稍候...")

    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                # 核心修正：使用 label_st1 键名
                label = item.get('label_st1')

                if label is not None:
                    labels.append(label)
                else:
                    error_count += 1
                total_count += 1
            except Exception as e:
                error_count += 1
                continue

    # 2. 计算统计数据
    counts = Counter(labels)

    print("\n" + "=" * 40)
    print("📊 POLAR Subtask 1 数据分布报告")
    print("=" * 40)
    print(f"✅ 读取总行数: {total_count}")
    print(f"⚠️ 无效/缺失行: {error_count}")
    print("-" * 40)

    if labels:
        for label, count in sorted(counts.items()):
            percentage = (count / len(labels)) * 100
            label_name = "极化 (1)" if label == 1 else "非极化 (0)"
            print(f"📍 {label_name}: {count:6d} 条 | 占比: {percentage:6.2f}%")

        # 3. 计算比例诊断
        num_0 = counts.get(0, 0)
        num_1 = counts.get(1, 0)
        if num_1 > 0:
            ratio = num_0 / num_1
            print("-" * 40)
            print(f"💡 类别比例 (0:1) 为: {ratio:.2f} : 1")

            if ratio > 2.0:
                print("\n🚨 诊断结果：存在严重【类别不平衡】！")
                print(f"模型目前只需猜 0 就能获得 {(num_0 / total_count) * 100 :.1f}% 的准确率。")
                print("这就是为什么你的 F1 Macro 只有 0.35 的根本原因。")
            else:
                print("\n✅ 诊断结果：数据相对平衡。")
    else:
        print("❌ 未能提取到任何有效标签，请检查文件格式。")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    run_statistics()