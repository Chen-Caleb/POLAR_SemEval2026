import os
import shutil
import pandas as pd
from pathlib import Path


def generate_submission_zip(df, output_dir="subtask_1", zip_name="submission_st1"):
    """
    将预测结果 DataFrame 按照语种拆分并打包成官方要求的格式
    :param df: 包含 'id' 和 'polarization' 列的 DataFrame
    :param output_dir: 临时存放 CSV 的文件夹
    :param zip_name: 最终生成的压缩文件名
    """
    print(f"📦 正在按照官方规范格式化文件...")

    # 确保输出目录干净
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    # 1. 逻辑：从 id 中提取语言前缀 (例如: 'amh_001' -> 'amh')
    # 增加防御性编程，防止 ID 格式异常
    df['lang'] = df['id'].apply(lambda x: str(x).split('_')[0] if '_' in str(x) else 'unknown')

    # 2. 验证语种是否完整 (可选：根据官方要求的 22 种语言列表进行检查)
    languages = df['lang'].unique()
    print(f"   💡 检测到语种数量: {len(languages)}")

    # 3. 按语种保存 CSV
    for lang in languages:
        lang_df = df[df['lang'] == lang]
        # 官方通常要求：文件名为 pred_{lang}.csv，列名为 id, polarization
        file_path = output_path / f"pred_{lang}.csv"
        lang_df[['id', 'polarization']].to_csv(file_path, index=False)
        print(f"   ✅ 已生成 {file_path.name}")

    # 4. 打包为 zip
    # root_dir 是要打包的文件夹的父目录，base_dir 是文件夹名
    shutil.make_archive(zip_name, 'zip', root_dir=".", base_dir=output_dir)

    # 清理临时文件夹 (可选)
    # shutil.rmtree(output_path)

    print("\n" + "=" * 50)
    print(f"🎉 提交包制作完成！文件名: {zip_name}.zip")
    print("=" * 50)


def validate_submission(zip_path):
    """验证生成的压缩包是否符合官方基本规范（例如文件数量）"""
    # 可以在此处添加逻辑
    pass