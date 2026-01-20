import os
import shutil
import pandas as pd
from pathlib import Path


def generate_submission_zip(df, output_dir="subtask_1", zip_name="submission_st1"):
    """
    Split prediction DataFrame by language and package into the official submission format.

    :param df: DataFrame containing 'id' and 'polarization' columns
    :param output_dir: Temporary folder to store per-language CSV files
    :param zip_name: Final submission archive name (without .zip)
    """
    print(f"📦 Formatting prediction files according to official submission specification...")

    # Ensure output directory is clean
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    # 1. Extract language prefix from id (e.g., 'amh_001' -> 'amh')
    # Add defensive programming to handle unexpected ID formats
    df['lang'] = df['id'].apply(lambda x: str(x).split('_')[0] if '_' in str(x) else 'unknown')

    # 2. (Optional) Validate language coverage against the expected 22 languages
    languages = df['lang'].unique()
    print(f"   💡 Detected {len(languages)} languages")

    # 3. Save per-language CSV files
    for lang in languages:
        lang_df = df[df['lang'] == lang]
        # Official requirement: file name pred_{lang}.csv with columns id, polarization
        file_path = output_path / f"pred_{lang}.csv"
        lang_df[['id', 'polarization']].to_csv(file_path, index=False)
        print(f"   ✅ Generated {file_path.name}")

    # 4. Create zip archive
    # root_dir is the parent of the folder to archive, base_dir is the folder name
    shutil.make_archive(zip_name, 'zip', root_dir=".", base_dir=output_dir)

    # Optionally clean up temporary folder
    # shutil.rmtree(output_path)

    print("\n" + "=" * 50)
    print(f"🎉 Submission archive created: {zip_name}.zip")
    print("=" * 50)


def validate_submission(zip_path):
    """Validate generated submission archive against basic official requirements (e.g., file count)."""
    # TODO: implement validation logic if needed
    pass