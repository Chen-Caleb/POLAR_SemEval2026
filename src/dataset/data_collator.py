import torch
from transformers import DataCollatorWithPadding


class PolarDataCollator(DataCollatorWithPadding):
    """
    定制化动态填充器
    处理逻辑：
    1. 自动对 input_ids 和 attention_mask 进行补齐
    2. 处理测试模式下可能存在的非 Tensor 字段（如 'id'）
    """

    def __call__(self, features):
        # 提取非 Tensor 字段（如测试集的 ID）防止填充器报错
        ids = [f.pop("id") for f in features if "id" in f]

        # 使用 Transformers 默认逻辑进行动态填充
        batch = super().__call__(features)

        # 将 ID 放回 batch 中，方便 get_outputs.py 使用
        if ids:
            batch["ids"] = ids

        return batch


def get_polar_collator(tokenizer):
    return PolarDataCollator(tokenizer=tokenizer, padding=True)