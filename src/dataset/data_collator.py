import torch
from transformers import DataCollatorWithPadding


class PolarDataCollator(DataCollatorWithPadding):
    """
    Customized dynamic padding collator.

    Logic:
    1. Automatically pad input_ids and attention_mask.
    2. Handle non-tensor fields (such as 'id') that may appear in test mode.
    """

    def __call__(self, features):
        # Extract non-tensor fields (e.g., IDs in test set) to avoid collator errors
        ids = [f.pop("id") for f in features if "id" in f]

        # Use Transformers default logic for dynamic padding
        batch = super().__call__(features)

        # Put IDs back into batch so that get_outputs.py can use them
        if ids:
            batch["ids"] = ids

        return batch


def get_polar_collator(tokenizer):
    return PolarDataCollator(tokenizer=tokenizer, padding=True)