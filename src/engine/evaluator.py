import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def compute_metrics(eval_pred):
    """适配二分类的多维度指标计算"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "f1_macro": f1_score(labels, predictions, average='macro'),
        "accuracy": accuracy_score(labels, predictions)
    }