import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def compute_metrics(eval_pred, task="st1"):
    """
    根据任务类型计算相应的评估指标
    
    Args:
        eval_pred: (logits, labels) 元组
        task: 任务类型 ("st1", "st2", "st3")
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    logits, labels = eval_pred
    
    if task == "st1":
        # ST1: 二分类任务
        predictions = np.argmax(logits, axis=-1)
        labels_flat = labels.flatten()
        
        return {
            "f1_macro": f1_score(labels_flat, predictions, average='macro', zero_division=0),
            "f1_micro": f1_score(labels_flat, predictions, average='micro', zero_division=0),
            "f1_binary": f1_score(labels_flat, predictions, average='binary', zero_division=0),
            "accuracy": accuracy_score(labels_flat, predictions),
            "precision": precision_score(labels_flat, predictions, zero_division=0),
            "recall": recall_score(labels_flat, predictions, zero_division=0)
        }
    else:
        # ST2/ST3: 多标签任务
        # 使用 sigmoid 激活并应用阈值
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        predictions = (probs > 0.5).astype(int)
        
        # 确保 labels 是正确的形状
        if labels.ndim == 1:
            # 如果 labels 是 1D，尝试 reshape
            labels = labels.reshape(-1, logits.shape[1])
        labels = labels.astype(int)
        
        # 展平用于 micro 指标
        labels_flat = labels.flatten()
        predictions_flat = predictions.flatten()
        
        metrics = {
            "f1_macro": f1_score(labels, predictions, average='macro', zero_division=0),
            "f1_micro": f1_score(labels, predictions, average='micro', zero_division=0),
            "f1_samples": f1_score(labels, predictions, average='samples', zero_division=0),  # 每个样本的 F1
            "accuracy": accuracy_score(labels_flat, predictions_flat),
            "precision_macro": precision_score(labels, predictions, average='macro', zero_division=0),
            "recall_macro": recall_score(labels, predictions, average='macro', zero_division=0),
        }
        
        # 为每个标签添加单独的 F1 分数（仅当标签数 <= 10 时，避免输出过长）
        if logits.shape[1] <= 10:
            for i in range(logits.shape[1]):
                metrics[f"f1_label_{i}"] = f1_score(
                    labels[:, i], 
                    predictions[:, i], 
                    zero_division=0
                )
        
        return metrics


def get_compute_metrics_fn(task="st1"):
    """
    工厂函数：返回一个适配特定任务的 compute_metrics 函数
    
    Args:
        task: 任务类型 ("st1", "st2", "st3")
    
    Returns:
        function: 适配该任务的 compute_metrics 函数
    """
    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, task=task)
    
    return compute_metrics_wrapper