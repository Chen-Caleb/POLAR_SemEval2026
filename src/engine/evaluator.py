import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def compute_metrics(eval_pred, task="st1"):
    """
    Compute evaluation metrics according to the task type.

    Args:
        eval_pred: Tuple of (logits, labels).
        task: Task type ("st1", "st2", "st3").

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    logits, labels = eval_pred
    
    if task == "st1":
        # ST1: binary classification
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
        # ST2/ST3: multi-label classification
        # Use sigmoid activation and apply threshold
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        predictions = (probs > 0.5).astype(int)
        
        # Ensure labels have the correct shape
        if labels.ndim == 1:
            # If labels are 1D, try to reshape
            labels = labels.reshape(-1, logits.shape[1])
        labels = labels.astype(int)
        
        # Flatten for micro metrics
        labels_flat = labels.flatten()
        predictions_flat = predictions.flatten()
        
        metrics = {
            "f1_macro": f1_score(labels, predictions, average='macro', zero_division=0),
            "f1_micro": f1_score(labels, predictions, average='micro', zero_division=0),
            "f1_samples": f1_score(labels, predictions, average='samples', zero_division=0),  # F1 per sample
            "accuracy": accuracy_score(labels_flat, predictions_flat),
            "precision_macro": precision_score(labels, predictions, average='macro', zero_division=0),
            "recall_macro": recall_score(labels, predictions, average='macro', zero_division=0),
        }
        
        # Optionally add per-label F1 scores (only when num_labels <= 10 to avoid very long output)
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
    Factory function that returns a task-specific compute_metrics function.

    Args:
        task: Task type ("st1", "st2", "st3").

    Returns:
        function: A compute_metrics wrapper bound to the given task.
    """
    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, task=task)
    
    return compute_metrics_wrapper