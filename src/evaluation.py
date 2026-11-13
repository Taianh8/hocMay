"""Classification metrics and confusion matrix plotting."""
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def metrics(y, pred) -> dict:
    """Return requested classification metrics; macro averages cover all 102 classes."""
    labels = list(range(102))
    return {'accuracy': float(accuracy_score(y, pred)), 'macro_precision': float(precision_score(y, pred, labels=labels, average='macro', zero_division=0)), 'macro_recall': float(recall_score(y, pred, labels=labels, average='macro', zero_division=0)), 'macro_f1': float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)), 'weighted_f1': float(f1_score(y, pred, labels=labels, average='weighted', zero_division=0))}

def plot_confusion(y, pred, path):
    """Save a complete 102-class confusion matrix in official ID order."""
    cm = confusion_matrix(y, pred, labels=range(102))
    fig, ax = plt.subplots(figsize=(12, 10)); image = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set(xlabel='Predicted class ID (1–102)', ylabel='True class ID (1–102)', title='Test confusion matrix')
    ticks = list(range(0, 102, 10)); ax.set_xticks(ticks, [str(t+1) for t in ticks]); ax.set_yticks(ticks, [str(t+1) for t in ticks]); fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return cm
