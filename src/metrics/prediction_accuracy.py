from torchmetrics.functional.classification import accuracy, confusion_matrix


def compute_metrics(y_true, y_pred):
    acc = accuracy(y_pred, y_true, task="multiclass", num_classes=10)
    cm = confusion_matrix(y_pred, y_true, task="multiclass", num_classes=10)
    return acc, cm
