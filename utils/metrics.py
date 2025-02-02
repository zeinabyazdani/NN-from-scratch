
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def report_result(y_true, y_pred):
    """
    Report result of testing phase.

    Args:
        y_true: Actual labels of test data
        y_pred: predicted labels of test data
    """

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec  = recall_score(y_true, y_pred, average='macro')
    f1   = f1_score(y_true, y_pred, average='macro')
    cm   = confusion_matrix(y_true, y_pred)

    print('-' * 100)
    print(f"Accuracy score: {acc:.4f}")
    print(f"Precision score: {prec:.4f}")
    print(f"Recall score: {rec:.4f}")
    print(f"f1_score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print('-' * 100)
