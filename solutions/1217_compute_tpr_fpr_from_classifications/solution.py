from collections import Counter

def compute_tpr_fpr(y_true, y_pred):
    """
    Compute TPR and FPR from true and predicted binary labels.

    Args:
        y_true (array-like): Ground-truth labels (0 or 1).
        y_pred (array-like): Predicted labels (0 or 1).

    Returns:
        tuple: (tpr, fpr) as Python floats.
    """
    # TODO: compute TP, FN, FP, TN then TPR and FPR
    data = [(t, p) for t, p in zip(y_true, y_pred)]
    c = Counter(data)
    tp, fn, fp, tn = c[(1, 1)], c[(1, 0)], c[(0, 1)], c[(0, 0)]

    tpr = 0 if tp + fn == 0 else tp/(tp + fn)
    fpr = 0 if fp + tn == 0 else fp/(fp + tn)

    return tpr, fpr
