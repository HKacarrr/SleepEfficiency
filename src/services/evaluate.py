import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)

CLASS_ORDER = ["W", "N1", "N2", "N3", "REM"]


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy":         accuracy_score(y_true, y_pred),
        "macro_f1":         f1_score(y_true, y_pred, average="macro"),
        "cohen_kappa":      cohen_kappa_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_ORDER))),
        "report":           classification_report(y_true, y_pred, target_names=CLASS_ORDER),
    }


def print_fold_summary(fold_metrics: list, model_name: str = "Model"):
    """
    Tum fold'larin ortalama + std sapmasini yazdirir.

    Args:
        fold_metrics : her fold icin evaluate_predictions() ciktisi listesi
        model_name   : gosterim icin model adi
    """
    keys = ["accuracy", "macro_f1", "cohen_kappa"]
    print(f"\n{'=' * 60}")
    print(f"{model_name} — {len(fold_metrics)}-Fold Sonuclari")
    print(f"{'=' * 60}")
    for k in keys:
        values = [m[k] for m in fold_metrics]
        print(f"  {k:<20}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    print(f"{'=' * 60}\n")