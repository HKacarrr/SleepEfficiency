import time
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from services.evaluate import evaluate_predictions, print_fold_summary, CLASS_ORDER


def run(data: dict, folds: list):
    """
    Args:
        data  : PreProcessing().process() ciktisi — index.py'den gelir
        folds : Split(data).get_folds() ciktisi  — index.py'den gelir
                Tum modeller ayni folds'u kullanir, karsilastirma adil olur.
    """
    X      = data["X"]
    y      = data["y"]
    groups = data["groups"]

    # ------------------------------------------------------------------
    # Model tanimi
    # ------------------------------------------------------------------
    # n_neighbors=7 : literatürde Sleep-EDF icin sik kullanilan deger
    # weights='distance': yakin komsuya daha fazla agirlik ver
    #                     (uniform'a gore genelde daha iyi sonuc)
    # metric='euclidean': StandardScaler sonrasi euclidean mantikli
    # n_jobs=-1: paralel hesaplama
    model = KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        metric="euclidean",
        n_jobs=-1,
    )

    # ------------------------------------------------------------------
    # 5-fold egitim ve degerlendirme
    # ------------------------------------------------------------------
    fold_metrics = []
    total_start  = time.time()

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        fold_start = time.time()

        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])

        metrics = evaluate_predictions(y[test_idx], y_pred)
        metrics["train_time_sec"] = round(time.time() - fold_start, 2)
        fold_metrics.append(metrics)

        test_subjects = sorted(set(groups[test_idx]))
        print(f"Fold {fold_id} | test: {test_subjects} | "
              f"accuracy={metrics['accuracy']:.4f} | "
              f"macro_f1={metrics['macro_f1']:.4f} | "
              f"kappa={metrics['cohen_kappa']:.4f} | "
              f"sure={metrics['train_time_sec']}s")

    total_time = round(time.time() - total_start, 2)

    # ------------------------------------------------------------------
    # Ozet
    # ------------------------------------------------------------------
    print_fold_summary(fold_metrics, model_name="K-Nearest Neighbors (k=7)")
    print(f"Toplam sure: {total_time}s\n")

    best_fold_id = int(np.argmax([m["macro_f1"] for m in fold_metrics]))
    print(f"En iyi fold (Fold {best_fold_id}) classification report:")
    print(fold_metrics[best_fold_id]["report"])

    print("Confusion Matrix (en iyi fold):")
    cm = fold_metrics[best_fold_id]["confusion_matrix"]
    print(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER))

    return fold_metrics


if __name__ == "__main__":
    from services.preprocessing import PreProcessing
    from services.split import Split

    data  = PreProcessing().process()
    folds = Split(data).get_folds()
    run(data, folds)