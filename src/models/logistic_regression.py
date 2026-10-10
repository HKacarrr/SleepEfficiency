import time
import numpy as np

from sklearn.linear_model import LogisticRegression
from services.evaluate import evaluate_predictions, print_fold_summary, CLASS_ORDER


def run(data: dict, folds: list):
    X      = data["X"]
    y      = data["y"]
    groups = data["groups"]

    # ------------------------------------------------------------------
    # 2) Model tanimi
    # ------------------------------------------------------------------
    # max_iter=1000: LR'nin yakinsama icin yeterli iterasyon
    # class_weight='balanced': sinif dengesizligine karsi (Faz 4 karari)
    # random_state=0: tekrar uretilebilirlik
    # n_jobs=-1: tum CPU cekirdeklerini kullan
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=0,
        n_jobs=-1,
    )

    # ------------------------------------------------------------------
    # 3) 5-fold egitim ve degerlendirme
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
    # 4) Ozet
    # ------------------------------------------------------------------
    print_fold_summary(fold_metrics, model_name="Logistic Regression")
    print(f"Toplam egitim suresi: {total_time}s\n")

    # En iyi fold'un classification report'unu goster
    best_fold_id = int(np.argmax([m["macro_f1"] for m in fold_metrics]))
    print(f"En iyi fold (Fold {best_fold_id}) classification report:")
    print(fold_metrics[best_fold_id]["report"])

    print("Confusion Matrix (en iyi fold):")
    cm = fold_metrics[best_fold_id]["confusion_matrix"]
    import pandas as pd
    print(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER))

    return fold_metrics


if __name__ == "__main__":
    # Tek basina calistirildiginda kendi preprocessing'ini yapar
    from services.preprocessing import PreProcessing
    from services.split import Split

    data  = PreProcessing().process()
    folds = Split(data).get_folds()
    run(data, folds)