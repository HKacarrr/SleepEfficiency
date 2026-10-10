import time
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from services.evaluate import evaluate_predictions, print_fold_summary, CLASS_ORDER


def run(data: dict, folds: list):
    """
    Args:
        data  : PreProcessing().process() ciktisi — index.py'den gelir
        folds : Split(data).get_folds() ciktisi  — index.py'den gelir
                Tum modeller ayni folds'u kullanir, karsilastirma adil olur.
    """
    X            = data["X"]
    y            = data["y"]
    groups       = data["groups"]
    feature_cols = data["feature_cols"]

    # ------------------------------------------------------------------
    # Model tanimi
    # ------------------------------------------------------------------
    # n_estimators=300  : 300 agac — 100'den iyi, 500'den hizli
    # max_depth=None    : agaclar tam derinlige kadar buyur (RF icin standart)
    # class_weight='balanced': sinif dengesizligine karsi (Faz 4 karari)
    # random_state=0    : tekrar uretilebilirlik
    # n_jobs=-1         : tum CPU cekirdeklerini kullan
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=0,
        n_jobs=-1,
    )

    # ------------------------------------------------------------------
    # 5-fold egitim ve degerlendirme
    # ------------------------------------------------------------------
    fold_metrics    = []
    total_start     = time.time()

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
    print_fold_summary(fold_metrics, model_name="Random Forest")
    print(f"Toplam sure: {total_time}s\n")

    best_fold_id = int(np.argmax([m["macro_f1"] for m in fold_metrics]))
    print(f"En iyi fold (Fold {best_fold_id}) classification report:")
    print(fold_metrics[best_fold_id]["report"])

    print("Confusion Matrix (en iyi fold):")
    cm = fold_metrics[best_fold_id]["confusion_matrix"]
    print(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER))

    # ------------------------------------------------------------------
    # Feature Importance (son fold'un modeli uzerinden)
    # makalede "hangi sinyal/bant en ayirt edici?" sorusuna cevap
    # ------------------------------------------------------------------
    print("\nFeature Importance (Top 10):")
    importance_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(importance_df.head(10).to_string(index=False))

    return fold_metrics, importance_df


if __name__ == "__main__":
    from services.preprocessing import PreProcessing
    from services.split import Split

    data  = PreProcessing().process()
    folds = Split(data).get_folds()
    run(data, folds)