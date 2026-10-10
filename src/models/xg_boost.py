"""
src/models/xgboost_model.py — Faz 5: XGBoost (eXtreme Gradient Boosting)

Neden XGBoost:
    - Her yeni agac, bir oncekinin HATALARINI duzeltmeye odaklanir (boosting)
    - Sleep-EDF uzerinde son literaturde en yuksek performansi veren klasik ML yontemi
    - Feature importance verir — Random Forest ile karsilastirmak icin ideal
    - Hiperparametre optimizasyonuna cok duyarli — dogru ayarlarla RF ve SVM'yi gecer

    Literatur referanslari:
    - XGBoost + Feature Selection (Bioengineering 2025):
      accuracy: 87.0%, F1: 86.6%, Kappa: 0.81 (Sleep-EDF)
    - Sequential XGBoost (Springer 2025):
      accuracy: 91.5%, Kappa: 0.826 (Sleep-EDF)
    - XGBoost vs RF vs SVM vs KNN karsilastirmasi:
      XGBoost en iyi — accuracy: 81.34%, Kappa: 0.7388

Kullanim:
    python src/models/xgboost_model.py    <- tek basina
    index.py uzerinden                    <- data/folds disaridan gelir
"""
import time
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
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
    # n_estimators=300      : agac sayisi — literaturde 200-500 arasi sik kullanilir
    # max_depth=6           : agac derinligi — XGBoost icin standart baslangic degeri
    # learning_rate=0.1     : adim buyuklugu — kucuk = yavaş ama stabil ogrenme
    # subsample=0.8         : her agac icin verinin %80'ini kullan — overfitting onler
    # colsample_bytree=0.8  : her agac icin ozelliklerin %80'ini kullan — cesitlilik saglar
    # use_label_encoder=False ve eval_metric='mlogloss': cok sinifli problem icin standart
    # class_weight:         : XGBoost'ta direkt class_weight yok,
    #                         sample_weight ile Faz 4 kararini uyguluyoruz (asagida)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=0,
        n_jobs=-1,
        verbosity=0,
    )

    # ------------------------------------------------------------------
    # Sinif agirliklari (Faz 4 karari: class_weight='balanced' karsiligi)
    # XGBoost'ta sample_weight parametresiyle uygulanir
    # ------------------------------------------------------------------
    from sklearn.utils.class_weight import compute_sample_weight

    # ------------------------------------------------------------------
    # 5-fold egitim ve degerlendirme
    # ------------------------------------------------------------------
    fold_metrics = []
    total_start  = time.time()

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        fold_start = time.time()

        # Her fold icin sample_weight hesapla (sadece train seti icin)
        sample_weights = compute_sample_weight(class_weight="balanced", y=y[train_idx])

        model.fit(X[train_idx], y[train_idx], sample_weight=sample_weights)
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
    print_fold_summary(fold_metrics, model_name="XGBoost")
    print(f"Toplam sure: {total_time}s\n")

    best_fold_id = int(np.argmax([m["macro_f1"] for m in fold_metrics]))
    print(f"En iyi fold (Fold {best_fold_id}) classification report:")
    print(fold_metrics[best_fold_id]["report"])

    print("Confusion Matrix (en iyi fold):")
    cm = fold_metrics[best_fold_id]["confusion_matrix"]
    print(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER))

    # ------------------------------------------------------------------
    # Feature Importance
    # Son fold modeli uzerinden — RF ile karsilastirmak icin
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