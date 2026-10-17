"""
src/models/svm.py — Faz 5: Support Vector Machine (RBF Kernel)

Neden SVM (RBF):
    - Literatürde Sleep-EDF üzerinde en sık kullanilan klasik ML yontemlerinden biri
    - RBF (Radial Basis Function) kernel: lineer ayrilamayan siniflar icin
      yuksek boyutlu bir uzayda sinif sinirlari cizer — EEG ozellikleri icin uygun
    - StandardScaler olmadan cok kotu calisir — Faz 3'teki olcekleme bu yuzden kritik
    - Dezavantaj: buyuk veri setlerinde yavas (27k epoch icin SVC yerine LinearSVC
      veya kernel approximation dusunulebilir, ama oncelikle standart SVC deneyelim)

    Literatur referansi:
    - Automated Multimodal Sleep Staging (PMC13511945):
      SVM-RBF -> Macro-F1: 0.7322, Kappa: 0.6784 (Sleep-EDF Cassette, 5-fold)

Kullanim:
    python src/models/svm.py     <- tek basina
    index.py uzerinden           <- data/folds disaridan gelir
"""
import time
import numpy as np
import pandas as pd

from sklearn.svm import SVC
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
    # kernel='rbf'      : Radial Basis Function — lineer olmayan siniflar icin
    # C=1.0             : regularizasyon parametresi
    #                     kucuk C -> genis margin, bazi hatalara izin verir (underfitting riski)
    #                     buyuk C -> dar margin, tum noktalari dogru siniflamaya calisir (overfitting riski)
    # gamma='scale'     : kernel genisligi — 'scale' = 1 / (n_features * X.var()) otomatik hesaplar
    # class_weight='balanced': sinif dengesizligine karsi (Faz 4 karari)
    # random_state=0    : tekrar uretilebilirlik
    model = SVC(
        kernel="rbf",
        C=100.0,
        gamma=0.001,
        class_weight="balanced",
        random_state=0,
    )

    # ------------------------------------------------------------------
    # 5-fold egitim ve degerlendirme
    # ------------------------------------------------------------------
    fold_metrics = []
    total_start  = time.time()

    print("Not: SVM buyuk veri setlerinde yavas calisabilir, lutfen bekleyin...\n")

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
    print_fold_summary(fold_metrics, model_name="SVM (RBF kernel)")
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