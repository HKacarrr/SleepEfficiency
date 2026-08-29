"""
test_split.py — Faz 4: Train/Test Split Stratejilerini Kiyaslama

Amac: Farkli split yontemlerini AYNI veri, AYNI model ile deneyip, hangisinin
gercekci (subject leakage'siz) sonuc verdigini SAYIYLA gostermek.

Denenen yontemler:
    A) train_test_split (rastgele, epoch bazli)      -> KOTU (subject leakage riski)
    B) GroupKFold (5-fold, kisi bazli)                -> ONERILEN
    C) GroupShuffleSplit (kisi bazli, tek split)      -> ALTERNATIF

Kullanim:
    python test_split.py
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


# ------------------------------------------------------------------
# 0) VERI HAZIRLIGI (Faz 3'teki ile ayni adimlar, kisaca tekrar)
# ------------------------------------------------------------------
CSV_PATH = "dataset/sleep_edf_epochs.csv"

RAW_TO_AASM = {
    "Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
    "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM",
}
CLASS_ORDER = ["W", "N1", "N2", "N3", "REM"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_ORDER)}
RANDOM_STATE = 0

df = pd.read_csv(CSV_PATH).dropna(subset=["Sleep_stage"])
df["stage_aasm"] = df["Sleep_stage"].map(RAW_TO_AASM)
df["stage_id"] = df["stage_aasm"].map(CLASS_TO_ID)

drop_cols = [c for c in df.columns if c.startswith("Event_marker")]
metadata_cols = ["subject", "Sleep_stage", "epoch_idx", "epoch_start_sec", "stage_aasm", "stage_id"]
feature_cols = [c for c in df.columns if c not in metadata_cols and c not in drop_cols]

X_raw = df[feature_cols]
y = df["stage_id"].values
groups = df["subject"].values

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

print(f"Veri hazir: X={X.shape}, y={y.shape}, {len(set(groups))} kisi\n")


# ------------------------------------------------------------------
# Ortak degerlendirme fonksiyonu (Faz 4 - Adim 11 ile ayni)
# ------------------------------------------------------------------
def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }


def train_quick_model(X_train, y_train, X_test, y_test):
    """Hizli bir RandomForest ile egit/test et (split yontemlerini kiyaslamak icin, sabit model)."""
    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate(y_test, y_pred)


results = []


# ====================================================================
# YONTEM A: train_test_split (RASTGELE, epoch bazli) — KOTU YONTEM
# ====================================================================
print("=" * 70)
print("YONTEM A: train_test_split (rastgele, epoch bazli)")
print("=" * 70)

X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.33, random_state=RANDOM_STATE
)

# Subject leakage kontrolu: kac kisi HEM train'de HEM test'te var?
subjects_in_train = set(groups_train)
subjects_in_test = set(groups_test)
overlap = subjects_in_train & subjects_in_test

print(f"Train'de {len(subjects_in_train)} kisi, Test'te {len(subjects_in_test)} kisi")
print(f"Hem train HEM test'te olan kisi sayisi (LEAKAGE): {len(overlap)} / {len(set(groups))}")
print(f"Cakisan kisiler: {sorted(overlap)}")

metrics_a = train_quick_model(X_train, y_train, X_test, y_test)
print(f"Sonuc: {metrics_a}")
results.append({"yontem": "A) train_test_split (rastgele)", "leakage_var_mi": "EVET", **metrics_a})


# ====================================================================
# YONTEM B: GroupKFold (5-fold, kisi bazli) — ONERILEN
# ====================================================================
print("\n" + "=" * 70)
print("YONTEM B: GroupKFold (5-fold, kisi bazli)")
print("=" * 70)

gkf = GroupKFold(n_splits=5)
fold_metrics = []
for fold_id, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    subjects_in_train = set(groups[train_idx])
    subjects_in_test = set(groups[test_idx])
    overlap = subjects_in_train & subjects_in_test
    assert len(overlap) == 0, f"Fold {fold_id}: leakage tespit edildi! {overlap}"

    m = train_quick_model(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
    fold_metrics.append(m)
    print(f"Fold {fold_id} (test kisi: {sorted(subjects_in_test)}): {m}")

metrics_b = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
metrics_b_std = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
print(f"\nOrtalama (5 fold): {metrics_b}")
print(f"Std sapma (5 fold): {metrics_b_std}")
results.append({"yontem": "B) GroupKFold (5-fold ortalama)", "leakage_var_mi": "HAYIR", **metrics_b})


# ====================================================================
# YONTEM C: GroupShuffleSplit (kisi bazli, tek split) — ALTERNATIF
# ====================================================================
print("\n" + "=" * 70)
print("YONTEM C: GroupShuffleSplit (kisi bazli, tek split, test_size=0.33)")
print("=" * 70)

gss = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups))

subjects_in_train = set(groups[train_idx])
subjects_in_test = set(groups[test_idx])
overlap = subjects_in_train & subjects_in_test

print(f"Train'de {len(subjects_in_train)} kisi, Test'te {len(subjects_in_test)} kisi")
print(f"Kisi cakismasi: {len(overlap)} (0 olmali)")

metrics_c = train_quick_model(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
print(f"Sonuc: {metrics_c}")
results.append({"yontem": "C) GroupShuffleSplit (tek split)", "leakage_var_mi": "HAYIR", **metrics_c})



# ====================================================================
# YONTEM D: train_test_split — subject SUTUNU cikarilmis, epoch bazli
# ====================================================================
print("=" * 70)
print("YONTEM D: train_test_split (subject sutunu cikarilmis, epoch bazli)")
print("=" * 70)

# Subject bilgisini feature'lardan cikart (zaten feature_cols icinde yok,
# ama groups array'ini train/test'e ayirarak kisi cakismasini gosterelim)
X_train_d, X_test_d, y_train_d, y_test_d, groups_train_d, groups_test_d = train_test_split(
    X, y, groups, test_size=0.33, random_state=RANDOM_STATE
)

subjects_in_train_d = set(groups_train_d)
subjects_in_test_d = set(groups_test_d)
overlap_d = subjects_in_train_d & subjects_in_test_d

print(f"Train'de {len(subjects_in_train_d)} kisi, Test'te {len(subjects_in_test_d)} kisi")
print(f"Kisi cakismasi: {len(overlap_d)} / {len(set(groups))}")

metrics_d = train_quick_model(X_train_d, y_train_d, X_test_d, y_test_d)
print(f"Sonuc: {metrics_d}")





# ====================================================================
# CONFUSION MATRIX KARSILASTIRMASI — A vs B
# ====================================================================
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

CLASS_ORDER = ["W", "N1", "N2", "N3", "REM"]

print("=" * 70)
print("CONFUSION MATRIX — YONTEM A (train_test_split, leakage VAR)")
print("=" * 70)

# Yontem A - ayni modeli tekrar egit
from sklearn.ensemble import RandomForestClassifier
X_train_a, X_test_a, y_train_a, y_test_a, _, _ = train_test_split(
    X, y, groups, test_size=0.33, random_state=RANDOM_STATE
)
model_a = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
model_a.fit(X_train_a, y_train_a)
y_pred_a = model_a.predict(X_test_a)

print(classification_report(y_test_a, y_pred_a, target_names=CLASS_ORDER))
cm_a = confusion_matrix(y_test_a, y_pred_a)
print("Confusion Matrix (satir=gercek, sutun=tahmin):")
print(pd.DataFrame(cm_a, index=CLASS_ORDER, columns=CLASS_ORDER))


print("\n" + "=" * 70)
print("CONFUSION MATRIX — YONTEM B (GroupKFold fold 3, leakage YOK)")
print("=" * 70)

# Yontem B - fold 3'u kullan (en iyi fold, karsilastirma icin)
gkf2 = GroupKFold(n_splits=5)
fold_splits = list(gkf2.split(X, y, groups))
train_idx_b, test_idx_b = fold_splits[3]

model_b = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
model_b.fit(X[train_idx_b], y[train_idx_b])
y_pred_b = model_b.predict(X[test_idx_b])

print(classification_report(y[test_idx_b], y_pred_b, target_names=CLASS_ORDER))
cm_b = confusion_matrix(y[test_idx_b], y_pred_b)
print("Confusion Matrix (satir=gercek, sutun=tahmin):")
print(pd.DataFrame(cm_b, index=CLASS_ORDER, columns=CLASS_ORDER))




# ====================================================================
# KIYASLAMA TABLOSU
# ====================================================================
print("\n" + "=" * 70)
print("KIYASLAMA TABLOSU")
print("=" * 70)

comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.round(4)
print(comparison_df.to_string(index=False))

comparison_df.to_csv("faz4_split_comparison.csv", index=False)
print("\nKarsilastirma tablosu kaydedildi: faz4_split_comparison.csv")

print("""
YORUM:
- Yontem A'nin skoru sisirilmis olabilir: ayni kisinin epoch'lari hem train hem
  test'te bulunuyor (leakage), model kismen "kisiyi taniyarak" tahmin ediyor.
- Yontem B ve C leakage'siz, dolayisiyla gercek dunyada ("hic gormedigi bir
  hastada") beklenecek performansi daha dogru yansitiyor.
- Yontem B (GroupKFold), tek bir train/test bolunmesine bagli kalmadigi icin
  (5 farkli kombinasyonun ortalamasi + std sapmasi) daha guvenilir/az varyansli
  bir tahmin verir -> Faz 4 icin ONERILEN yontem budur.
""")