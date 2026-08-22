"""
index.py — Faz 3: On Isleme ve Oznitelik Muhendisligi (BRUTE FORCE / duz akis)

Class yok, hepsi yukaridan asagiya sirayla calisiyor. Amac: once akisi net
gormek. Servis/class mimarisine bolme islemi sonra yapilacak.

Bu script X/y/groups'u ayri dosyalara (.npy/.joblib) KAYDETMEZ. Cunku Faz 4/5
zaten ayni DataFrame'i bu script icinde/devaminda kullanacak. Tek cikti,
dokumantasyon amacli, bir kerelik uretilen okunur bir CSV (Faz_3 klasorune
konulacak).

Kullanim:
    python index.py
(CSV yolu asagida CSV_PATH sabitinde, istersen degistir)
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------
# 0) AYARLAR
# ------------------------------------------------------------------
CSV_PATH = "Dataset/sleep_edf_epochs.csv"
DOC_OUTPUT_CSV = "faz3_processed_epochs.csv"  # bir kerelik, dokumantasyon amacli cikti

LABEL_COL = "Sleep_stage"

# N3 ve N4 birleştirme işlemi. Sleep stage 3 ve 4 ikiside N3'e eşitlenerek birleştirilmiş oldu - R&K etiketlerini AASM'e indirgeme (3+4 -> N3)
RAW_TO_AASM = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}
CLASS_ORDER = ["W", "N1", "N2", "N3", "REM"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_ORDER)}

# Faz 2 EDA'da dogrulanan karar: Event_marker'in Sleep_stage ile korelasyonu
# ~0.01 (bilgi tasimiyor) -> cikar. Temp_rectal ~0.52 (bilgi tasiyor) -> tut.
DROP_COLUMN_PREFIXES = ["Event_marker"]

# X'e girmeyecek, ama veri setinde kalmasi gereken (id/etiket) sutunlar
METADATA_COLS = ["subject", "Sleep_stage", "epoch_idx", "epoch_start_sec"]


# ------------------------------------------------------------------
# 1) VERI YUKLEME
# ------------------------------------------------------------------
print("=== Faz 3 - Brute Force Pipeline ===\n")

df = pd.read_csv(CSV_PATH)
print(f"[1] CSV yuklendi: {df.shape[0]} satir, {df.shape[1]} sutun")


# ------------------------------------------------------------------
# 2) ETIKETSIZ EPOCH TEMIZLIGI
# ------------------------------------------------------------------
before = len(df)
df = df.dropna(subset=[LABEL_COL]) # Sleep Stage değeri NaN olan satırları çıkar. Eğitimde kullanılamaz çünkü. W-N1-N2-N3-REM gibi değerler olmalı sadece
dropped = before - len(df)
print(f"[2] Etiketsiz epoch temizligi: {dropped} satir cikarildi (kalan: {len(df)})")


# ------------------------------------------------------------------
# 3) N3/N4 BIRLESTIRME (AASM STANDARDI)
# ------------------------------------------------------------------
df["stage_aasm"] = df[LABEL_COL].map(RAW_TO_AASM)

unmapped = df["stage_aasm"].isna().sum()
if unmapped > 0:
    raise ValueError(f"{unmapped} epoch bilinmeyen etikete sahip: "
                      f"{df.loc[df['stage_aasm'].isna(), LABEL_COL].unique()}")

df["stage_id"] = df["stage_aasm"].map(CLASS_TO_ID)

print("[3] AASM birlestirme sonrasi sinif dagilimi:")
print(df["stage_aasm"].value_counts().to_string())


# ------------------------------------------------------------------
# 4) DUSUK BILGI DEGERLI SUTUNLARI CIKARMA (Event_marker)
# ------------------------------------------------------------------
all_cols = df.columns.tolist()
cols_to_drop = [c for c in all_cols if any(c.startswith(p) for p in DROP_COLUMN_PREFIXES)]

print(f"\n[4] Dusuk bilgi degeri nedeniyle cikarilan sutunlar: {cols_to_drop}")
df = df.drop(columns=cols_to_drop)


# ------------------------------------------------------------------
# 5) X / y / groups AYRIMI
# ------------------------------------------------------------------
exclude_from_features = set(METADATA_COLS) | {"stage_aasm", "stage_id"}
feature_cols = [c for c in df.columns if c not in exclude_from_features]

X_raw = df[feature_cols]
y = df["stage_id"].values
groups = df["subject"].values

print(f"\n[5] X/y/groups ayrildi:")
print(f"    X shape   : {X_raw.shape}")
print(f"    y shape   : {y.shape}")
print(f"    groups    : {len(set(groups))} benzersiz kisi")
print(f"    feature_cols ({len(feature_cols)} adet): {feature_cols}")


# ------------------------------------------------------------------
# 6) OLCEKLEME (StandardScaler)
# ------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"\n[6] Olcekleme tamamlandi -> mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")

# NOT: X_scaled, y, groups burada bellekte hazir duruyor. Faz 4/5 (split,
# model egitimi) bu script'in devamina veya ayni oturumdaki bir sonraki
# hucreye/dosyaya bu degiskenleri direkt kullanarak devam edebilir.
# Ayrica dosyaya kaydetmeye gerek yok.


# ------------------------------------------------------------------
# 7) BIR KERELIK, DOKUMANTASYON AMACLI CSV CIKTISI
# ------------------------------------------------------------------
doc_df = pd.DataFrame(X_scaled, columns=feature_cols)
doc_df["stage_id"] = y
doc_df["stage_name"] = df["stage_aasm"].values
doc_df["subject"] = groups

doc_df.to_csv(DOC_OUTPUT_CSV, index=False)
print(f"\n[7] Dokumantasyon amacli CSV kaydedildi: {DOC_OUTPUT_CSV} "
      f"({doc_df.shape[0]} satir, {doc_df.shape[1]} sutun)")
print("    Bu dosya sadece Faz_3 klasorune / makaleye referans icin;")
print("    egitim asamasinda tekrar kullanilmayacak.")

print("\n=== Bitti ===")