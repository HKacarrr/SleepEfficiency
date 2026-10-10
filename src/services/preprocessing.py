import pandas as pd
from sklearn.preprocessing import StandardScaler


class PreProcessing:
    # R&K -> AASM etiket donusumu (3+4 -> N3)
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

    # Faz 2 EDA kararı: Event_marker Sleep_stage ile korelasyonu ~0.01 -> cikar
    DROP_COLUMN_PREFIXES = ["Event_marker"]

    # X'e girmeyecek metadata sutunlari
    METADATA_COLS = ["subject", "Sleep_stage", "epoch_idx", "epoch_start_sec"]

    def __init__(self, csv_path: str = "Dataset/sleep_edf_epochs.csv"):
        self.csv_path = csv_path

    def process(self) -> dict:
        """
        Tum on isleme adimlarini sirayla calistirir.

        Returns:
            dict with keys:
                "X"           : np.ndarray, scaled feature matrix (n_epochs x n_features)
                "y"           : np.ndarray, class ids (0=W,1=N1,2=N2,3=N3,4=REM)
                "groups"      : np.ndarray, subject names per epoch (GroupKFold icin)
                "feature_cols": list[str],  X'teki sutun isimleri
                "scaler"      : StandardScaler, fit edilmis scaler
        """
        df = self._load()
        df = self._clean_labels(df)
        df = self._merge_stages(df)
        df = self._drop_low_info_cols(df)
        X_raw, y, groups, feature_cols = self._split_X_y_groups(df)
        X_scaled, scaler = self._scale(X_raw)

        print(f"[PreProcessing] Tamamlandi: X={X_scaled.shape}, "
              f"y={y.shape}, {len(set(groups))} kisi, {len(feature_cols)} oznitelik")

        return {
            "X": X_scaled,
            "y": y,
            "groups": groups,
            "feature_cols": feature_cols,
            "scaler": scaler,
        }

    # ------------------------------------------------------------------
    # Adim 1: CSV yukleme
    # ------------------------------------------------------------------
    def _load(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        print(f"[PreProcessing] CSV yuklendi: {df.shape[0]} satir, {df.shape[1]} sutun")
        return df

    # ------------------------------------------------------------------
    # Adim 2: Etiketsiz (NaN) epoch temizligi
    # ------------------------------------------------------------------
    def _clean_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=["Sleep_stage"])
        dropped = before - len(df)
        if dropped > 0:
            print(f"[PreProcessing] {dropped} etiketsiz epoch cikarildi.")
        return df

    # ------------------------------------------------------------------
    # Adim 3: N3/N4 birlestirme (AASM standardi)
    # ------------------------------------------------------------------
    def _merge_stages(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["stage_aasm"] = df["Sleep_stage"].map(self.RAW_TO_AASM)

        unmapped = df["stage_aasm"].isna().sum()
        if unmapped > 0:
            raise ValueError(
                f"{unmapped} epoch bilinmeyen etikete sahip: "
                f"{df.loc[df['stage_aasm'].isna(), 'Sleep_stage'].unique()}"
            )

        df["stage_id"] = df["stage_aasm"].map(self.CLASS_TO_ID)
        return df

    # ------------------------------------------------------------------
    # Adim 4: Dusuk bilgi degerli sutunlari cikar (Event_marker)
    # ------------------------------------------------------------------
    def _drop_low_info_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [
            c for c in df.columns
            if any(c.startswith(p) for p in self.DROP_COLUMN_PREFIXES)
        ]
        if cols_to_drop:
            print(f"[PreProcessing] Cikarilan sutunlar: {cols_to_drop}")
        return df.drop(columns=cols_to_drop)

    # ------------------------------------------------------------------
    # Adim 5: X / y / groups ayrimi
    #
    # y nedir?
    #   Bizim tahmin etmeye calistigimiz deger Sleep Stage'dir.
    #   Ancak model string ile degil, sayi ile calisir. Bu yuzden
    #   her sinifa bir id atanmistir:
    #       W   -> 0
    #       N1  -> 1
    #       N2  -> 2
    #       N3  -> 3
    #       REM -> 4
    #   Dolayisiyla y = stage_id, yani Sleep Stage'in sayisal karsiligi.
    #   Model tahmin ettikten sonra bu sayilar CLASS_ORDER listesiyle
    #   tekrar sinif ismine donusturulur:
    #       y_pred = [3, 0, 2, ...] -> ["N3", "W", "N2", ...]
    # ------------------------------------------------------------------
    def _split_X_y_groups(self, df: pd.DataFrame):
        exclude = set(self.METADATA_COLS) | {"stage_aasm", "stage_id"}
        feature_cols = [c for c in df.columns if c not in exclude]

        X_raw      = df[feature_cols]
        y          = df["stage_id"].values
        groups     = df["subject"].values

        return X_raw, y, groups, feature_cols

    # ------------------------------------------------------------------
    # Adim 6: Olcekleme (StandardScaler)
    # ------------------------------------------------------------------
    def _scale(self, X_raw: pd.DataFrame):
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        return X_scaled, scaler