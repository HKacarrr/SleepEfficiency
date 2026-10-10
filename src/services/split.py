from sklearn.model_selection import GroupKFold


class Split:

    N_SPLITS     = 5   # 10 kisi / 5 fold = fold basina 2 kisi test'te
    RANDOM_STATE = 0   # tekrar uretilebilirlik icin sabit

    def __init__(self, data: dict):
        self.X      = data["X"]
        self.y      = data["y"]
        self.groups = data["groups"]

    def get_folds(self) -> list:
        """
        GroupKFold ile kisi bazli train/test indekslerini uretir.
        Ayni kisinin epoch'lari asla hem train hem test'te olmaz (subject leakage yok).

        Returns:
            list of (train_idx, test_idx) tuples — N_SPLITS adet
        """
        gkf          = GroupKFold(n_splits=self.N_SPLITS)
        fold_indices = list(gkf.split(self.X, self.y, self.groups))

        print(f"[Split] GroupKFold hazir: {self.N_SPLITS} fold, "
              f"{len(set(self.groups))} kisi, subject leakage: YOK")

        return fold_indices