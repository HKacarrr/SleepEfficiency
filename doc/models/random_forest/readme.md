## Not: Random Forest — GridSearch Hyperparameter Tuning

**Dosya:** `src/models/random_forest.py`

### Ne Denedik?

Random Forest için en kritik iki parametreyi GridSearchCV + nested GroupKFold ile taradık.

**Denenen grid:**
```
n_estimators : [100, 300, 500]
max_depth    : [None, 10, 20, 30]
```
12 kombinasyon x 3 inner fold = 36 fit, toplam süre: 208.83s

### Sonuçlar

| Versiyon | Macro-F1 | Kappa | Süre |
|---|---|---|---|
| RF Default (n_estimators=300, max_depth=None) | 0.7364 | 0.8161 | ~60s |
| **RF Tuned (n_estimators=100, max_depth=10)** | **0.7429** | **0.7960** | 208.83s |

### Seçilen Parametreler

| Fold | n_estimators | max_depth |
|---|---|---|
| Fold 0 | 100 | 10 |
| Fold 1 | 100 | 10 |
| Fold 2 | 500 | 10 |
| Fold 3 | 100 | 10 |
| Fold 4 | 100 | 10 |

5 fold'un 5'inde de `max_depth=10` seçildi — çok tutarlı. Ağaçların derinliği sınırlandırılınca model daha iyi genelleme yapıyor (overfitting azalıyor).

### Dikkat Çeken Nokta

Macro-F1 hafif arttı (+0.007) ama Kappa **geriledi** (0.8161 → 0.7960). Bu çelişkili görünüyor — sebebi Fold 2'nin (SC4031E0, SC4032E0) Macro-F1'inin 0.6600'e düşmesi ve yüksek std sapması (±0.0534). Faz 2'den hatırlanan SC4032E0'nin anormal EMG değerleri bu fold'u zorluyor, `max_depth=10` bu kişide yetersiz kalıyor olabilir.

### Karar

GridSearch sonucunda bulunan `max_depth=10, n_estimators=100` parametreleri **kalıcı olarak `random_forest.py`'ye yazıldı.** Macro-F1'de küçük bir iyileşme sağlandı. Ancak Kappa'nın gerilemiş olması ve yüksek fold varyansı (±0.0534) nedeniyle RF bu veri setinde SVM (Macro-F1: 0.7674, Kappa: 0.8181) ve LR'nin (Macro-F1: 0.7606, Kappa: 0.8147) gerisinde kaldı.

### Feature Importance (Top 10)

| Öznitelik | Önem |
|---|---|
| EEG_Fpz-Cz_delta_power | 0.1219 |
| EEG_Fpz-Cz_std | 0.0887 |
| EMG_submental_mean | 0.0696 |
| EEG_Pz-Oz_delta_power | 0.0675 |
| EMG_submental_min | 0.0482 |
| EOG_horizontal_std | 0.0476 |
| Temp_rectal_mean | 0.0467 |
| EOG_horizontal_max | 0.0458 |
| EOG_horizontal_min | 0.0456 |
| Resp_oro-nasal_std | 0.0385 |

**Delta power (EEG_Fpz-Cz) açık ara en ayırt edici öznitelik** — Faz 2 EDA'daki beklentiyle uyumlu. Temp_rectal_mean ilk 10'da yer aldı, Faz 3'teki "tutulacak" kararı burada doğrulandı.