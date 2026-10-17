## Not: Logistic Regression — GridSearch Hyperparameter Tuning Denemesi

**Tarih:** Faz 5 model geliştirme aşaması
**Dosya:** `src/models/logistic_regression_tuned.py` (deneme tamamlandı, üretimde kullanılmıyor)

### Ne Denedik?

Logistic Regression'ın varsayılan parametrelerinin (`C=1.0, solver=lbfgs`) daha iyi bir kombinasyonla geçilip geçilemeyeceğini test etmek için **GridSearchCV + nested GroupKFold** deneyleri yapıldı.

**Denenen grid:**

    C : [0.01, 0.1, 1.0, 10.0, 100.0]
    solver : ['lbfgs', 'saga']


### Sonuçlar

| Versiyon | Macro-F1 | Kappa | Süre |
|---|---|---|---|
| LR Default (C=1.0, lbfgs) | **0.7606** | **0.8147** | **1.55s** |
| LR Tuned (lbfgs + saga) | 0.7541 | 0.8082 | 92.34s |
| LR Tuned (sadece lbfgs) | 0.7561 | 0.8097 | 9.45s |

### Neden Başarılı Olmadı?

1. **`saga` yakınsamadı** — `max_iter=1000` ile `saga` solver yeterli iterasyon bulamadı (`ConvergenceWarning`). Güvenilmez sonuçlar üretti.
2. **Orijinal parametre zaten optimaldi** — 5 fold'un 3'ünde GridSearch yine `C=1.0, lbfgs` seçti. Varsayılan değer zaten en iyisiymiş.
3. **LR'nin tavanı burada** — LR lineer bir karar sınırı çiziyor. EEG özelliklerinin non-lineer yapısını parametre değiştirerek aşmak mümkün değil. Asıl kazanım RF/XGBoost gibi non-lineer modellerde olacak.

### Karar

**Orijinal LR (C=1.0, lbfgs, max_iter=1000) baseline olarak korundu.** GridSearch denemesi iptal edildi. Hyperparameter tuning zamanı RF ve XGBoost'a harcandı — onların parametreleri (n_estimators, max_depth, learning_rate) performansı gerçekten etkiliyor.