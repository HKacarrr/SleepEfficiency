## Not: SVM (RBF Kernel) — GridSearch Hyperparameter Tuning

**Dosya:** `src/models/svm.py`

### Ne Denedik?

SVM'in varsayılan parametrelerinin (`C=1.0, gamma='scale'`) daha iyi bir kombinasyonla geçilip geçilemeyeceğini test etmek için **GridSearchCV + nested GroupKFold** deneyleri yapıldı.

**Denenen grid:**
```
C     : [0.1, 1.0, 10.0, 100.0]
gamma : ['scale', 'auto', 0.001, 0.01]
```
16 kombinasyon x 3 inner fold = 48 fit

### Sonuçlar

| Versiyon | Macro-F1 | Kappa | Süre |
|---|---|---|---|
| SVM Default (C=1.0, gamma=scale) | 0.7364 | 0.8161 | 9s |
| **SVM Tuned (C=100.0, gamma=0.001)** | **0.7674** | **0.8181** | 7.88s |

### Neden Bu Parametreler Seçildi?

5 fold'un 4'ünde `C=100.0, gamma=0.001` seçildi — çok tutarlı bir sonuç. `gamma=0.001` küçük bir değer, yani kernel çok geniş bir alanı etkiliyor. Bu veri setinin EEG öznitelikleri için doğru — sınıflar arasındaki sınır geniş tutulunca daha iyi genelleme yapıyor. `C=100.0` ise modelin hatalara karşı daha az toleranslı olduğu anlamına geliyor, yani sınıf sınırlarını daha sıkı çiziyor.

### Karar

**`C=100.0, gamma=0.001` kalıcı parametre olarak `svm.py`'ye yazıldı.** GridSearch kodu kaldırıldı. LR'den farklı olarak burada GridSearch gerçekten fark yarattı — Macro-F1'de +0.031 iyileşme sağlandı ve aynı zamanda literatürdeki SVM-RBF referans sonucunu (Macro-F1: 0.7322) geçtik.