# Faz 4 — Deney Tasarımı ve Train/Test Stratejisi — Sonuç Raporu

**Kullanılan model:** RandomForest (100 ağaç, class_weight='balanced') — sadece split yöntemlerini karşılaştırmak için sabit tutuldu, asıl model karşılaştırması Faz 5'te yapılacak.
**Kullanılan script:** `test_split.py`

---

## Önce Temel Kavramlar

### Accuracy neden tek başına yeterli değil?

Bizim veri setimizde W (uyanıklık) epoch'ları toplamın **%67.4'ünü** oluşturuyor. Yani hiçbir şey öğrenmeyip her epoch'a "W" dese bile model %67 accuracy alır. Bu yüzden sınıf dengesizliği olan verilerde accuracy yanıltıcıdır — ek metrikler şart.

---

### Precision nedir?

> "Model o sınıfı tahmin ettiğinde ne kadar haklı?"

```
Precision(N1) = Doğru N1 tahmini / Modelin N1 dediği tüm epoch'lar
```

**Yüksek olması iyi.** Düşükse model o sınıfı "fazla tahmin ediyor" demek — olmadığı yerde de o sınıfı görüyor.

---

### Recall nedir?

> "Gerçekte o sınıftan olan epoch'ların kaçını yakaladı?"

```
Recall(N1) = Doğru N1 tahmini / Gerçekte N1 olan tüm epoch'lar
```

**Yüksek olması iyi.** Düşükse model o sınıfı "kaçırıyor" demek — N1 olduğu halde başka bir şey tahmin ediyor.

---

### F1-Score nedir?

Precision ve Recall'un dengeli ortalaması. İkisi de yüksekse yüksek çıkar; biri düşükse cezalandırır. Tek bir sayıyla "bu sınıfta ne kadar iyi?" sorusunu yanıtlar.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Max değer 1.0, yüksek olması iyi.**

---

### Macro-F1 nedir?

Her sınıfın F1 skorunu ayrı hesaplayıp **sınıf büyüklüğüne bakmadan** ortalamasını alır. Yani N1 gibi az örnekli ama klinik olarak önemli sınıfları da eşit ağırlıkla değerlendirir. **Sınıf dengesizliğinde accuracy yerine bu bakılmalı.**

---

### Cohen's Kappa nedir?

Tesadüfen doğru tahmin etme olasılığını düzelterek gerçek başarıyı ölçer. Literatürde insan-insan skorlayıcı uyumuyla (genelde 0.7-0.8 Kappa) kıyaslanabilir standart bir metrik. **Yüksek olması iyi, max 1.0.**

---

### Support nedir?

O sınıftan test setinde kaç gerçek örnek vardı — performans metriği değil, bağlam bilgisi. Support düşükse o sınıf için hesaplanan metrikler daha az güvenilir olabilir.

---

### Confusion Matrix nasıl okunur?

**Satır = Gerçek etiket | Sütun = Modelin tahmini**

```
        W     N1    N2   N3  REM
W    6079    81    18    4    5      <- Gerçekte W olan 6187 epoch
N1     14   205    75    3   41      <- Gerçekte N1 olan 338 epoch
N2      5    45  1415   42   35      <- Gerçekte N2 olan 1542 epoch
N3      2     0    35  385    0      <- Gerçekte N3 olan 422 epoch
REM     2    15    29    0  573      <- Gerçekte REM olan 619 epoch
```

- **Köşegen (sol üstten sağ alta)** = doğru tahminler ✅
- **Köşegen dışı** = hatalar ❌
- Örneğin W satırı - N1 sütunundaki **81**: Gerçekte W olan 81 epoch, model tarafından yanlışlıkla N1 olarak tahmin edilmiş.

---

## Neden GroupKFold? — train_test_split'in Sorunu

`train_test_split` her **epoch'u bağımsız** görüp rastgele karıştırır. Ama aynı kişinin epoch'ları birbirine çok benziyor (aynı EEG karakteristiği, aynı gece, aynı elektrot yerleşimi). Sonuç:

- SC4001E0'nin sabah 02:00 epoch'u → train
- SC4001E0'nin sabah 02:30 epoch'u → test

Model bu kişiyi "tanıyarak" tahmin ediyor — gerçek Sleep Stage örüntüsünü değil, kişinin EEG karakteristiğini öğrenmiş olabilir. Buna **subject leakage** denir.

**Klinikte gerçek senaryo:** Yeni bir hasta geliyor, model bu kişiyi hiç görmemiş. EEG'sinden Sleep Stage tahmin etmesi gerekiyor. Dolayısıyla modeli test ederken de "hiç görmediği kişi" üzerinde test etmek şart.

---

## Denenen Yöntemler ve Sonuçlar

### Yöntem A — train_test_split (rastgele, epoch bazlı) ❌

Her epoch bağımsız rastgele bölündü. Aynı kişinin epoch'ları hem train hem test'te.

```
10 kişiden 10'u hem train hem test'te (leakage: 10/10)
```

| Metrik | Değer |
|---|---|
| Accuracy | 0.9505 |
| Macro-F1 | 0.8595 |
| Cohen's Kappa | 0.9024 |

```
Confusion Matrix:
        W     N1    N2   N3  REM
W    6079    81    18    4    5
N1     14   205    75    3   41
N2      5    45  1415   42   35
N3      2     0    35  385    0
REM     2    15    29    0  573

Classification Report:
         precision  recall  f1-score  support
W         1.00      0.98      0.99     6187
N1        0.59      0.61      0.60      338
N2        0.90      0.92      0.91     1542
N3        0.89      0.91      0.90      422
REM       0.88      0.93      0.90      619
macro     0.85      0.87      0.86
```

**Sorun:** Skorlar şişirilmiş. Model kısmen "kişiyi tanıyarak" tahmin yapıyor — gerçek EEG örüntüsünü ne kadar öğrendiği belirsiz.

---

### Yöntem B — GroupKFold, 5-fold, kişi bazlı ✅ SEÇİLEN YÖNTEM

Her fold'da bazı kişilerin **tüm epoch'ları** test'e, geri kalanların tümü train'e alındı. Hiçbir kişi hem train hem test'te yer almadı.

```
Fold 0: test → SC4012E0, SC4041E0   — leakage: YOK
Fold 1: test → SC4001E0, SC4002E0   — leakage: YOK
Fold 2: test → SC4031E0, SC4032E0   — leakage: YOK
Fold 3: test → SC4021E0, SC4022E0   — leakage: YOK
Fold 4: test → SC4011E0, SC4042E0   — leakage: YOK
```

| Fold | Accuracy | Macro-F1 | Cohen's Kappa |
|---|---|---|---|
| Fold 0 | 0.8955 | 0.7332 | 0.8128 |
| Fold 1 | 0.9190 | 0.7319 | 0.8340 |
| Fold 2 | 0.8838 | 0.6219 | 0.7202 |
| Fold 3 | 0.9318 | 0.8036 | 0.8636 |
| Fold 4 | 0.9195 | 0.7915 | 0.8496 |
| **Ortalama** | **0.9099** | **0.7364** | **0.8161** |
| Std sapma | 0.0176 | 0.0644 | 0.0508 |

**Fold 2 neden düşük?** Test kümesinde SC4032E0 var — Faz 2'de tespit ettiğimiz anormal yüksek EMG genliğine sahip kişi. Bu kişi-bağımlı zayıflık, `GroupKFold` sayesinde görünür hale geldi. `train_test_split` kullansaydık bunu hiç fark etmezdik.

```
Confusion Matrix (Fold 3 — en iyi fold):
        W     N1   N2   N3  REM
W    3756    22    0    0    0
N1     20   131   57    0   70
N2      6    61  813   11   56
N3      9     0   51  154    0
REM     0     6   10    0  326

Classification Report (Fold 3):
         precision  recall  f1-score  support
W         0.99      0.99      0.99     3778
N1        0.60      0.47      0.53      278
N2        0.87      0.86      0.87      947
N3        0.93      0.72      0.81      214
REM       0.72      0.95      0.82      342
macro     0.82      0.80      0.80
```

---

### Yöntem C — GroupShuffleSplit, kişi bazlı, tek split

Kişi bazlı bölme ama tek bir train/test kombinasyonu (5 fold değil).

| Metrik | Değer |
|---|---|
| Accuracy | 0.8905 |
| Macro-F1 | 0.7463 |
| Cohen's Kappa | 0.7976 |

Leakage yok ama tek bir split olduğu için tek bir "şanslı/şanssız" kişi kombinasyonuna bağlı kalıyor — daha az güvenilir.

---

### Yöntem D — train_test_split, subject sütunu çıkarılmış

Önerilen "subject'i X'ten çıkarırsak düzelir mi?" sorusunun cevabı:

```
Yöntem A ile birebir aynı sonuç → accuracy: 0.9505, macro_f1: 0.8595
```

**Çünkü subject zaten feature_cols içinde değildi.** Sorun sütun değil, epoch'ların rastgele bölünmesinin kendisi. Subject'i X'ten çıkarmak leakage'i ortadan kaldırmıyor.

---

## A vs B Karşılaştırması — Recall Üzerinden

| Sınıf | A (leakage VAR) | B Fold 3 (leakage YOK) | Yorum |
|---|---|---|---|
| W | %98 | %99 | Her ikisinde de iyi, veri bol |
| N1 | %61 | %47 | B'de gerçek performans ortaya çıktı |
| N2 | %92 | %86 | Makul düşüş |
| N3 | %91 | %72 | B'de belirgin düşüş |
| REM | %93 | %95 | B'de daha iyi — ilginç |

N1 ve N3 recall'u leakage olmadan ciddi düşüyor — model bu sınıfları kısmen kişiyi "ezberleyerek" doğru tahmin ediyordu.

---

## Genel Bulgular ve Yorumlar

**N1 her zaman en zor sınıf olacak.** Hem precision hem recall ~0.47-0.61 arasında — iki sebebi var:
1. Fizyolojik: N1, uyanıklıktan uykuya geçiş evresi. EEG sinyali hem W'ye hem N2'ye benziyor, net bir imza yok.
2. Veri azlığı: Support sadece 278-338, diğer sınıflara kıyasla çok az örnek.

**Sonuçlar "çok iyi değil" mi?** Macro-F1 ~0.73 ve Kappa ~0.82 aslında bu problem için kötü sayılmaz — literatürde insan-insan skorlayıcı uyumu da ~0.70-0.80 Kappa aralığında. Ama kesinlikle iyileştirme payı var. Faz 5'te farklı algoritmalar (SVM, XGBoost) denenerek daha iyi sonuç çıkıp çıkmadığı görülecek.

---

## Faz 4 Kontrol Listesi

- [x] **Kişi bazlı ayrım şart**: `GroupKFold`, grup = `subject`
  - Kanıtlandı: `train_test_split` Macro-F1'i yapay olarak ~0.12 şişiriyor (0.86 → 0.74). Subject sütununu X'ten çıkarmak leakage'i çözmüyor — sorun bölünme mantığının kendisinde.
- [x] Kaç fold kullanılacağına karar ver → **5-fold seçildi**
  - 10 kişi / 5 fold = fold başına 2 kişi test'te. Daha fazla fold (örn. 10) fold başına 1 kişi bırakır, tek kişinin skoru tüm fold'u domine eder.
- [x] Sınıf dengesizliğine karşı strateji seç → **`class_weight='balanced'` seçildi (birincil)**
  - Hesaplanan ağırlıklar: W:0.297, N1:5.492, N2:1.147, N3:4.356, REM:2.902
  - SMOTE, Faz 5'te ayrı bir karşılaştırma olarak denenecek.
- [x] Değerlendirme metrikleri belirlendi:
  - **Accuracy** — referans için, tek başına yeterli değil (sınıf dengesizliği nedeniyle yanıltıcı olabilir)
  - **Macro-F1** — asıl performans metriği, sınıf dengesizliğine duyarlı
  - **Cohen's Kappa** — insan skorlayıcı uyumuyla kıyaslanabilir literatür standardı
  - **Confusion Matrix** — hangi evreler birbirine karışıyor (N1↔W, N1↔N2 en sık karışan çiftler)

**Çıktı:** ✅ Sabit, tekrar üretilebilir deney protokolü hazır (`random_state=0`, `GroupKFold(n_splits=5)`, `class_weight='balanced'`, `evaluate_predictions()` fonksiyonu `index.py`'de tanımlı).

---

## Faz 5'e Taşınan Açık Notlar

1. RandomForest baseline olarak iyi başlangıç ama Macro-F1 ~0.73 — SVM, XGBoost, LightGBM ile karşılaştırılacak.
2. N1 sınıfı her algoritmada sorun çıkarmaya devam edecek — komşu epoch özelliği (Faz 3'te ertelenen opsiyonel adım) burada fark yaratabilir.
3. SC4032E0 (anormal EMG) Fold 2'yi tek başına ciddi düşürüyor — Faz 5'te her modelin Fold 2 performansı ayrıca izlenecek.
4. SMOTE ile `class_weight='balanced'` karşılaştırması Faz 5'te yapılacak.