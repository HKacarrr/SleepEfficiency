# Faz 3 — Ön İşleme ve Öznitelik Mühendisliği — Sonuç Raporu

**Kullanılan script:** `index.py` (Adım 1-7)
**Girdi:** `sleep_edf_epochs.csv` (27.597 satır, 40 sütun)
**Çıktı:** Model-hazır `X` (27.597×32), `y` (27.597,), `groups` (27.597,) — bellekte hazır

---

## Önce Temel Kavramlar

### StandardScaler nedir, neden gerekli?

Veri setindeki öznitelikler çok farklı ölçeklerde:
- `delta_power` → ~1e-9 (çok küçük sayılar)
- `Temp_rectal_mean` → ~36-37 (derece cinsinden)

Mesafe veya gradyan tabanlı algoritmalar (SVM, KNN, MLP) büyük sayılı sütunlara yanlış şekilde daha fazla önem verir. `StandardScaler` her sütunu **ortalama=0, std=1** olacak şekilde dönüştürür — böylece tüm öznitelikler eşit ölçekte yarışır.

```
z = (değer - ortalama) / standart_sapma
```

RandomForest/XGBoost gibi ağaç tabanlı modeller ölçeklemeye ihtiyaç duymaz ama zarar da vermez — script'te tüm modeller için tek tip pipeline kurmak adına uygulandı.

---

### N3/N4 birleştirme neden yapıldı?

R&K (Rechtschaffen & Kales) sisteminde evre 3 ve evre 4 ayrı tutulur. AASM (American Academy of Sleep Medicine) standardında ikisi de "yavaş dalga uykusu (derin uyku)" olarak tek bir N3 evresinde birleştirilir. Bu çalışma AASM standardını benimsedi — hem klinik geçerlilik hem de sınıf sayısını azaltarak modeli basitleştirmek için.

---

### X, y, groups nedir?

- **X:** Modelin öğreneceği tüm sayısal öznitelikler — her satır bir epoch, her sütun bir sinyal özelliği (32 sütun).
- **y:** Modelin tahmin etmeyi öğreneceği hedef — her epoch'un AASM sınıf numarası (0=W, 1=N1, 2=N2, 3=N3, 4=REM).
- **groups:** Her epoch'un hangi kişiye ait olduğu bilgisi — Faz 4'te `GroupKFold` ile kişi bazlı train/test ayrımı yapabilmek için gerekli. Bu olmadan aynı kişinin epoch'ları hem train hem test'e düşer (subject leakage).

---

## Adım Adım Yapılanlar

### Adım 1 — Veri Yükleme

```
Girdi : sleep_edf_epochs.csv
Çıktı : 27.597 satır, 40 sütun
```

---

### Adım 2 — Etiketsiz Epoch Temizliği

`Sleep_stage` sütununda NaN olan epoch'lar çıkarıldı.

```
Çıkarılan satır: 0
Kalan         : 27.597
```

Bu veri setinde hiç NaN yoktu — Faz 1'deki `edf_to_epoch_dataframe.py` script'i hypnogram'da etiketlenmemiş bölgeleri kaynağında zaten dışarıda bırakmış. Kod yine de güvenlik amaçlı bırakıldı; ileride tüm 153 kayda geçilirse NaN çıkabilir.

---

### Adım 3 — N3/N4 Birleştirme (AASM Standardı)

`Sleep_stage` sütunundaki R&K etiketleri AASM'e göre yeniden kodlandı. Hiçbir feature sütununa dokunulmadı — sadece etiket sütununda isim eşlemesi yapıldı.

| R&K etiketi | AASM etiketi | stage_id |
|---|---|---|
| Sleep stage W | W | 0 |
| Sleep stage 1 | N1 | 1 |
| Sleep stage 2 | N2 | 2 |
| Sleep stage 3 | **N3** | 3 |
| Sleep stage 4 | **N3** | 3 |
| Sleep stage R | REM | 4 |

```
Sonuç: 6 sınıf → 5 sınıf
W: 18.612  |  N2: 4.811  |  REM: 1.902  |  N3: 1.267  |  N1: 1.005
```

---

### Adım 4 — Düşük Bilgi Değerli Sütunların Çıkarılması

Faz 2'de açık bırakılan karar (`Temp_rectal_*` ve `Event_marker_*` tut/çıkar) bu adımda korelasyon ölçülerek netleştirildi:

| Sütun grubu | Sleep_stage ile korelasyon | Karar |
|---|---|---|
| `Temp_rectal_mean` | r ≈ **-0.52** | ✅ **Tutuldu** — vücut ısısı uyku derinliğiyle fizyolojik ilişki taşıyor |
| `Temp_rectal_max` | r ≈ **-0.52** | ✅ **Tutuldu** |
| `Event_marker_mean` | r ≈ **-0.01** | ❌ **Çıkarıldı** — uyku evresiyle sistematik ilişki yok, gürültüden farksız |
| `Event_marker_std` | r ≈ **-0.003** | ❌ **Çıkarıldı** |

```
Çıkarılan sütunlar: Event_marker_mean, Event_marker_std, Event_marker_min, Event_marker_max
Kalan öznitelik sayısı: 40 → 32
```

---

### Adım 5 — X / y / groups Ayrımı

```
X shape   : (27.597, 32)   — 32 öznitelik sütunu
y shape   : (27.597,)      — 0-4 arası sınıf id
groups    : 10 benzersiz kişi
```

**32 öznitelik sütunu:**
- EEG_Fpz-Cz: mean, std, min, max, delta, theta, alpha, beta power (8 sütun)
- EEG_Pz-Oz: mean, std, min, max, delta, theta, alpha, beta power (8 sütun)
- EOG_horizontal: mean, std, min, max (4 sütun)
- Resp_oro-nasal: mean, std, min, max (4 sütun)
- EMG_submental: mean, std, min, max (4 sütun)
- Temp_rectal: mean, std, min, max (4 sütun)

---

### Adım 6 — Ölçekleme (StandardScaler)

```
Ölçekleme sonrası: mean ≈ 0.0000, std ≈ 1.0000  ✅
```

---

### Adım 7 — Dokümantasyon CSV'si

```
faz3_processed_epochs.csv → 27.597 satır, 35 sütun
(32 öznitelik + stage_id + stage_name + subject)
Sadece referans/makale amaçlı, eğitimde tekrar kullanılmayacak.
```

---

## Faz 3 Kontrol Listesi

- [x] **N3/N4 birleştirme**: R&K'nin `Sleep stage 3` ve `Sleep stage 4` etiketlerini AASM standardına uygun tek `N3` sınıfında birleştir
  - Sadece etiket sütununda isim eşlemesi yapıldı, hiçbir feature sütununa dokunulmadı. Sınıf sayısı 6'dan 5'e indi.

- [x] Etiketlenmemiş (`NaN`) epoch'ları çıkar
  - Bu veri setinde 0 NaN epoch — Faz 1'de kaynağında temizlenmiş. Kod güvenlik amaçlı bırakıldı.

- [x] Düşük bilgi değerli sütunları değerlendir (`Temp_rectal_*`, `Event_marker_*`) — tut/çıkar kararı, gerekçeyle
  - `Event_marker_*` çıkarıldı — Sleep_stage ile korelasyon ~0.01, bilgi taşımıyor.
  - `Temp_rectal_*` tutuldu — Sleep_stage ile korelasyon ~0.52, vücut ısısı uyku derinliğiyle fizyolojik ilişki taşıyor.

- [x] Özellik ölçekleme (StandardScaler)
  - SVM, KNN, MLP gibi mesafe/gradyan tabanlı algoritmalar için gerekli. Tüm sütunlar mean=0, std=1'e normalize edildi.

- [ ] (Opsiyonel, ileri seviye) Komşu epoch bilgisini ekleme
  - Ertelendi. Faz 5'te baseline modeller kurulduktan sonra, bu özellik eklenmiş/eklenmemiş iki versiyon karşılaştırılıp (ablation testi) sonuca göre kalıcı olarak eklenecek ya da eklenmeyecek.

**Çıktı:** ✅ Model-hazır `X` (27.597×32, ölçeklenmiş), `y` (27.597, sınıf id 0-4), `groups` (27.597, subject isimleri) bellekte hazır.

---

## Faz 5'e Taşınan Açık Notlar

1. Komşu epoch özelliği (önceki epoch'un sinyal değerleri) eklenmedi — Faz 5'te performans karşılaştırması yapılıp karar verilecek.
2. `Temp_rectal_*` tutuldu ama bu kanalın tüm kayıtlarda aynı kalitede olduğu varsayılıyor — 153 kayda geçilince tekrar kontrol edilmeli.