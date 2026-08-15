# Veri Seti Dokümantasyonu

**Proje:** Ham EEG/EOG/EMG Sinyallerinden Otomatik Uyku Evresi Sınıflandırması ve Uyku Verimliliği Tahmini
**Kaynak veri seti:** [Sleep-EDF Database Expanded (Sleep-EDFx) v1.0.0](https://www.physionet.org/content/sleep-edfx/1.0.0/) — PhysioNet
**Son güncelleme:** _(doldurulacak)_

---

## 1. Ham Veri Seti Hakkında

Sleep-EDFx, iki alt çalışmadan oluşan, polisomnografi (PSG) kayıtları içeren açık bir veri setidir:

| Alt küme | Kişi sayısı | Amaç |
|---|---|---|
| **Sleep Cassette (SC)** | 78 kişi (sağlıklı, çoğunluğu 2 gece) | Yaşa bağlı uyku değişimlerini incelemek |
| **Sleep Telemetry (ST)** | 22 kişi (2 gece: placebo / temazepam) | Bir uyku ilacının uyku yapısına etkisini incelemek |

Bu çalışmada **Sleep Cassette (SC)** alt kümesi kullanılmaktadır. *(ST kullanılıyorsa bu satırı güncelle.)*

### 1.1 Dosya Adlandırma Kuralı (SC)

```
SC 4 ss N Ec
│  │ │  │ │└─ sabit harf/kod
│  │ │  │ └── kayıt türü kodu (E = ölçüm)
│  │ │  └──── gece numarası (1 = ilk gece, 2 = ikinci gece)
│  │ └─────── kişi numarası (00-82 arası, 78 kişi)
│  └───────── çalışma kodu
└──────────── "Sleep Cassette" sabiti
```

Örnek: `SC4001E0-PSG.edf` → kişi **00**, gece **1**.
`SC4002E0-PSG.edf` → **aynı kişi**, gece **2**.

> ⚠️ **Kritik nokta:** Aynı kişinin farklı geceleri farklı dosyalarda olsa da, **aynı kişiye ait olduğu için model eğitim/test ayrımında (train/test split) aynı grupta tutulmalıdır.** Aksi halde veri sızıntısı (data leakage) oluşur ve sonuçlar yapay olarak yüksek çıkar. Bkz. `subject_root` sütunu (Bölüm 3).

### 1.2 Her Gece İçin İki Dosya

| Dosya | İçerik |
|---|---|
| `SC4ssNE0-PSG.edf` | Ham sinyal (7 kanal, 100 Hz örnekleme) |
| `SC4ssNEC-Hypnogram.edf` | Uzman (PSG teknisyeni) tarafından verilmiş uyku evresi etiketleri, 30 sn'lik bloklar halinde (annotation) |

### 1.3 Ham Kayıttaki Kanallar

| Kanal | Ölçtüğü şey | Uyku analizindeki rolü |
|---|---|---|
| EEG Fpz-Cz | Beyin elektriksel aktivitesi | Ana evre belirleyici (frekans bandı içeriği) |
| EEG Pz-Oz | Beyin elektriksel aktivitesi (oksipital) | İkincil EEG kanalı |
| EOG horizontal | Göz hareketleri | REM tespiti için kritik |
| Resp oro-nasal | Solunum akışı | Dolaylı evre/apne bilgisi |
| EMG submental | Çene altı kas aktivitesi | REM'de kas atonisi → REM ayrımı |
| Temp rectal | Vücut iç sıcaklığı | Sirkadiyen bilgi, zayıf ayırt edicilik |
| Event marker | Manuel işaretleme | Genelde düşük/sıfır bilgi değeri |

---

## 2. Ön İşleme Süreci (Pipeline)

```
Ham EDF dosyaları (PSG + Hypnogram)
        │
        │  edf_to_epoch_dataframe.py
        │  - 30 sn'lik epoch'lara bölme (AASM standardı)
        │  - Zaman domain özellik çıkarma (mean/std/min/max)
        │  - Frekans domain özellik çıkarma (Welch PSD → bant güçleri)
        │  - Hypnogram etiketlerini epoch'lara eşleme
        ▼
sleep_edf_epochs.csv   (EPOCH BAZLI, tek ve ana veri seti)
```

### 2.1 Neden 30 Saniyelik Epoch?

R&K (1968) ve AASM (2007) uyku evrelendirme standartlarının ikisi de 30 sn'lik epoch kullanır. Sleep-EDFx hypnogram etiketleri de bu uzunlukta bloklar halinde verilmiştir; farklı bir epoch uzunluğu seçmek ground-truth etiketlerle uyuşmazlık yaratır.

### 2.2 Frekans Bandı Tanımları (sadece EEG kanalları için hesaplanır)

| Bant | Frekans aralığı | Klinik ilişki |
|---|---|---|
| Delta | 0.5 – 4 Hz | Derin uyku (N3) ile ilişkili |
| Theta | 4 – 8 Hz | N1 (uykuya geçiş) karakteristiği |
| Alpha | 8 – 12 Hz | Uyanıklık / gözler kapalı rahatlama |
| Beta | 12 – 30 Hz | Uyanık, aktif zihinsel durum |

Hesaplama yöntemi: Welch Power Spectral Density (PSD), her epoch için ayrı ayrı, band içindeki alan (trapezoidal integrasyon) alınarak.

---

## 3. `sleep_edf_epochs.csv` — Sütun Sözlüğü

**Satır birimi:** 1 satır = 1 kişinin 1 gecesinden 1 epoch (30 sn)
**Beklenen boyut:** ~N kişi-gece × (~800-1200 epoch/gece) satır

### 3.1 Kimlik / Zaman Sütunları

| Sütun | Tip | Açıklama |
|---|---|---|
| `subject` | string | Kişi-gece kimliği (örn. `SC4001E0`) |
| `subject_root` | string | Kişi grubu kimliği, ilk 5 karakter (örn. `SC400`) — **train/test split bunun üzerinden yapılmalı** |
| `epoch_idx` | int | Kaydın kaçıncı epoch'u (0'dan başlar) |
| `epoch_start_sec` | int | Epoch'un kayıt başından itibaren başlangıç saniyesi |

### 3.2 Sinyal Özellik Sütunları

Her fiziksel kanal (Bölüm 1.3) için:

| Sonek | Anlamı | Hangi kanallarda var |
|---|---|---|
| `_mean` | Epoch içi ortalama genlik | Tüm kanallar |
| `_std` | Epoch içi standart sapma | Tüm kanallar |
| `_min` / `_max` | Epoch içi en düşük/yüksek değer | Tüm kanallar |
| `_delta_power`, `_theta_power`, `_alpha_power`, `_beta_power` | Frekans bandı gücü (Welch PSD) | **Sadece EEG kanalları** (Fpz-Cz, Pz-Oz) |

### 3.3 Hedef Değişken

| Sütun | Tip | Değer kümesi | Açıklama |
|---|---|---|---|
| `Sleep_stage` | kategorik | `Sleep stage W`, `1`, `2`, `3`, `4`, `R` | Uzman tarafından verilmiş ground-truth etiket. **ML modelinin tahmin etmeye çalışacağı değer (y)** |

> ⚠️ **Veri sızıntısı uyarısı:** `Sleep_stage` sütunu **yalnızca hedef (y) olarak** kullanılmalıdır. Bu sütundan türetilen hiçbir bilgi (örn. "önceki epoch'un evresi") girdi özelliği (X) olarak modele verilmemelidir; aksi halde model gerçek sinyalden değil etiketin kendisinden öğrenmiş olur.

---

## 4. Bilinen Sınırlamalar / Dikkat Edilmesi Gerekenler

1. **Sınıf dengesizliği**: Gerçek verilerde `Sleep stage W` ve `Sleep stage 2` diğer sınıflara göre çok daha fazla örneğe sahiptir; `Sleep stage 1` ve `Sleep stage 4` genelde az örneklidir. Model değerlendirmesinde salt accuracy yerine **macro-F1 / Cohen's Kappa** kullanılmalıdır (Bkz. Yol Haritası, Faz 5).
2. **N3/N4 birleştirme**: AASM (2007) sonrası standart, R&K'nin ayrı tuttuğu N3 ve N4 evrelerini tek bir "N3" evresinde birleştirir. Sleep-EDFx orijinal R&K etiketleme kullandığı için bu birleştirme ön işleme aşamasında yapılmalıdır (bkz. Faz 3).
3. **Düşük bilgi değerli kanallar**: `Temp_rectal` ve `Event_marker` sütunlarının sınıflandırmaya katkısı düşüktür; feature selection aşamasında değerlendirilmelidir.
4. **Kayıt başı/sonu Wake bloğu**: Ham kayıtlarda kişi yatmadan önce ve uyandıktan sonra da bir miktar `Sleep stage W` verisi bulunur (kayıt cihazının açık kalma süresi). Bu, evre sınıflandırması için sorun değildir (zaten W etiketi doğru), yalnızca **verimlilik hesaplaması** (Faz 6) yapılırken dikkate alınmalıdır.

---

## 5. Versiyon Geçmişi / Notlar

- ~~Gece-bazlı özet veri seti (`sleep_efficiency_night_level.csv`)~~ — İlk tasarımda ayrı bir dosya olarak planlanmıştı, ancak araştırma kapsamı **epoch-bazlı sınıflandırmaya** odaklandığı için bu ara katman kaldırıldı. Uyku verimliliği artık ayrı bir veri seti değil, model çıktılarının **downstream (sonraki aşama) değerlendirmesi** olarak hesaplanıyor (bkz. Yol Haritası Faz 6).
