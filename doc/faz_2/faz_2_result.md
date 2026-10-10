# Faz 2 — Keşifsel Veri Analizi (EDA - Exploratory Data Analysis) — Sonuç Raporu

**Veri seti:** `sleep_edf_epochs.csv`
**Toplam epoch:** 27.597 | **Kişi sayısı:** 10 (subject_root: SC4001E0 … SC4042E0)
**Kullanılan script:** `faz2_eda.py`

---

## Kavramlar (raporu okurken lazım olacak sözlük)

- **EDA (Exploratory Data Analysis / Keşifsel Veri Analizi):** Modele hiç dokunmadan önce veriyi tanıma aşaması — eksik veri var mı, sınıflar dengeli mi, öznitelikler gerçekten ayırt edici mi gibi soruların cevaplandığı adım.
- **Boxplot (kutu grafiği):** Bir özniteliğin (örn. `delta_power`) her sınıf için değer dağılımını gösteren grafik. Y ekseni (dikey eksen) o özniteliğin **değeri** — ham EEG sinyali değil, sinyalden hesaplanmış bir sayı (örn. delta bandı güç yoğunluğu). Kutu, değerlerin %25-%75'lik aralığını, ortadaki çizgi medyanı gösterir. Kutular sınıflar arasında ne kadar ayrışıksa, o öznitelik sınıfları o kadar iyi ayırt ediyor demektir.
- **z-score / z>5 taraması:** Bir değerin ortalamadan kaç standart sapma uzakta olduğunu ölçer; |z|>5 "istatistiksel olarak çok nadir" demektir, otomatik olarak "hatalı" anlamına gelmez.
- **Macro-F1 / Cohen's Kappa:** Faz 5'te modelin başarısını ölçmek için kullanılacak metrikler (bu raporun konusu değil, ileride kullanılacak).

---

## 1. `Sleep_stage` Sınıf Dağılımı ve Dengesizlik

| Sınıf (AASM, 3+4→N3 birleşik) | Epoch | Oran |
|---|---|---|
| W | 18.612 | %67.4 |
| N2 | 4.811 | %17.4 |
| REM | 1.902 | %6.9 |
| N3 | 1.267 | %4.6 |
| N1 | 1.005 | %3.6 |

**En büyük/en küçük sınıf oranı: ~18.5x** — ciddi bir sınıf dengesizliği. Faz 4/5'te:
- Ham `accuracy` metriği tek başına yanıltıcı olur (her şeye "W" diyen bir model bile ~%67 accuracy alır).
- `class_weight='balanced'` veya SMOTE değerlendirilmeli.
- Değerlendirme **Macro-F1** ve **Cohen's Kappa** ile yapılmalı.

---

## 2. Eksik Veri Kontrolü

- Hiçbir sütunda `NaN` bulunmadı.
- `Sleep_stage` alanı tamamen dolu — hypnogram'da etiketlenmemiş (skorlanmamış) bölge yok.

**Sonuç: Faz 3'te ayrıca "NaN epoch çıkarma" adımına gerek yok**, ama script pipeline'da yine de bir güvenlik kontrolü olarak bırakılmalı (gelecekte farklı kayıtlar eklenirse).

---

## 3. Kanal Bazında Özellik Dağılımları (Sınıfa Göre Boxplot)

İncelenen öznitelik/kanal çiftleri ve doğrulanan/doğrulanmayan hipotezler:

| Öznitelik | Hipotez | Sonuç |
|---|---|---|
| `EEG_Fpz-Cz_delta_power` | N3'te yüksek | ✅ Doğrulandı — net ayrışma |
| `EEG_Pz-Oz_delta_power` | N3'te yüksek | ✅ Doğrulandı — aynı kalıp, farklı kanal |
| `EMG_submental_std` | REM'de düşük (atoni) | ⚠️ Kısmen — W'de en düşük, ama REM ile N1/N2 arasında belirgin ayrışma yok → REM↔N2 karışması modelde beklenmeli |
| `EOG_horizontal_std` | REM'de yüksek (hızlı göz hareketi) | İncelendi, grafik ekli |
| `EEG_Fpz-Cz_beta_power` | W'de yüksek | İncelendi, grafik ekli |
| `Resp_oro-nasal_std` | REM'de düzensizlik artabilir | İncelendi, grafik ekli |

**Ek bulgu:** `EEG_Pz-Oz_delta_power` ile `EEG_Fpz-Cz_delta_power` arasındaki korelasyon sadece **r ≈ 0.18** — iki kanal, delta gücü açısından birbirini tekrar etmiyor, yani modelde **her iki kanalı da tutmak** bilgi kaybını önler (biri diğerinin yerine geçmiyor).

Görselleştirme `faz2_eda.py` çalıştırıldığında yeniden üretilebilir; ayrı dosya olarak saklanmadı, yukarıdaki bulgular görsel incelemenin metne dökülmüş halidir.

---

## 4. Aykırı Değer (Outlier) Taraması

Yöntem: her sayısal sütun için z-score hesaplanıp `|z| > 5` olan epoch'lar işaretlendi (silinmedi). Tam liste: `outlier_report.csv`.

En çok aykırı değer taşıyan sütunlar: `EEG_Pz-Oz_delta_power` (%0.74), `Temp_rectal_std` (%0.70), `EMG_submental_std` (%0.67), `Resp_oro-nasal_mean` (%0.57).

**Doğrulama incelemesi (`EEG_Pz-Oz_delta_power`):**

| Sınıf | Genel veri setindeki oran | Aykırı değerler içindeki oran |
|---|---|---|
| N3 | %4.6 | **%66.8** |
| W | %67.4 | %19.0 |
| N2 | %17.4 | %12.8 |

Aykırı değerlerin büyük çoğunluğu N3 sınıfında yoğunlaşıyor → bunlar **artefakt değil, gerçek derin uyku sinyali** (delta gücü fizyolojik olarak zaten çok yüksek). `SC4031E0` özelinde de aynı doğrulama yapıldı: 83 aykırı epoch'un 72'si N2/N3'te.

**Karar:** Bu aşamada hiçbir epoch veri setinden çıkarılmadı. Faz 3'te mesafe-tabanlı modeller (SVM/KNN) için gerekirse **winsorizing** (üst/alt persantile kırpma) uygulanabilir; ağaç-tabanlı modeller (Random Forest/XGBoost) bu aykırı değerlere zaten dayanıklıdır.

---

## 5. Kişiler Arası (Subject) Varyasyon

Kişi başına W oranı %60–75 arasında değişiyor (beklenen doğal varyasyon; görsel `faz2_eda.py` çalıştırıldığında yeniden üretilebilir, ayrı dosya olarak saklanmadı).

**Dikkat çeken bulgu — `SC4032E0`:**

| Subject | EMG_submental_std ortalaması |
|---|---|
| **SC4032E0** | **7.29e-7** |
| Diğer 9 kişinin ortalaması | 1.59e-7 |

`SC4032E0`'nin EMG sinyali, diğer kişilerin **~4.6 katı** — tek bir aykırı epoch değil, kaydın tamamında sistematik olarak yüksek. Olası nedenler: elektrot yerleşimi/empedans farkı, kayıt gecesi kas geriliminin gerçekten yüksek olması.

**Öneri:** Bu kişi veri setinden çıkarılmamalı (hâlâ geçerli fizyolojik veri), ancak:
- Faz 3'te ölçekleme **kişi bazlı değil global** yapılmalı ki model bu farkı da öğrensin.
- Faz 5'te confusion matrix incelenirken `SC4032E0`'nin hata oranı ayrıca kontrol edilmeli (bu kişide sistematik yanlış sınıflandırma varsa EMG tabanlı ayrımın bu kişide zayıfladığını gösterir).

---

## 6. Düşük Bilgi Değerli Sütun Adayları (`Temp_rectal_*`, `Event_marker_*`)

Her iki grup da **sabit/sıfır-varyans değil** — gerçek varyasyon taşıyorlar (`Temp_rectal_mean` std=0.42, `Event_marker_mean` std=150.5). Bu nedenle roadmap'teki "düşük bilgi değerli, çıkar" varsayımı bu haliyle doğrulanmadı.

**Faz 3 için öneri:** Doğrudan çıkarmak yerine, bu sütunların `Sleep_stage` ile korelasyonuna/feature importance'ına bakılıp karar verilmeli. Özellikle `Temp_rectal` sirkadiyen ritimle ilişkili olabileceğinden potansiyel sinyal taşıyor olabilir.

---

## Genel Özet (Makale "Veri Seti" Bölümü İçin Taslak)

> Sleep-EDFx Sleep Cassette alt kümesinden 10 kişi-gece, 30 saniyelik epoch bazında toplam 27.597 örnek içeren bir veri seti oluşturulmuştur. Veri setinde eksik değer bulunmamaktadır. `Sleep_stage` etiketleri belirgin bir sınıf dengesizliği göstermektedir (W: %67.4, N3: %4.6), bu durum model eğitiminde sınıf ağırlıklandırma stratejilerini gerekli kılmaktadır. Keşifsel analiz, delta bant gücünün N3 evresinde beklenen şekilde yükseldiğini (her iki EEG kanalında da), buna karşın EMG genliğinin REM evresini diğer uyku evrelerinden net biçimde ayırmadığını göstermiştir; bu durum, sınıflandırma modellerinde REM–N2 karışıklığına işaret edebilir. Z-skor tabanlı aykırı değer taraması, tespit edilen uç değerlerin büyük ölçüde N3 evresine ait gerçek fizyolojik sinyaller olduğunu, artefakt olmadığını göstermiştir. Kişiler arası karşılaştırmada bir kişide (SC4032E0) sistematik olarak yüksek EMG genliği tespit edilmiş, bu durum kayıttan çıkarma nedeni olarak değil, sonraki analiz aşamalarında izlenmesi gereken bir gözlem olarak not edilmiştir.

---

## Sade Dille — Faz 2'de Ne Öğrendik?

1. **Veri temiz.** Eksik değer yok, direkt kullanılabilir.
2. **Sınıflar çok dengesiz.** Uyanıklık (W) epoch'ları N3'ün ~18.5 katı — model eğitiminde bunu telafi eden bir strateji (class_weight, Macro-F1 gibi) şart.
3. **Öznitelikler işe yarıyor ama eşit değil.** `delta_power` N3'ü çok iyi ayırt ediyor; `EMG_submental_std` ise REM'i diğer evrelerden net ayıramıyor → modelin REM-N2 arasında zorlanacağını şimdiden tahmin edebiliyoruz.
4. **"Aykırı" görünen değerler çoğunlukla gerçek.** N3'teki yüksek delta_power değerleri hata değil, gerçek derin uyku sinyali — silinmedi.
5. **Bir kişi (SC4032E0) farklı sinyal karakteristiğine sahip.** Modelin bu kişide daha çok hata yapma ihtimaline karşı ileride ayrıca kontrol edilecek.

Kısacası: **veri sağlam ama dengesiz, ve bazı sinyaller bazı uyku evrelerini diğerlerinden daha iyi ayırt ediyor** — bu bilgiyle Faz 3'e (ön işleme) geçiliyor.

---

## Faz 2 Kontrol Listesi (Güncellenmiş Roadmap)

- [x] `Sleep_stage` sınıf dağılımını incele (kaç epoch her sınıfta) → sınıf dengesizliği derecesini ölç
- [x] Eksik veri kontrolü (`Sleep_stage` NaN olan epoch var mı — hypnogram'da etiketlenmemiş bölgeler)
- [x] Kanal bazında özellik dağılımlarını görselleştir (boxplot / histogram, sınıfa göre ayrılmış)
  - Beklenti: `delta_power`, N3'te belirgin şekilde yüksek olmalı; `EMG_submental_std`, REM'de düşük olmalı
- [x] Aykırı değer (outlier) taraması — sinyal artefaktlarından kaynaklanan aşırı `min`/`max` değerleri
- [x] Kişiler arası (subject) varyasyon kontrolü — bazı kişilerin sinyal karakteristiği belirgin farklı mı

**Çıktı:** ✅ Tamamlandı — bu belge (`Faz_2_result.md`), makalenin "Veri Seti" bölümüne temel oluşturacak EDA özetidir.

---

## Faz 3'e Taşınan Açık Notlar

1. `SC4032E0`'nin yüksek EMG genliği → Faz 5 confusion matrix analizinde bu kişiye özel hata oranı kontrol edilmeli.
2. `Temp_rectal_*` / `Event_marker_*` sütunları için tut/çıkar kararı, korelasyon/feature-importance analizine ertelendi (körlemesine çıkarılmayacak).
3. EMG_submental_std'nin REM'i net ayırmaması → model karşılaştırmasında REM sınıfının recall/precision değerleri ayrıca izlenmeli.