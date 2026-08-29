# Araştırma Yol Haritası

**Proje:** Ham EEG/EOG/EMG Sinyallerinden Otomatik Uyku Evresi Sınıflandırması ve Uyku Verimliliği Tahmini
**Veri seti:** Sleep-EDFx (PhysioNet) — bkz. `01_veri_seti_dokumantasyonu.md`

---

## Araştırma Sorusu

> Ham PSG sinyallerinden (EEG/EOG/EMG) çıkarılan özniteliklerle, uzman tarafından yapılan manuel uyku evresi skorlaması **makine öğrenmesi ile ne derece doğru** tahmin edilebilir? Bu otomatik tahminler, klinik olarak anlamlı bir metrik olan **uyku verimliliğini** ne kadar doğru yansıtır?

İki katmanlı bir tasarım:

```
KATMAN 1 (ML çekirdeği — ana katkı)
   Ham sinyal özellikleri (X)  →  [Model]  →  Tahmini Sleep_stage (ŷ)

KATMAN 2 (klinik doğrulama — makaleye ek değer)
   ŷ'lerden hesaplanan uyku verimliliği  vs  gerçek etiketlerden hesaplanan uyku verimliliği
```

---

## Faz 0 — Kapsam Kararları ✅ (Tamamlandı)

- [x] Veri seti: Sleep-EDFx / Sleep Cassette
- [x] Epoch uzunluğu: 30 sn (AASM/R&K standardı)
- [x] Hedef: `Sleep_stage` (epoch bazlı çok sınıflı sınıflandırma)
- [x] Gece-bazlı ara katman terk edildi, epoch-bazlı tek veri seti (`sleep_edf_epochs.csv`) esas alındı

---

## Faz 1 — Veri Hazırlama ✅ (Tamamlandı)

- [x] `edf_to_epoch_dataframe.py` ile PSG+Hypnogram çiftlerinden epoch-bazlı CSV üretimi
- [x] Zaman domain özellikleri (mean/std/min/max, tüm kanallar)
- [x] Frekans domain özellikleri (delta/theta/alpha/beta bant gücü, sadece EEG kanalları)
- [x] Kanal tipi tespiti isim tabanlı düzeltildi (EDF header güvenilir değil)

**Kalan iş:**
- [ ] Kaç kişi-gece işlenecek karar ver (`MAX_FILES`), tüm SC seti mi yoksa bir alt küme mi
- [ ] CSV'yi üret ve boyutunu doğrula (satır sayısı, dosya boyutu)

---

## Faz 2 — Keşifsel Veri Analizi (EDA) ✅ (Tamamlandı)

- [x] `Sleep_stage` sınıf dağılımını incele (kaç epoch her sınıfta) → sınıf dengesizliği derecesini ölç
  - En büyük/en küçük sınıf oranı ~18.5x (W: %67.4, N3: %4.6) — Faz 4'te class_weight/SMOTE gerekli
- [x] Eksik veri kontrolü (`Sleep_stage` NaN olan epoch var mı — hypnogram'da etiketlenmemiş bölgeler)
  - Eksik veri yok, 27.597 epoch tamamı etiketli
- [x] Kanal bazında özellik dağılımlarını görselleştir (boxplot / histogram, sınıfa göre ayrılmış)
  - Beklenti: `delta_power`, N3'te belirgin şekilde yüksek olmalı; `EMG_submental_std`, REM'de düşük olmalı
  - `delta_power` beklentisi her iki EEG kanalında da doğrulandı; `EMG_submental_std` beklentisi kısmen doğrulandı (REM'i N1/N2'den net ayırmıyor → model karşılaştırmasında izlenmeli)
- [x] Aykırı değer (outlier) taraması — sinyal artefaktlarından kaynaklanan aşırı `min`/`max` değerleri
  - z>5 taraması yapıldı (`outlier_report.csv`); aykırı değerlerin çoğu N3'e ait gerçek fizyolojik sinyal, artefakt değil — hiçbir epoch silinmedi
- [x] Kişiler arası (subject) varyasyon kontrolü — bazı kişilerin sinyal karakteristiği belirgin farklı mı
  - `SC4032E0`'de sistematik olarak yüksek EMG genliği tespit edildi (diğerlerinin ~4.6 katı) — veri setinden çıkarılmadı, Faz 5'te confusion matrix'te ayrıca izlenecek

**Çıktı:** ✅ `Faz_2_result.md` — EDA bulgularını özetleyen bölüm, makalenin "Veri Seti" bölümüne hazır.

---

## Faz 3 — Ön İşleme ve Öznitelik Mühendisliği

- [x] **N3/N4 birleştirme**: R&K'nin `Sleep stage 3` ve `Sleep stage 4` etiketlerini AASM standardına uygun tek `N3` sınıfında birleştir
  - Çünkü AASM standardı evre 3 ve evre 4'ü ikisini de "yavaş dalga uykusu (derin uyku)" olarak kabul ediyor, ayrı tutmanın klinik/bilimsel bir karşılığı yok. `RAW_TO_AASM` sözlüğünde ikisi de `"N3"`'e eşlendi, sınıf sayısı 6'dan 5'e indi.
  - Bununla ilgili makale örnekleri bak. Kanıt gerekebilir.
  
- [x] Etiketlenmemiş (`NaN`) epoch'ları çıkar
  - Çünkü etiketi (hedef değişkeni) bilinmeyen bir örnekle model eğitilemez. Kontrol edildi: bu veri setinde hiç NaN yoktu (Faz 1'de kaynağında zaten temizlenmiş), o yüzden 0 satır çıkarıldı — ama kod güvenlik amaçlı kalıcı, ileride tüm 153 kayda geçilirse NaN çıkabilir.

- [x] Düşük bilgi değerli sütunları değerlendir (`Temp_rectal_*`, `Event_marker_*`) — tut/çıkar kararı, gerekçeyle
  - `Event_marker_*` çıkarıldı, çünkü `Sleep_stage` ile korelasyonu ~0.01 — uyku evresi ne olursa olsun bu değer sistematik değişmiyor, model için gürültüden farksız.
  - `Temp_rectal_*` tutuldu, çünkü `Sleep_stage` ile korelasyonu ~0.52 — vücut ısısı uyku derinliğiyle gerçek bir fizyolojik ilişki taşıyor, bilgi kaybına neden olmadan model bundan faydalanabilir.
  - Bununla ilgili makale bak. Kanıt gerekebilir.

- [x] Özellik ölçekleme (StandardScaler / MinMaxScaler) — mesafe/gradyan tabanlı algoritmalar (SVM, KNN, MLP) için gerekli
  - Çünkü öznitelikler çok farklı ölçeklerde (örn. `delta_power` ~1e-9, `Temp_rectal_mean` ~36-37) — ölçeklenmezse mesafe-tabanlı algoritmalar büyük sayılı sütunlara yanlış şekilde daha fazla önem verir. `StandardScaler` ile her sütun ortalama=0, std=1 olacak şekilde dönüştürüldü.

- [?] (Opsiyonel, ileri seviye) Komşu epoch bilgisini ekleme — örn. önceki epoch'un **ham sinyal özellikleri** (etiketi değil!) yeni bir X sütunu olarak eklenebilir
  - Şimdilik ertelendi, çünkü performansa gerçekten katkısı olup olmadığı ölçülmeden eklemek gereksiz karmaşıklık yaratabilir. Faz 5'te baseline modeller kurulduktan sonra, bu özellik eklenmiş/eklenmemiş iki versiyon karşılaştırılıp (ablation testi) sonuca göre kalıcı olarak eklenecek ya da eklenmeyecek.

**Çıktı:** ✅ Model-hazır `X` (27597×32, ölçeklenmiş), `y` (27597, sınıf id 0-4), `groups` (27597, subject isimleri) matrisleri bellekte hazır.
- `X` oluşturuldu, çünkü modelin öğreneceği tüm sayısal öznitelikleri (32 sütun) tek bir matriste topluyor.
- `y` oluşturuldu, çünkü modelin tahmin etmeyi öğreneceği hedef — her epoch'un AASM'e göre kodlanmış sınıf numarası.
- `groups` oluşturuldu, çünkü Faz 4'te `GroupKFold` ile **kişi bazlı** train/test ayrımı yapabilmek için hangi epoch'un hangi kişiye ait olduğunu bilmemiz gerekiyor — rastgele split yaparsak aynı kişinin epoch'ları hem train hem test'e dağılır, model kişiyi "tanır" ve test skoru yapay olarak şişer (subject leakage).

---

## Faz 4 — Deney Tasarımı (Train/Test Stratejisi)

- [x] **Kişi bazlı ayrım şart**: `GroupKFold` seçildi, grup = `subject`
  - 4 farklı split yöntemi karşılaştırıldı (`test_split.py`):
    - `train_test_split` → leakage var (10/10 kişi), Macro-F1 yapay şişmiş: 0.86
    - `GroupShuffleSplit` → leakage yok ama tek split, Macro-F1: 0.75 — tek bir kişi kombinasyonuna bağlı, güvenilirliği düşük
    - `GroupKFold (5-fold)` → leakage yok, 5 farklı kombinasyonun ortalaması, Macro-F1: 0.74 ± 0.06 — **seçilen yöntem**
    - `train_test_split (subject sütunu çıkarılmış)` → leakage hâlâ var, A ile birebir aynı sonuç — sorun sütun değil bölünme mantığı
    
- [x] Kaç fold kullanılacağına karar ver → **5-fold**
  - 10 kişi / 5 fold = fold başına 2 kişi test'te. Daha fazla fold (örn. 10-fold) fold başına 1 kişi bırakır, tek kişinin skoru tüm fold'u domine eder; 5-fold daha dengeli.

- [x] Sınıf dengesizliğine karşı strateji seç → **`class_weight='balanced'` (birincil)**
  - Hesaplanan ağırlıklar: W:0.297, N1:5.492, N2:1.147, N3:4.356, REM:2.902 — az örnekli sınıflar (N1, N3) daha yüksek ağırlık alıyor.
  - SMOTE alternatifi Faz 5'te ayrı bir karşılaştırma olarak denenecek.

- [x] Değerlendirme metrikleri belirlendi:
  - **Accuracy** — referans için tutuldu, ama sınıf dengesizliği nedeniyle tek başına yanıltıcı (W epoch'ları %67.4, her şeye "W" diyen model bile %67 alır)
  - **Macro-F1** — asıl performans metriği; her sınıfı eşit ağırlıkla değerlendiriyor, N1 gibi az örnekli ama klinik önemli sınıfları da hesaba katıyor
  - **Cohen's Kappa** — tesadüfi doğruluğu düzeltiyor, literatürde insan skorlayıcı uyumuyla (~0.70-0.80) kıyaslanabilir
  - **Confusion Matrix** — hangi evreler birbirine karışıyor; N1↔W ve N1↔N2 en sık karışan çiftler olarak doğrulandı

**Çıktı:** ✅ Sabit, tekrar üretilebilir deney protokolü hazır — `random_state=0`, `GroupKFold(n_splits=5)`, `class_weight='balanced'`, `evaluate_predictions()` fonksiyonu `index.py`'de tanımlı.

---

## Faz 5 — Model Eğitimi ve Karşılaştırma

Aşağıdaki algoritmalar aynı X/y/split ile eğitilip karşılaştırılacak:

| # | Algoritma | Gerekçe |
|---|---|---|
| 1 | Logistic Regression | Basit, yorumlanabilir baseline |
| 2 | K-Nearest Neighbors | Klasik, mesafe tabanlı karşılaştırma |
| 3 | Random Forest | Güçlü, feature importance verir |
| 4 | SVM (RBF kernel) | Literatürde sık kullanılan güçlü yöntem |
| 5 | XGBoost / LightGBM | Genelde en yüksek performanslı yöntemlerden |
| 6 | (opsiyonel) Basit MLP / 1D-CNN | Derin öğrenme karşılaştırması |

- [ ] Her algoritma için hiperparametre araması (GridSearch/RandomSearch, `GroupKFold` içinde)
- [ ] Sonuçları tek bir karşılaştırma tablosunda topla (Accuracy, Macro-F1, Kappa, eğitim süresi)
- [ ] En iyi modelin confusion matrix'ini detaylı incele
- [ ] (Opsiyonel) Random Forest / XGBoost feature importance analizi → "hangi sinyal/bant en ayırt edici" tartışması

**Çıktı:** Model karşılaştırma tablosu + confusion matrix görselleri (makalenin "Bulgular" bölümü)

---

## Faz 6 — Downstream Doğrulama: Uyku Verimliliği

- [ ] En iyi modelin (Faz 5) test setindeki tahminlerini (`ŷ`) kullanarak, her gece için:
  - `TIB` = kayıttaki toplam epoch sayısı × 30 sn (gerekirse baş/son Wake bloğu için tampon uygulanır, bkz. veri dokümantasyonu Bölüm 4.4)
  - `TST_tahmini` = `ŷ != W` olan epoch sayısı × 30 sn
  - `verimlilik_tahmini (%)` = `TST_tahmini / TIB × 100`
- [ ] Aynı hesaplamayı gerçek etiketlerle (`y`) tekrarla → `verimlilik_gercek (%)`
- [ ] Karşılaştırma:
  - Pearson korelasyon katsayısı (tahmini vs gerçek verimlilik)
  - Ortalama mutlak hata (MAE), kök ortalama kare hata (RMSE)
  - **Bland-Altman grafiği** (klinik yöntem karşılaştırmalarında standart görselleştirme — iki ölçüm yönteminin uyumunu gösterir)
- [ ] (Opsiyonel) Verimlilik hatasının hangi sınıf hatalarından (örn. N1 karışıklığı) kaynaklandığını tartış

**Çıktı:** "Otomatik tahmin, klinik verimlilik hesaplamasında ne kadar güvenilir" sorusuna sayısal cevap

---

## Faz 7 — Makale Yazımı

- [ ] **Giriş**: Uyku evrelendirmenin klinik önemi, manuel skorlamanın maliyeti, otomatik yöntemlere ihtiyaç
- [ ] **İlgili Çalışmalar**: Sleep-EDF üzerinde daha önce yapılmış çalışmalarla kıyaslama tablosu (accuracy/kappa bazında)
- [ ] **Veri Seti ve Yöntem**: `01_veri_seti_dokumantasyonu.md` içeriğinin özetlenmiş hali + Faz 1-4 metodolojisi
- [ ] **Bulgular**: Faz 5 model karşılaştırması + Faz 6 verimlilik doğrulaması
- [ ] **Tartışma**: Hangi model neden daha iyi, hangi evreler neden karışıyor (fizyolojik açıklamayla — örn. N1'in kısa süreli ve geçiş evresi olması nedeniyle zor ayırt edilmesi), sınırlamalar
- [ ] **Sonuç ve Gelecek Çalışmalar**: Sınırlamalar (tek veri seti, sağlıklı popülasyon, sadece 2 EEG kanalı), gelecek iş önerileri (daha büyük veri seti, derin öğrenme, gerçek zamanlı uygulama)

---

## Genel Notlar / Açık Kararlar

> Bu bölümü proje ilerledikçe güncelle — henüz netleşmemiş kararlar burada takip edilebilir.

- [ ] SC mi, ST mi, yoksa ikisi birden mi kullanılacak? (İkisi birden kullanılırsa, ilaç etkisi bir konfor değişkeni olarak ele alınabilir — ayrı bir alt analiz konusu olabilir.)
- [ ] Kaç kişi-gece ile çalışılacak (tüm 153 kayıt mı, alt küme mi)?
- [ ] Derin öğrenme (CNN/LSTM) kapsam dahilinde mi, yoksa klasik ML ile mi sınırlı tutulacak?



Claude AI Cht Link : https://claude.ai/chat/8a5a7e91-b15b-4fd8-b7df-175f9c80e236 
Account : hfbkacar1907@gmail.com