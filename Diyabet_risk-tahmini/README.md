
# 🩺 Diyabet Risk Tahmini Projesi

Bu proje, **Pima Indians Diabetes Dataset** üzerinde gerçekleştirilmiş bir **Diyabet Risk Tahmin Sistemi** ve **Streamlit Dashboard** uygulamasını kapsamaktadır. Proje kapsamında veri analizi, modelleme, SHAP yorumlama ve kullanıcı dostu bir web arayüzü geliştirilmiştir.

## 📂 Proje İçeriği

### Ana Python Modelleme Kodu
- 📊 Veri Ön İşleme (eksik veri tamamlama, aykırı değer düzeltme, ölçekleme)
- 🔎 Boxplot ve Korelasyon analizleri
- ⚖️ SMOTE ile dengesiz veri problemi çözümü
- 🤖 3 farklı model eğitimi: Logistic Regression, Random Forest, XGBoost
- 🎨 ROC Curve, Feature Importance görselleri
- 🔍 SHAP ile model yorumlama

### Streamlit Dashboard Özellikleri
- ✅ Kullanıcıdan **Glucose, BMI** gibi bilgiler alınır
- ✅ **Anlık diyabet riski tahmini** yapılır
- ✅ Sağ üstte model değerlendirme skorları ve grafikler gösterilir
- ✅ Dataset genel istatistikleri, korelasyon matrisi ve boxplot grafikleri görüntülenir
- ✅ **XGBoost** modeli üzerinden özellik önem sıralaması görselleştirilir

---

## 🚀 Kurulum ve Çalıştırma

### 1️⃣ Gerekli Kütüphaneleri Kurun:
```bash
pip install -r requirements.txt
```

### 2️⃣ Ana Modelleme Kodunu Çalıştırın
```bash
python main_model.py
```
> `main_model.py` dosyası, verileri analiz eder, modelleri eğitir ve sonuçları konsolda gösterir.

### 3️⃣ Streamlit Dashboard'u Başlatın
```bash
streamlit run dashboard_app.py
```
> `dashboard_app.py` çalıştırıldığında tarayıcıda interaktif bir diyabet tahmin arayüzü açılacaktır.

---

## 📊 Dashboard Ekran Görüntüleri

### Anlık Tahmin
![Anlık Tahmin](screenshots/tahmin.png)

### Genel Veri Analizi
![Veri Analizi](screenshots/analiz.png)

### Özellik Önem Grafiği
![Özellik Önem](screenshots/importance.png)

---

## 📁 Proje Yapısı
```
├── diabetes.csv
├── main_model.py
├── dashboard_app.py
├── requirements.txt
└── README.md
```

---

## 📌 Kullanılan Teknolojiler
- Python
- Scikit-Learn, XGBoost
- Imbalanced-Learn (SMOTE)
- SHAP
- Streamlit
- Matplotlib, Seaborn

---

## 💡 Proje Hedefi
Bu proje, sağlık alanında veri bilimi uygulamalarına örnek teşkil etmekte olup;
- Model performans analizini
- SHAP ile model yorumlamayı
- Web arayüzü üzerinden basit bir diyabet tahmin sistemini
tek bir projede birleştirmeyi amaçlamaktadır.

---

## 📬 İletişim
📧 [Email](bektasguner772@gmail.com)  
