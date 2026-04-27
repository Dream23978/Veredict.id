# 🛡️ Veredict.id: Advanced DDoS Detection System

**Veredict.id** adalah sistem keamanan berbasis Machine Learning yang dirancang untuk mendeteksi serangan **DDoS (Distributed Denial of Service)** pada lalu lintas jaringan secara real-time. Proyek ini menggabungkan algoritma **XGBoost** yang kuat dengan antarmuka web interaktif menggunakan **Streamlit**.

---

## 🚀 Fitur Utama
- **Automated Data Cleaning**: Menangani dataset jaringan mentah (dirty data) secara otomatis, termasuk pembersihan nilai infinity, NaN, dan spasi pada nama fitur.
- **High Performance AI**: Menggunakan model XGBoost Classifier yang telah dioptimalkan dengan akurasi **~94.06%**.
- **Interactive Dashboard**: Visualisasi metrik performa model seperti **ROC-AUC** dan **Feature Importance**.
- **Batch Prediction**: Kemampuan untuk mengunggah file CSV lalu lintas jaringan dalam skala besar untuk dianalisis sekaligus.
- **Real-time Metrics**: Menampilkan tingkat kepercayaan (Confidence Score) AI untuk setiap prediksi.

---

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Model**: XGBoost Classifier
- **Data Science**: Scikit-Learn, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Web App**: Streamlit
- **Model Storage**: Joblib

---

## 📊 Hasil Evaluasi Model
Berdasarkan pengujian menggunakan **10-Fold Stratified Cross-Validation**:
- **Accuracy**: 94.06%
- **Precision**: 95.25%
- **Recall**: 92.99%
- **F1-Score**: 93.77%
- **ROC-AUC**: 0.9745

---

## 📂 Struktur Direktori
```text
Veredict.id/
├── Veredict.ipynb             # Notebook untuk training & analisis data
├── app.py                     # Source code aplikasi Streamlit
├── model_ddos_xgboost.joblib  # Model XGBoost yang sudah dilatih
├── scaler_ddos.joblib         # Object scaler (StandardScaler)
├── feature_columns.joblib      # Daftar fitur input model
├── model_metrics.joblib       # Data metrik performa model
└── cleaned_dataset_veredict.csv # Dataset hasil pembersihan (opsional)
```

---

## ⚙️ Cara Instalasi & Penggunaan

### 1. Clone Repository
```bash
git clone https://github.com/username/Veredict.id.git
cd Veredict.id
```

### 2. Install Dependensi
```bash
pip install -r requirements.txt
```
*(Catatan: Jika file requirements.txt belum ada, install manual: `pip install xgboost scikit-learn pandas streamlit matplotlib seaborn joblib`)*

### 3. Training Model (Opsional)
Buka file `Veredict.ipynb` menggunakan Jupyter Notebook atau VS Code, lalu jalankan semua cell untuk melatih model dan mengekspor aset (`.joblib`).

### 4. Jalankan Aplikasi Web
Jalankan perintah berikut di terminal:
```bash
python -m streamlit run app.py
```

---

## 🛡️ Dataset
Proyek ini dikembangkan menggunakan dataset **CIC-IDS2017 (Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv)** yang berisi data lalu lintas jaringan normal (BENIGN) dan serangan DDoS.

---

