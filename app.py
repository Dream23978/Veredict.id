import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Veredict.id Pro", page_icon="🛡️", layout="wide")

# --- Load Assets ---
@st.cache_resource
def load_assets():
    model = joblib.load('model_ddos_xgboost.joblib')
    scaler = joblib.load('scaler_ddos.joblib')
    features = joblib.load('feature_columns.joblib')
    try:
        metrics = joblib.load('model_metrics.joblib')
    except:
        metrics = None
    return model, scaler, features, metrics

model, scaler, feature_columns, model_metrics = load_assets()

# --- Fungsi Preprocessing Otomatis (Handle Data Kotor) ---
def clean_and_prepare(df):
    # 1. Bersihkan spasi di nama kolom
    df.columns = df.columns.str.strip()
    
    # 2. Handle Infinity dan NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0) # Mengisi data kosong dengan 0
    
    # 3. Pilih hanya fitur yang dibutuhkan model
    # Jika ada fitur yang kurang, beri nilai 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
            
    return df[feature_columns]

# --- UI Styling ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("🛡️ Veredict.id Pro")
menu = st.sidebar.radio("Navigasi", ["Dashboard & Metrics", "Deteksi Data Kotor"])

if menu == "Dashboard & Metrics":
    st.title("📊 Model Insights & Performance")
    
    if model_metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("K-Fold Accuracy", f"{model_metrics['kfold_acc']}%")
        m2.metric("ROC-AUC Score", f"{model_metrics['roc_auc']}")
        m3.metric("Precision", f"{model_metrics['kfold_prec']}%")
        m4.metric("F1-Score", f"{model_metrics['kfold_f1']}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🔝 Top 10 Feature Importance")
        # Ambil importance dari model XGBoost
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=feature_columns).sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots()
        sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis", ax=ax)
        plt.title("Fitur Paling Berpengaruh")
        st.pyplot(fig)

    with col2:
        st.write("### 📈 Penjelasan Model")
        st.info("""
            **K-Fold Cross Validation (10-Folds)** memastikan model stabil dan tidak overfitting. 
            Akurasi ~94% menunjukkan kemampuan deteksi yang sangat andal pada trafik jaringan.
            
            **ROC-AUC** mendekati 1.0 (0.97) berarti model sangat baik dalam membedakan 
            antara trafik normal dan serangan di jaringan.
        """)

    st.markdown("---")

    # --- Deskripsi Algoritma XGBoost ---
    st.write("### 🤖 Tentang Algoritma XGBoost")

    tab1, tab2, tab3 = st.tabs(["📌 Apa itu XGBoost?", "⚙️ Cara Kerja", "🎯 Mengapa XGBoost?"])

    with tab1:
        st.markdown("""
        **XGBoost** *(Extreme Gradient Boosting)* adalah algoritma machine learning berbasis 
        **ensemble learning** yang dikembangkan oleh Tianqi Chen pada tahun 2014. 
        XGBoost membangun banyak **pohon keputusan (decision tree)** secara berurutan, 
        di mana setiap pohon baru belajar dari **kesalahan pohon sebelumnya**.

        > *"XGBoost bukan sekadar model — ia adalah sistem optimasi gradient boosting yang dirancang 
        untuk efisiensi, fleksibilitas, dan portabilitas tinggi."*

        **Karakteristik Utama:**
        - ✅ Berbasis **Gradient Boosted Decision Trees (GBDT)**
        - ✅ Menggunakan teknik **regularisasi L1 & L2** untuk mencegah overfitting
        - ✅ Mendukung **parallel computing** sehingga sangat cepat
        - ✅ Terbukti unggul di banyak kompetisi data science (Kaggle, dll.)
        """)

    with tab2:
        st.markdown("""
        #### Alur Kerja XGBoost:

        1. **Inisialisasi** — Model dimulai dengan prediksi awal (biasanya rata-rata target).
        2. **Hitung Residual** — Hitung selisih antara nilai aktual dan prediksi (*residual/error*).
        3. **Bangun Pohon Keputusan** — Pohon baru dilatih untuk memprediksi residual tersebut.
        4. **Update Prediksi** — Prediksi diperbarui dengan menambahkan output pohon baru dikalikan *learning rate (η)*.
        5. **Ulangi** — Proses diulang sebanyak `n_estimators` kali hingga error diminimalkan.

        **Formula Prediksi Final:**
        """)
        st.latex(r"\hat{y} = \sum_{k=1}^{K} \eta \cdot f_k(x)")
        st.markdown(r"""
        Di mana:
        - $\hat{y}$ = prediksi akhir
        - $K$ = jumlah total pohon
        - $\eta$ = learning rate
        - $f_k(x)$ = output pohon ke-$k$

        **Objective Function dengan Regularisasi:**
        """)
        st.latex(r"\mathcal{L} = \sum_{i} l(y_i, \hat{y}_i) + \sum_{k} \Omega(f_k)")
        st.markdown(r"""
        - $l(y_i, \hat{y}_i)$ = loss function (misal: log loss untuk klasifikasi)
        - $\Omega(f_k) = \gamma T + \frac{1}{2}\lambda \|w\|^2$ = regularisasi untuk mengontrol kompleksitas pohon
        """)

    with tab3:
        st.markdown("""
        #### Mengapa XGBoost untuk Deteksi DDoS?

        | Keunggulan | Penjelasan |
        |---|---|
        | 🚀 **Kecepatan Tinggi** | Komputasi paralel membuat training jauh lebih cepat dari algoritma boosting lainnya |
        | 🎯 **Akurasi Tinggi** | Ensemble dari banyak pohon menghasilkan prediksi yang sangat akurat |
        | 🔒 **Tahan Overfitting** | Regularisasi L1/L2 + early stopping mencegah model terlalu hafal data training |
        | 📊 **Feature Importance** | Memberikan transparansi fitur mana yang paling berpengaruh dalam deteksi |
        | 🧹 **Toleran Data Kotor** | Dapat menangani missing values secara internal |
        | 🌐 **Terbukti di Industri** | Dipakai luas di bidang keamanan siber, finansial, dan riset akademis |

        Pada sistem **Veredict.id**, XGBoost dilatih menggunakan dataset **CIC-IDS2017** 
        dengan validasi **10-Fold Cross Validation** untuk memastikan performa yang stabil 
        dan dapat diandalkan dalam mendeteksi serangan DDoS secara real-time.
        """)

elif menu == "Deteksi Data Kotor":
    st.title("📂 Deteksi Batch (Support Data Kotor)")
    st.markdown("Unggah file CSV mentah hasil dump jaringan. Sistem akan membersihkan data secara otomatis.")
    
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        raw_df = pd.read_csv(file)
        st.write("Data Mentah (5 baris pertama):")
        st.dataframe(raw_df.head())
        
        if st.button("Jalankan Pembersihan & Prediksi"):
            with st.spinner("Processing Data..."):
                # Preprocessing
                processed_df = clean_and_prepare(raw_df)
                
                # Scaling
                scaled_data = scaler.transform(processed_df)
                
                # Predict
                preds = model.predict(scaled_data)
                probs = model.predict_proba(scaled_data)[:, 1]
                
                # Hasil
                raw_df['Hasil_Analisis'] = ['DDoS' if p == 1 else 'BENIGN' for p in preds]
                raw_df['Confidence'] = probs
                
                st.success("Analisis Selesai!")
                
                c1, c2 = st.columns(2)
                c1.write("### Hasil Prediksi")
                c1.dataframe(raw_df[['Hasil_Analisis', 'Confidence']].head(100))
                
                # Visualisasi Distribusi
                with c2:
                    st.write("### Distribusi Trafik")
                    fig2, ax2 = plt.subplots()
                    raw_df['Hasil_Analisis'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], ax=ax2)
                    st.pyplot(fig2)
