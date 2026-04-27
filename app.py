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
menu = st.sidebar.radio("Navigasi", ["Dashboard & Metrics", "Deteksi Data Kotor", "Manual Prediction"])

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

elif menu == "Manual Prediction":
    st.title("⌨️ Manual Input Analysis")
    # Form input manual yang sudah ada sebelumnya
    with st.form("manual_form"):
        st.write("Masukkan nilai fitur secara manual:")
        cols = st.columns(3)
        input_dict = {}
        for i, col in enumerate(feature_columns):
            with cols[i % 3]:
                input_dict[col] = st.number_input(col, value=0.0)
        
        if st.form_submit_button("Cek Trafik"):
            input_df = pd.DataFrame([input_dict])
            scaled_input = scaler.transform(input_df)
            res = model.predict(scaled_input)[0]
            if res == 1:
                st.error("⚠️ HASIL: SERANGAN DDOS TERDETEKSI!")
            else:
                st.success("✅ HASIL: TRAFIK AMAN (BENIGN)")
