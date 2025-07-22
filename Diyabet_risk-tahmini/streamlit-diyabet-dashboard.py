import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

##### Sayfa Başlığı #####
st.set_page_config(page_title="Diyabet Risk Tahmini Dashboard", layout="wide")

##### Model Yükleme #####
# Eğitimli modelleri pickle ile kaydedip yükleyebilirsin. Şimdilik örnek için yeniden eğitilmiş dummy modeller yükleniyor.
@st.cache_data
def load_models():
    df = pd.read_csv('diabetes.csv')
    from sklearn.preprocessing import MinMaxScaler
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    scaler = MinMaxScaler()
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    return rf, xgb, X, y

rf_model, xgb_model, X, y = load_models()

##### Sidebar - Kullanıcı Girişi #####
st.sidebar.header("Hasta Bilgileri Girişi")
def user_input():
    Pregnancies = st.sidebar.number_input('Pregnancies', 0, 20, 1)
    Glucose = st.sidebar.slider('Glucose', 50, 200, 120)
    BloodPressure = st.sidebar.slider('BloodPressure', 40, 120, 70)
    SkinThickness = st.sidebar.slider('SkinThickness', 10, 100, 20)
    Insulin = st.sidebar.slider('Insulin', 10, 800, 80)
    BMI = st.sidebar.slider('BMI', 15.0, 70.0, 30.0)
    DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', 0.0, 3.0, 0.5)
    Age = st.sidebar.slider('Age', 20, 80, 35)
    data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }
    return pd.DataFrame([data])

user_data = user_input()

##### 🏠 Sayfa 1: Anasayfa #####
st.title("🏥 Diyabet Risk Tahmini Dashboard")
st.write("Bu dashboard üzerinde diyabet riski hesaplama, veri keşfi ve model analizleri yapılmaktadır.")

##### 📈 Sayfa 2: Anlık Tahmin #####
st.header("📈 Anlık Diyabet Riski Tahmini")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Girilen Hasta Verisi")
    st.write(user_data)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
user_scaled = scaler.transform(user_data)

with col2:
    st.subheader("🎯 Tahmin Sonucu")
    prediction = xgb_model.predict(user_scaled)
    probability = xgb_model.predict_proba(user_scaled)[0][1]
    if prediction[0]==1:
        st.error(f"Yüksek Diyabet Riski! Tahmin olasılığı: {probability:.2f}")
    else:
        st.success(f"Düşük Diyabet Riski. Tahmin olasılığı: {probability:.2f}")

##### 📊 Sayfa 3: Dataset Genel Analizi #####
st.header("📊 Dataset Genel İstatistik ve Görseller")

st.subheader("Genel İstatistik")
st.write(X.describe())

st.subheader("Korelasyon Matrisi")
corr = X.corr()
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.subheader("Boxplotlar")
selected_col = st.selectbox("Boxplot için değişken seçin:", X.columns)
fig2, ax2 = plt.subplots()
sns.boxplot(x=X[selected_col], ax=ax2)
st.pyplot(fig2)

##### 📈 Sayfa 4: Model Performansı #####
st.header("📈 Model Performansı")

st.subheader("Özellik Önem Skoru - XGBoost")
importance = xgb_model.feature_importances_
fig3, ax3 = plt.subplots()
sns.barplot(x=importance, y=X.columns, ax=ax3)
st.pyplot(fig3)
