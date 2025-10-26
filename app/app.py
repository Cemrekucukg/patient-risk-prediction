import streamlit as st
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# --- Model yükleme ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "diabetes_xgb.pkl")
model = joblib.load(os.path.abspath(MODEL_PATH))

st.set_page_config(page_title="Patient Risk Prediction", page_icon="🩺")
st.title("🩺 Patient Risk Prediction System")
st.write("Bu uygulama sağlık göstergelerine göre diyabet riskini tahmin eder.")

# --- Sidebar inputlar ---
st.sidebar.header("🧍‍♀️ Kullanıcı Bilgileri")

bmi = st.sidebar.number_input("BMI (Vücut Kitle İndeksi)", min_value=10.0, max_value=60.0, value=25.0)
age = st.sidebar.slider("Yaş", 18, 100, 30)
high_bp = st.sidebar.selectbox("Yüksek Tansiyon", ["Hayır", "Evet"])
high_chol = st.sidebar.selectbox("Yüksek Kolesterol", ["Hayır", "Evet"])
chol_check = st.sidebar.selectbox("Son 5 yılda kolesterol kontrolü yaptırdınız mı?", ["Evet", "Hayır"])
smoker = st.sidebar.selectbox("Sigara Kullanımı", ["Evet", "Hayır"])
stroke = st.sidebar.selectbox("Felç (Stroke) Geçirdiniz mi?", ["Hayır", "Evet"])
heart_disease = st.sidebar.selectbox("Kalp Hastalığı / Kalp Krizi", ["Hayır", "Evet"])
phys_activity = st.sidebar.selectbox("Düzenli Fiziksel Aktivite", ["Evet", "Hayır"])
fruits = st.sidebar.selectbox("Günlük meyve tüketiyor musunuz?", ["Evet", "Hayır"])
veggies = st.sidebar.selectbox("Günlük sebze tüketiyor musunuz?", ["Evet", "Hayır"])
alcohol = st.sidebar.selectbox("Ağır alkol kullanımı var mı?", ["Hayır", "Evet"])
healthcare = st.sidebar.selectbox("Sağlık hizmetine erişiminiz var mı?", ["Evet", "Hayır"])
no_doc_cost = st.sidebar.selectbox("Maddi yetersizlik nedeniyle doktora gitmediniz mi?", ["Hayır", "Evet"])
gen_health = st.sidebar.slider("Genel Sağlık (1=Çok İyi, 5=Çok Kötü)", 1, 5, 3)
ment_health = st.sidebar.number_input("Kötü Ruhsal Sağlık Günleri (0–30)", 0, 30, 5)
phys_health = st.sidebar.number_input("Kötü Fiziksel Sağlık Günleri (0–30)", 0, 30, 5)
diff_walk = st.sidebar.selectbox("Yürürken Zorluk", ["Hayır", "Evet"])
sex = st.sidebar.selectbox("Cinsiyet", ["Kadın", "Erkek"])
education = st.sidebar.slider("Eğitim Seviyesi (1–6)", 1, 6, 3)
income = st.sidebar.slider("Gelir Seviyesi (1–8)", 1, 8, 4)

# --- Özellik dizisi (21 sütun sırasıyla) ---
input_data = np.array([[
    1 if high_bp == "Evet" else 0,
    1 if high_chol == "Evet" else 0,
    1 if chol_check == "Evet" else 0,
    bmi,
    1 if smoker == "Evet" else 0,
    1 if stroke == "Evet" else 0,
    1 if heart_disease == "Evet" else 0,
    1 if phys_activity == "Evet" else 0,
    1 if fruits == "Evet" else 0,
    1 if veggies == "Evet" else 0,
    1 if alcohol == "Evet" else 0,
    1 if healthcare == "Evet" else 0,
    1 if no_doc_cost == "Evet" else 0,
    gen_health,
    ment_health,
    phys_health,
    1 if diff_walk == "Evet" else 0,
    1 if sex == "Erkek" else 0,
    age,
    education,
    income
]])

# --- Tahmin ---
if st.sidebar.button("🔎 Tahmin Et"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ **Yüksek Diyabet Riski** — Olasılık: %{prob*100:.1f}")
    else:
        st.success(f"✅ **Düşük Diyabet Riski** — Olasılık: %{prob*100:.1f}")

    st.caption("ℹ️ Bu model sadece bilgilendirme amaçlıdır. Kesin tanı için lütfen bir uzmana danışınız.")



# SHAP 


feature_names = [
    "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke",
    "HeartDisease","PhysActivity","Fruits","Veggies",
    "HvyAlcohol","AnyHealthcare","NoDocbcCost","GenHealth",
    "MentHealth","PhysHealth","DiffWalk","Sex","Age","Education","Income"
]

st.subheader("📊 Özellik Etki Analizi")

try:
    import shap
    try:
        booster = model.get_booster()
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(input_data)
    except Exception:
        explainer = shap.Explainer(model)
        shap_values = explainer(input_data)

    fig, ax = plt.subplots(figsize=(8, 6))
    vals = shap_values if isinstance(shap_values, np.ndarray) else shap_values.values
    vals = np.squeeze(vals)
    order = np.argsort(np.abs(vals))[::-1]
    ax.barh(np.array(feature_names)[order][::-1], np.array(vals)[order][::-1])
    ax.set_xlabel("SHAP value (impact on model output)")
    st.pyplot(fig)
    plt.close(fig)

except Exception:
    st.warning("SHAP bu ortamda çalıştırılamadı, modelin yerleşik önem değerlerini gösteriyoruz.")
    try:
        importances = model.feature_importances_
        order = np.argsort(importances)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(np.array(feature_names)[order], importances[order])
        ax.set_xlabel("Feature importance (XGBoost)")
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        st.error("Özellik önem grafiği oluşturulamadı.")
