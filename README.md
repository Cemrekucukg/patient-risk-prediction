# 🩺 Patient Risk Prediction System

This project predicts the **likelihood of diabetes** based on personal health indicators such as BMI, age, cholesterol, blood pressure, and lifestyle habits.

It is a full **end-to-end data science application**, including:
- Data preprocessing and model training in Python  
- XGBoost-based classification model  
- Interactive Streamlit web interface  
- SHAP explainability visualization  

---

## 🚀 Demo

> Coming soon — will be live on Streamlit Cloud after deployment  
> (Example link: https://cemrekucukg-patient-risk-prediction.streamlit.app)

---

## 📊 Project Overview

| Stage | Description |
|--------|-------------|
| **Data Source** | [Diabetes Health Indicators Dataset (CDC BRFSS 2015)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) |
| **Model** | XGBoost Classifier (binary classification) |
| **Explainability** | SHAP summary plot — feature importance visualization |
| **Interface** | Streamlit app for real-time risk prediction |
| **Deployment** | Streamlit Cloud (free hosting) |

---

## 🧠 Tech Stack

- **Language:** Python 3.11  
- **Libraries:** pandas, numpy, scikit-learn, xgboost, streamlit, shap, matplotlib, seaborn, joblib  
- **Environment:** VS Code (venv)  
- **Deployment:** Streamlit Cloud  

---

## 📁 Project Structure

patient-risk-prediction/
├── app/
│ └── app.py # Streamlit web interface
├── data/
│ └── diabetes_binary_health_indicators_BRFSS2015.csv
├── models/
│ └── diabetes_xgb.pkl # Trained XGBoost model
├── notebooks/
│ └── 1_data_analysis_and_model.ipynb
├── requirements.txt
└── README.md


---

## ▶️ How to Run Locally

1️⃣ Clone this repository  
```bash
git clone https://github.com/Cemrekucukg/patient-risk-prediction.git
cd patient-risk-prediction
```

2️⃣ Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
```

3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Run the Streamlit app
```bash
streamlit run app/app.py
```

---

## 📊 Example Output

After entering your health data, the system predicts your **diabetes risk**:

> ✅ **Low Risk — 9.7% Probability**

Below the prediction, a **SHAP bar plot** shows which features most influenced the model’s decision.

---

## 🧩 Explainability (SHAP)

The project uses **SHAP (SHapley Additive exPlanations)** to interpret the model’s predictions.

- 🔺 Positive SHAP values → increase diabetes risk  
- 🔻 Negative SHAP values → decrease risk  
- The bar plot helps visualize which health factors contribute most to predictions.

---

## 📜 License

This project is licensed under the **MIT License** — free for educational and professional use.

---

## 👩‍💻 Author

**Cemre Küçükgöde**  
🎓 *MSc in Computer Engineering, Ege University*  

💻 **Portfolio:** [cemrekucukgode.com](https://cemrekucukgode.com)  
📂 **GitHub:** [github.com/Cemrekucukg](https://github.com/Cemrekucukg)  
🔗 **LinkedIn:** [linkedin.com/in/cemre-kucukgode-](https://linkedin.com/in/cemre-kucukgode-)

---

> _This model is for informational and educational purposes only.  
> It does not provide a medical diagnosis._
