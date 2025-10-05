# 🩺 Disease Prediction from Medical Data

This project predicts the likelihood of **heart disease** based on patient data using **Machine Learning algorithms** such as **SVM**, **Logistic Regression**, **Random Forest**, and **XGBoost**.

---

## 📘 Overview

The goal of this project is to use structured medical data (like age, cholesterol, blood pressure, etc.) to automatically predict whether a person is at risk of heart disease.

This system can help in early diagnosis and risk analysis for patients, assisting doctors with data-driven decision support.

---

## 🧠 Algorithms Used
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

---

## 🧩 Dataset
- **Name:** UCI Heart Disease Dataset  
- **Samples:** 920  
- **Features:** 18 (age, sex, chest pain type, cholesterol, resting BP, etc.)  
- **Target:** `num` (0 = No Disease, 1 = Disease)

> Dataset link: [UCI Heart Disease Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)

---

## ⚙️ Workflow

1. **Data Loading and Exploration**  
   - Loaded the dataset using Pandas  
   - Checked missing values and datatypes  

2. **Data Preprocessing**  
   - Dropped irrelevant columns (`id`, `dataset`)  
   - Filled missing values (median for numeric, mode for categorical)  
   - Encoded categorical features using one-hot encoding  

3. **Model Training**  
   - Split data (80% training / 20% testing)  
   - Trained four ML models  
   - Evaluated accuracy and performance metrics  

4. **Model Evaluation**  
   - Compared models with accuracy bar chart  
   - Generated confusion matrix and classification report  

5. **Prediction Demo**  
   - Built a small demo where user input predicts heart disease risk  

6. **Model Saving**  
   - Saved best model (XGBoost) using `joblib`  
   - Stored model in Google Drive and GitHub  

---

## 📊 Results

| Model | Accuracy |
|--------|-----------|
| SVM | 71.2% |
| Logistic Regression | 79.9% |
| Random Forest | 86.9% |
| **XGBoost** | **87.5% (Best)** |

---

## 🧠 Example Prediction

```python
🔴 The model predicts: Heart Disease Detected
💾 Model File
The trained model is stored as:

Copy code
heart_disease_model.pkl
You can load and use it like this:

python
Copy code
import joblib
model = joblib.load("heart_disease_model.pkl")
prediction = model.predict(sample_df)
🚀 Future Improvements
Build a Streamlit web interface for live predictions

Deploy on Hugging Face or Render

Add support for multiple disease datasets (Diabetes, Breast Cancer)

👨‍💻 Author
Muhammad Tabish Adan
📧 muhammadtabishadan456@gmail.com

🏷️ License
This project is open-source and available under the MIT License.

⭐ How to Run
bash
Copy code
# Install dependencies
pip install pandas scikit-learn xgboost matplotlib seaborn joblib

# Run the script
python heart_disease_prediction.py
💬 How it works
“I designed this project with a simple and easy-to-use workflow. There’s an upload button where users can select an image or enter patient data. After clicking the ‘Analyze’ button, the system scans and predicts the disease result instantly.”







