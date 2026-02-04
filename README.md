## MindScope-2025 
![1000166454](https://github.com/user-attachments/assets/93d0677b-9de0-4260-b141-fd0994a110fd)

Mental Health Status Analysis using AI

> AI-driven mental health analytics project using PHQ-9, GAD-7, and stress indicators to identify risk patterns through data science and machine learning.

---

## Overview

**MindScope-2025** is an end-to-end data science and machine learning project focused on analyzing mental health trends using standardized psychological screening tools:

- **PHQ-9** — Depression Severity  
- **GAD-7** — Anxiety Severity  
- **Stress Indicators (PSS-style features)**  

The project applies statistical analysis, visualization, and ML models to understand relationships between mental health factors and predict risk categories, demonstrating real-world healthcare analytics using AI.

---

## Objectives

- Clean and preprocess mental health survey data  
- Categorize depression and anxiety using clinical thresholds  
- Perform Exploratory Data Analysis (EDA)  
- Analyze correlation between mental health variables  
- Build ML models for risk prediction  
- Visualize trends and insights for interpretation  

---

## Dataset Description

The dataset contains synthetic patient-level mental health records collected using standardized psychological screening tools and lifestyle indicators.

### Features Included

- Age  
- Gender  
- Country  
- Depression_Score (PHQ-9)  
- Anxiety_Score (GAD-7)  
- Stress_Level  
- Sleep_Hours  
- Physical_Activity  
- Chronic_Illness  
- Mental_Health_History  
- Treatment  
- Days_of_Treatment  
- Outcome  
- Work_Status  

### Target Variables

- Depression_Level (Derived from PHQ-9 thresholds)  
- Anxiety_Level (Derived from GAD-7 thresholds)

---

## Psychological Scales Used

### 🔹 PHQ-9 (Depression Levels)

| Score | Level |
|--------|--------|
| 0–4 | Minimal |
| 5–9 | Mild |
| 10–14 | Moderate |
| 15–19 | Moderately Severe |
| 20–27 | Severe |

### 🔹 GAD-7 (Anxiety Levels)

| Score | Level |
|--------|--------|
| 0–4 | Minimal |
| 5–9 | Mild |
| 10–14 | Moderate |
| 15–21 | Severe |

---

## Exploratory Data Analysis (EDA)

Key analyses performed:

- Distribution of depression and anxiety severity  
- Stress level vs sleep and activity patterns  
- Gender-wise and age-wise mental health trends  
- Treatment outcome comparison  
- PHQ-9 vs GAD-7 correlation analysis  

Visualizations include:

- Histograms  
- Bar charts  
- Box plots  
- Correlation heatmaps  

---

## Machine Learning Models

The project applies supervised learning to predict mental health risk categories:

- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  

### Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

Model comparison is used to select the best-performing approach.

---

## Tools & Technologies

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Jupyter Notebook  

---
## Application Architecture
```
Data Layer
│
├─ CSV Dataset (2,500+ records)
│  └─ 16 features + targets
│
Model Layer
│
├─ 02_model_training.py
│  ├─ Load & Clean Data
│  ├─ Feature Engineering
│  ├─ Train 3 Models
│  └─ Export Best Model (85% accuracy)
│
Application Layer
│
└─ app.py (Streamlit)
   ├─ Page 1: Home (Overview)
   ├─ Page 2: Test (16 Questions)
   ├─ Page 3: Results (Dashboards + Solutions)
   └─ Page 4: About (Resources)
```

## Project Structure
```
MindScope-2025/
├─ Data/
├── 01_Data_Cleaning.ipynb (original)
├── Global_Mental_Health_Dataset_2025.csv (original)
│
├── 02_model_training.py
├── app.py
├── requirements.txt
├── README.md
│
└── models/ (auto-created after training)
    ├── best_risk_model.pkl
    ├── encoders.pkl
    └── feature_cols.pkl
```
