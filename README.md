## MindScope-2025 — Mental Health Status Analysis using AI

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

## Dataset Features

| Feature | Description |
|--------|------------|
| Age | Patient age |
| Gender | Male / Female |
| Country | Patient location |
| Depression_Score | PHQ-9 score (0–27) |
| Anxiety_Score | GAD-7 score (0–21) |
| Stress_Level | Low / Moderate / High |
| Sleep_Hours | Average sleep duration |
| Physical_Activity | Activity level |
| Chronic_Illness | Yes / No |
| Treatment | Therapy / Medication / Both / None |
| Outcome | Treatment outcome |
| Work_Status | Employment status |

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

## Project Structure

