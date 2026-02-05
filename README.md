  <div align=center>
   
  ## MindScope-2025: AI-Powered Mental Health Assessment App
   
  ![Mental Health Assessment](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
  ![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red?logo=streamlit)
  ![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange?logo=scikit-learn)
  ![Status](https://img.shields.io/badge/Status-Production%20Ready-green)
  
  ![1000166454](https://github.com/user-attachments/assets/93d0677b-9de0-4260-b141-fd0994a110fd)
  </div>
  
  Mental Health Status Analysis using AI
  
  > AI-driven mental health analytics project using PHQ-9, GAD-7, and stress indicators to identify risk patterns through data science and machine learning.

---
<br>

## Overview

**MindScope-2025** is an end-to-end data science and machine learning project focused on analyzing mental health trends using standardized psychological screening tools:

- **PHQ-9** — Depression Severity  
- **GAD-7** — Anxiety Severity  
- **Stress Indicators (PSS-style features)**  

The project applies statistical analysis, visualization, and ML models to understand relationships between mental health factors and predict risk categories, demonstrating real-world healthcare analytics using AI.

---
<br>

## Objectives

- Clean and preprocess mental health survey data  
- Categorize depression and anxiety using clinical thresholds  
- Perform Exploratory Data Analysis (EDA)  
- Analyze correlation between mental health variables  
- Build ML models for risk prediction  
- Visualize trends and insights for interpretation  

---
<br>

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
<br>

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
<br>

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
<br>

## Machine Learning

### Models Trained
- **Logistic Regression**: Baseline model (~78% accuracy)
- **Decision Tree**: Interpretable model (~82% accuracy)
- **Random Forest**: Best performer (~85% accuracy)

### Performance Metrics
```
Best Model: Random Forest Classifier
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Train Accuracy:  95.2%
Test Accuracy:   85.3%
Precision:       0.84
Recall:          0.85
F1-Score:        0.84
```
 

### Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

Model comparison is used to select the best-performing approach.

---
<br>

## Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Frontend** | Streamlit | 1.31.0 |
| **Backend** | Python | 3.8+ |
| **ML Framework** | Scikit-learn | 1.3.2 |
| **Data Processing** | Pandas, NumPy | 2.1.4, 1.26.2 |
| **Visualization** | Plotly | 5.18.0 |
| **Serialization** | Joblib | 1.3.2 |

---
<br>

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
<br>

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
---
<br>

## Supported Features

### Completed
- [x] PHQ-9 screening questionnaire
- [x] GAD-7 screening questionnaire
- [x] ML model training pipeline
- [x] Risk level prediction
- [x] Personalized solutions database (96+ recommendations)
- [x] Interactive Streamlit UI
- [x] Visual gauge charts
- [x] Crisis resources
- [x] Data validation and error handling

### In Progress
- [ ] EDA and visualization notebook
- [ ] Model performance analysis
- [ ] Cross-validation studies

### Future Enhancements
- [ ] User authentication and profiles
- [ ] Assessment history tracking
- [ ] PDF report export
- [ ] Multi-language support (Hindi, Tamil)
- [ ] Mobile app (React Native)
- [ ] Real-time therapist integration
- [ ] Wearable device sync (Apple Health, Fitbit)

---
<br>

## Personalized Solutions by Risk Level

### Low Risk (Good Mental Health)
**4 Categories, 16 Tips:**
- Physical Activity: Continue regular exercise
- Sleep & Rest: Maintain 7-8 hours nightly
- Mental Wellness: Practice mindfulness
- Lifestyle: Balanced nutrition and nature time

### Moderate Risk (Take Action)
**5 Categories, 25 Tips:**
- Physical Activity: Increase to 45-60 mins daily
- Sleep & Rest: Prioritize 7-9 hours
- Social Support: Connect with friends, support groups
- Stress Management: Meditation, journaling, apps
- Professional Help: Consider counselor/therapist

### High Risk (Seek Help)
**5 Categories, 24 Tips:**
- Immediate Actions: Contact mental health professional NOW
- Professional Treatment: Psychiatrist, therapy, medication options
- Crisis Support: Hotlines, crisis text lines, helplines
- Daily Wellness: Light activity, structured routine
- Self-Care: Safety planning, journaling, emergency contacts

---
<br>

## Testing Guide

### Test Case 1: Low Risk Assessment
```
PHQ-9: 2 (all minimal responses)
GAD-7: 3 (all minimal responses)
Expected: Low Risk → Maintenance tips
```

### Test Case 2: Moderate Risk Assessment
```
PHQ-9: 12 (moderate depression)
GAD-7: 10 (moderate anxiety)
Expected: Moderate Risk → Intervention strategies
```

### Test Case 3: High Risk Assessment
```
PHQ-9: 24 (severe depression)
GAD-7: 18 (severe anxiety)
Expected: High Risk → Professional help recommendation
```

---
<br>

## Important Disclaimers

**CRITICAL**: 
- This app is for **educational and self-assessment** purposes only
- **NOT a substitute** for professional medical advice
- **NOT for diagnosis** or treatment of mental health conditions
- **Cannot replace** licensed therapists or psychiatrists
- Synthetic dataset used for training

**If you are in crisis:**
- Call emergency services: **911 (US) | 112 (India)**
- Suicide Prevention: **1-800-273-8255 (US)**
- Crisis Text: **Text HOME to 741741**
- AAMI (India): **9152987821**

---
<br>

## Data Privacy & Security

**Privacy-First Design:**
- All processing done **locally** on user's machine
- **NO data transmission** to external servers
- **NO data storage** (session-based only)
- **NO cookies or tracking**
- Assessments are **completely anonymous**

---
<br>

## Dataset Information

- **Source**: Global Mental Health Dataset 2025 (Kaggle)
- **Records**: 2,500+ synthetic patient samples
- **Features**: 16 (demographic + clinical + lifestyle)
- **Target Variables**: Depression Level, Anxiety Level, Risk Level
- **Use**: Educational and research purposes only

---
<br>

## Learning Resources

### Assessment Scales
- [PHQ-9 Official](https://www.phqscreeners.com/)
- [GAD-7 Details](https://adaa.org/gad-7)

### Mental Health
- [MIND (UK)](https://www.mind.org.uk/)
- [NAMI (USA)](https://www.nami.org/)
- [WHO Mental Health](https://www.who.int/teams/mental-health-and-substance-use)

### Technical Documentation
- [Streamlit Docs](https://docs.streamlit.io/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [Python ML Handbook](https://scikit-learn.org/stable/index.html)

---

## License

This project is open-source. See `LICENSE.md` for details.


**Last Updated**: February 3, 2026
**Version**: 1.0.0 Production Ready
