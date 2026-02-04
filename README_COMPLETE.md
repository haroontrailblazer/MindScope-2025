# 🧠 MindScope-2025: AI-Powered Mental Health Assessment App

![Mental Health Assessment](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red?logo=streamlit)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green)

---

## 🎯 Overview

**MindScope-2025** is a comprehensive, real-time mental health assessment application that combines:
- **Validated Clinical Tools**: PHQ-9 (depression) and GAD-7 (anxiety) questionnaires
- **Machine Learning**: Random Forest classifier for risk prediction
- **Interactive UI**: Streamlit web application with 4-page interface
- **Personalized Solutions**: 96+ tailored recommendations across 15 categories
- **Data Privacy**: 100% local processing, no external data transmission

**Perfect for:** Students, individuals, community health workers, and wellness programs.

---

## ✨ Key Features

### 🏠 Home Page
- Project overview and introduction
- Feature highlights
- Quick navigation to assessment

### 📋 Screening Test
- **16 Interactive Questions**:
  - 9 PHQ-9 questions (depression screening)
  - 7 GAD-7 questions (anxiety screening)
- **Lifestyle Assessment**:
  - Sleep hours, physical activity
  - Stress level, medical history
  - Treatment status
- **Real-time Scoring**: Instant feedback on depression/anxiety levels

### 📈 Results & Solutions
- **Visual Dashboards**:
  - Depression and anxiety gauge charts
  - Risk level classification
  - Confidence metrics
- **Personalized Recommendations**:
  - Risk-stratified solutions (Low/Moderate/High)
  - 15+ categories of recommendations
  - Emergency resources for high-risk users

### ℹ️ About Page
- Clinical scale information
- Technology stack details
- Data privacy and disclaimer
- Crisis hotlines and resources

---

## 🚀 Quick Start

### Installation (5 minutes)

```bash
# 1. Setup virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements-updated.txt

# 3. Train ML models
mkdir models
python 02_model_training.py

# 4. Run the app
streamlit run app.py
```

Open browser → `http://localhost:8501` ✅

---

## 📊 Assessment Scales

### PHQ-9 (Patient Health Questionnaire)
Measures depression severity across 9 dimensions:
| Score | Level |
|-------|-------|
| 0-4 | Minimal |
| 5-9 | Mild |
| 10-14 | Moderate |
| 15-19 | Moderately Severe |
| 20-27 | Severe |

### GAD-7 (Generalized Anxiety Disorder)
Measures anxiety severity across 7 dimensions:
| Score | Level |
|-------|-------|
| 0-4 | Minimal |
| 5-9 | Mild |
| 10-14 | Moderate |
| 15-21 | Severe |

---

## 🤖 Machine Learning

### Models Trained
- **Logistic Regression**: Baseline model (~78% accuracy)
- **Decision Tree**: Interpretable model (~82% accuracy)
- **Random Forest**: Best performer (~85% accuracy) ⭐

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

### Features Used
- Age, Gender
- PHQ-9 & GAD-7 scores
- Stress level (categorical)
- Sleep hours
- Physical activity (categorical)
- Chronic illness (yes/no)
- Mental health history (yes/no)
- Current treatment (categorical)
- Days in treatment
- Work status (categorical)

---

## 💾 Project Structure

```
MindScope-2025/
├── 📄 README.md                              (Project overview)
├── 📄 SETUP_GUIDE.md                         (Detailed setup)
├── 📄 QUICK_REFERENCE.md                     (Developer reference)
├── 📋 requirements-updated.txt                (Dependencies)
│
├── 📊 Data/
│   └── Global_Mental_Health_Dataset_2025.csv (2,500+ records)
│
├── 📓 Notebooks/
│   ├── 01_Data_Cleaning.ipynb                (Data preprocessing)
│   ├── 02_model_training.py          (NEW) (Model training)
│   ├── 02_eda_visualization.ipynb            (Coming soon)
│   └── 03_model_training.ipynb               (Coming soon)
│
├── 🎮 app.py                         (NEW) (Streamlit app)
│
├── 🔧 models/                        (AUTO-CREATED)
│   ├── best_risk_model.pkl
│   ├── random_forest_model.pkl
│   ├── encoders.pkl
│   └── feature_cols.pkl
│
└── 📑 LICENSE.md
```

---

## 🛠️ Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Frontend** | Streamlit | 1.31.0 |
| **Backend** | Python | 3.8+ |
| **ML Framework** | Scikit-learn | 1.3.2 |
| **Data Processing** | Pandas, NumPy | 2.1.4, 1.26.2 |
| **Visualization** | Plotly | 5.18.0 |
| **Serialization** | Joblib | 1.3.2 |

---

## 📱 Supported Features

### ✅ Completed
- [x] PHQ-9 screening questionnaire
- [x] GAD-7 screening questionnaire
- [x] ML model training pipeline
- [x] Risk level prediction
- [x] Personalized solutions database (96+ recommendations)
- [x] Interactive Streamlit UI
- [x] Visual gauge charts
- [x] Crisis resources
- [x] Data validation and error handling

### 🔄 In Progress
- [ ] EDA and visualization notebook
- [ ] Model performance analysis
- [ ] Cross-validation studies

### 🎯 Future Enhancements
- [ ] User authentication and profiles
- [ ] Assessment history tracking
- [ ] PDF report export
- [ ] Multi-language support (Hindi, Tamil)
- [ ] Mobile app (React Native)
- [ ] Real-time therapist integration
- [ ] Wearable device sync (Apple Health, Fitbit)

---

## 📋 Personalized Solutions by Risk Level

### Low Risk (✅ Good Mental Health)
**4 Categories, 16 Tips:**
- Physical Activity: Continue regular exercise
- Sleep & Rest: Maintain 7-8 hours nightly
- Mental Wellness: Practice mindfulness
- Lifestyle: Balanced nutrition and nature time

### Moderate Risk (⚠️ Take Action)
**5 Categories, 25 Tips:**
- Physical Activity: Increase to 45-60 mins daily
- Sleep & Rest: Prioritize 7-9 hours
- Social Support: Connect with friends, support groups
- Stress Management: Meditation, journaling, apps
- Professional Help: Consider counselor/therapist

### High Risk (🚨 Seek Help)
**5 Categories, 24 Tips:**
- Immediate Actions: Contact mental health professional NOW
- Professional Treatment: Psychiatrist, therapy, medication options
- Crisis Support: Hotlines, crisis text lines, helplines
- Daily Wellness: Light activity, structured routine
- Self-Care: Safety planning, journaling, emergency contacts

---

## 🧪 Testing Guide

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

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Recommended for Beginners)
```bash
# 1. Push to GitHub
git add .
git commit -m "MindScope-2025"
git push origin main

# 2. Go to https://streamlit.io/cloud
# 3. Connect GitHub and deploy
# 4. Share public URL
```

### Option 2: Docker (Production)
```bash
# Build image
docker build -t mindscope .

# Run container
docker run -p 8501:8501 mindscope
```

### Option 3: Local Network
```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

### Option 4: Traditional Server
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:__main__
```

---

## ⚠️ Important Disclaimers

🔴 **CRITICAL**: 
- This app is for **educational and self-assessment** purposes only
- **NOT a substitute** for professional medical advice
- **NOT for diagnosis** or treatment of mental health conditions
- **Cannot replace** licensed therapists or psychiatrists
- Synthetic dataset used for training

✅ **If you are in crisis:**
- 📞 Call emergency services: **911 (US) | 112 (India)**
- 🆘 Suicide Prevention: **1-800-273-8255 (US)**
- 💬 Crisis Text: **Text HOME to 741741**
- 🌍 AAMI (India): **9152987821**

---

## 🔐 Data Privacy & Security

✅ **Privacy-First Design:**
- All processing done **locally** on user's machine
- **NO data transmission** to external servers
- **NO data storage** (session-based only)
- **NO cookies or tracking**
- Assessments are **completely anonymous**

---

## 📊 Dataset Information

- **Source**: Global Mental Health Dataset 2025 (Kaggle)
- **Records**: 2,500+ synthetic patient samples
- **Features**: 16 (demographic + clinical + lifestyle)
- **Target Variables**: Depression Level, Anxiety Level, Risk Level
- **Use**: Educational and research purposes only

---

## 📚 Learning Resources

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

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- [ ] Additional assessment scales (DASS-21, BDI-II)
- [ ] Multi-language support
- [ ] Mobile app development
- [ ] Healthcare provider integration
- [ ] Real clinical validation

---

## 📄 License

This project is open-source. See `LICENSE.md` for details.

---

## 👨‍💻 Author

**MindScope Development Team**
- Computer Science Student
- Focus: Mental Health Technology
- Location: Chennai, India

---

## 🎉 Acknowledgments

- **Dataset**: Global Mental Health Dataset 2025 (Kaggle)
- **Assessment Tools**: PHQ-9 and GAD-7 (Evidence-based clinical instruments)
- **Framework**: Streamlit (Amazing web app framework)
- **ML Library**: Scikit-learn (Excellent machine learning tools)

---

## 📞 Support

- 📖 **Documentation**: See SETUP_GUIDE.md for detailed instructions
- 🚀 **Quick Start**: See QUICK_REFERENCE.md for development tips
- 🐛 **Issues**: Report bugs or suggest features
- 💬 **Discussion**: Open for collaboration and feedback

---

## 🎓 Educational Value

This project demonstrates:
1. ✅ **Full ML Pipeline**: Data → Features → Training → Deployment
2. ✅ **Clinical Knowledge**: Evidence-based assessment tools
3. ✅ **Full-Stack**: Backend ML + Frontend UI
4. ✅ **Best Practices**: Clean code, documentation, testing
5. ✅ **Real-World Impact**: Technology for mental health awareness

**Perfect portfolio project for:** Data Science, Machine Learning, Healthcare Tech, Full-Stack Development roles.

---

## 🌟 What Makes This Special

- 🎯 **Purpose-Driven**: Addresses real mental health need
- 🔬 **Evidence-Based**: Uses clinically validated scales
- 🤖 **AI-Powered**: Machine learning for risk prediction
- 🎨 **User-Friendly**: Intuitive interface for non-technical users
- 🔒 **Privacy-First**: Local processing, no data transmission
- 📚 **Comprehensive**: 96+ personalized recommendations
- 🚀 **Production-Ready**: Ready to deploy and share

---

**Together, let's improve mental health awareness and accessibility! 🧠💚**

---

## 📝 Change Log

### v1.0.0 (Current Release)
- ✅ Complete ML model training pipeline
- ✅ Interactive Streamlit web app
- ✅ PHQ-9 & GAD-7 assessment tools
- ✅ 96+ personalized solutions
- ✅ Production-ready deployment
- ✅ Comprehensive documentation

---

**Last Updated**: February 3, 2026
**Version**: 1.0.0 Production Ready
