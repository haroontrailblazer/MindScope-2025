from turtle import color
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components




# ==================== PAGE CONFIG ====================
# --- Page Configuration ---
st.set_page_config(
    page_title="MindScope-2025 - Mental Health Assessment",
    page_icon="https://github.com/haroontrailblazer/MomentoMonto/blob/main/MomentoMonto-icon.png?raw=true",
    layout="centered",
    menu_items={
        'About':"""
        ### About MindScope-2025
        
        **MindScope-2025** is an educational tool designed to help individuals assess their mental health 
        using evidence-based screening tools combined with machine learning.
        
        #### Technology Stack
        - **Framework**: Streamlit
        - **ML Models**: Scikit-learn (Random Forest, Logistic Regression)
        - **Data**: Global Mental Health Dataset 2025 (synthetic)
        - **Assessment Tools**: PHQ-9 & GAD-7
        
        #### Clinical Scales Used
        
        **PHQ-9 (Patient Health Questionnaire)**
        - 9-item screening tool for depression
        - Scores: 0-27
        - Based on DSM-IV depression criteria
        - Widely used in clinical and research settings
        
        **GAD-7 (Generalized Anxiety Disorder)**
        - 7-item screening tool for anxiety
        - Scores: 0-21
        - Clinical diagnostic validity
        - Recommended by WHO
        
        #### Disclaimer
        This application is for educational and self-assessment purposes only and should not be used 
        for professional medical diagnosis or treatment. If you are experiencing a mental health crisis, 
        please contact:
        
        - **Emergency Services**: 911 (US) | 112 (India)
        - **Suicide Prevention**: 1-800-273-8255 (US)
        - **Crisis Text**: Text HOME to 741741
        
        #### Data Privacy
        All assessments are processed locally. No data is stored or transmitted to external servers.
        
        #### Disclaimer
        This tool is designed for educational purposes only and cannot replace professional 
        mental health assessment or treatment.
        """
    }
)

# --- SEO META TAGS ---
st.markdown("""
<head>
  <title>MomentoMonto - Monitor Your Servers</title>
  <meta name="description" content="Check your Server Health and Analyze the Traffic of the server — a simple, secure, and smart tool.">
  <meta name="keywords" content="haroontrailblazer, Servers, security, Server Health checker, Server strength, MomentoMonto, Haroon K M">
  <meta name="robots" content="index, follow">
</head>
""", unsafe_allow_html=True)


# ==================== GLOBAL STYLES ====================
st.markdown("""
<style>
/* Primary buttons */
div.stButton > button {
    background-color: #7c3aed;
    color: #fffdd0;
    border-radius: 12px;
    font-weight: 900;
}
div.stButton > button:hover {
    background-color: #8b5cf6;
}

/* Radio buttons */
div[role="radiogroup"] input[type="radio"] {
    accent-color: #8b5cf6;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Remove extra top padding */
.block-container {
    padding-top: 1rem !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stActionButton {display: none;}  /* hides the stop/rerun icon */
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .main { padding: 2rem; }
    .stMetric { background-color: #5b6b73; padding: 1rem; border-radius: 0.5rem; }
    .result-box { padding: 2rem; border-radius: 1rem; margin: 1rem 0; }
    .result-low { background-color: #77ef98; border-left: 5px solid #28a745; }
    .result-moderate { background-color: #00b2ff; border-left: 5px solid #ffc107; }
    .result-high { background-color: #ff0000; border-left: 5px solid #dc3545; }
    .question-card { background-color: #a100ff; padding: 1.5rem; border-radius: 0.5rem; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)




# ==================== LOAD MODELS & ENCODERS ====================
@st.cache_resource
def load_models():
    """Load pre-trained models and encoders"""
    try:
        model = joblib.load('models/best_risk_model.pkl')
        encoders = joblib.load('models/encoders.pkl')
        feature_cols = joblib.load('models/feature_cols.pkl')
        return model, encoders, feature_cols
    except FileNotFoundError:
        st.error("Models not found. Please run '02_model_training.py' first to train models.")
        st.stop()




# ==================== SOLUTIONS DATABASE ====================
SOLUTIONS_DB = {
    'Low': {
        'color': '#d4edda',
        'title': 'Low Risk - Excellent!',
        'description': 'You are in good mental health. Maintain your current wellness habits.',
        'solutions': [
            {
                'category': '🏃 Physical Activity',
                'tips': [
                    'Continue regular exercise (30 mins daily)',
                    'Explore new physical activities you enjoy',
                    'Consider outdoor activities for vitamin D exposure'
                ]
            },
            {
                'category': '😴 Sleep & Rest',
                'tips': [
                    'Maintain consistent sleep schedule (7-8 hours)',
                    'Create a relaxing bedtime routine',
                    'Avoid screens 1 hour before bed'
                ]
            },
            {
                'category': '🧘 Mental Wellness',
                'tips': [
                    'Practice mindfulness or meditation (10 mins daily)',
                    'Maintain social connections',
                    'Engage in hobbies and creative activities'
                ]
            },
            {
                'category': '🍎 Lifestyle',
                'tips': [
                    'Eat balanced, nutritious meals',
                    'Limit caffeine and sugar intake',
                    'Spend time in nature regularly'
                ]
            }
        ]
    },
    'Moderate': {
        'color': '#fff3cd',
        'title': 'Moderate Risk - Take Action',
        'description': 'You are experiencing moderate stress/depression. Consider lifestyle changes and support.',
        'solutions': [
            {
                'category': '🏃 Physical Activity',
                'tips': [
                    'Increase exercise to 45-60 minutes daily',
                    'Try activities like yoga, walking, or swimming',
                    'Join fitness classes or sports groups for social support'
                ]
            },
            {
                'category': '😴 Sleep & Rest',
                'tips': [
                    'Prioritize 7-9 hours of sleep nightly',
                    'Develop relaxation techniques (breathing, progressive muscle relaxation)',
                    'Keep bedroom cool, dark, and quiet'
                ]
            },
            {
                'category': '💬 Social Support',
                'tips': [
                    'Spend quality time with family and friends',
                    'Join community groups or clubs',
                    'Consider peer support groups',
                    'Reach out to trusted people about your feelings'
                ]
            },
            {
                'category': '🧠 Stress Management',
                'tips': [
                    'Practice daily meditation (15-20 mins)',
                    'Try journaling to process emotions',
                    'Use stress-relief apps (Calm, Headspace, Insight Timer)',
                    'Identify and reduce stressors where possible'
                ]
            },
            {
                'category': '🩺 Professional Help',
                'tips': [
                    'Consider speaking with a counselor or therapist',
                    'Consult your doctor about mental health screening',
                    'Explore cognitive behavioral therapy (CBT) resources'
                ]
            }
        ]
    },
    'High': {
        'color': "#f8d7da",
        'title': 'High Risk - Seek Professional Help',
        'description': 'You are experiencing significant stress/depression. Professional support is strongly recommended.',
        'solutions': [
            {
                'category': '🚑 Immediate Actions',
                'tips': [
                    '🔴 URGENT: Contact a mental health professional or counselor immediately',
                    'Call your doctor and schedule an appointment',
                    'Consider visiting a mental health clinic or hospital',
                    'If in crisis, call emergency services or crisis helpline'
                ]
            },
            {
                'category': '🏥 Professional Treatment',
                'tips': [
                    'Consult a psychiatrist for comprehensive evaluation',
                    'Consider therapy (CBT, DBT, or psychotherapy)',
                    'Discuss medication options with doctor if appropriate',
                    'Explore inpatient or outpatient treatment programs'
                ]
            },
            {
                'category': '🆘 Crisis Support',
                'tips': [
                    'National Suicide Prevention Lifeline: 1-800-273-8255 (US)',
                    'Crisis Text Line: Text HOME to 741741',
                    'International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/',
                    'AAMI (Befrienders India): 9152987821 (India)'
                ]
            },
            {
                'category': '🏃 Daily Wellness (Supplementary)',
                'tips': [
                    'Engage in light physical activity (short walks)',
                    'Maintain regular sleep schedule',
                    'Eat nutritious meals at regular times',
                    'Avoid alcohol and drugs',
                    'Connect with supportive people'
                ]
            },
            {
                'category': '📝 Self-Care',
                'tips': [
                    'Create a safety plan with professional guidance',
                    'Keep emergency contact numbers accessible',
                    'Maintain daily routine and structure',
                    'Track mood and symptoms in a journal'
                ]
            }
        ]
    }
}







# ==================== CLINICAL THRESHOLDS ====================
PHQ9_THRESHOLDS = {
    'Minimal': (0, 4),
    'Mild': (5, 9),
    'Moderate': (10, 14),
    'Moderately Severe': (15, 19),
    'Severe': (20, 27)
}

GAD7_THRESHOLDS = {
    'Minimal': (0, 4),
    'Mild': (5, 9),
    'Moderate': (10, 14),
    'Severe': (15, 21)
}





# ==================== PHQ-9 & GAD-7 QUESTIONS ====================
PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself or feeling that you are a failure or have let your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed, or so fidgety or restless that you have been moving around a lot more than usual",
    "Thoughts that you would be better off dead or of hurting yourself in some way"
]

GAD7_QUESTIONS = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid as if something awful might happen"
]





# ==================== HELPER FUNCTIONS ====================
def get_depression_level(score):
    """Get depression level based on PHQ-9 score"""
    if score <= 4:
        return 'Minimal'
    elif score <= 9:
        return 'Mild'
    elif score <= 14:
        return 'Moderate'
    elif score <= 19:
        return 'Moderately Severe'
    else:
        return 'Severe'




def get_anxiety_level(score):
    """Get anxiety level based on GAD-7 score"""
    if score <= 4:
        return 'Minimal'
    elif score <= 9:
        return 'Mild'
    elif score <= 14:
        return 'Moderate'
    else:
        return 'Severe'
    
    
    

def get_risk_level(dep_score, anx_score):
    """Get overall risk level based on depression and anxiety scores"""
    risk = (dep_score + anx_score) / 2
    if risk < 5:
        return 'Low'
    elif risk < 12:
        return 'Moderate'
    else:
        return 'High'
    
    
    
    

def encode_features(gender, stress, activity, illness, history, treatment, work, encoders):
    """Encode categorical features using saved encoders"""
    try:
        gender_enc = encoders['gender'].transform([gender])[0]
        stress_enc = encoders['stress'].transform([stress])[0]
        activity_enc = encoders['activity'].transform([activity])[0]
        illness_enc = encoders['illness'].transform([illness])[0]
        history_enc = encoders['history'].transform([history])[0]
        treatment_enc = encoders['treatment'].transform([treatment])[0]
        work_enc = encoders['work'].transform([work])[0]
        
        return [gender_enc, stress_enc, activity_enc, illness_enc, history_enc, treatment_enc, work_enc]
    except Exception as e:
        st.error(f"Encoding error: {e}")
        return None
    
    
    
    
    

def predict_risk(age, gender, dep_score, anx_score, stress, sleep, activity, 
                 illness, history, treatment, treatment_days, work, model, encoders, feature_cols):
    """Predict risk level using ML model"""
    try:
        encoded = encode_features(gender, stress, activity, illness, history, treatment, work, encoders)
        if encoded is None:
            return None
        
        features = np.array([[
            age, 
            encoded[0],  # gender
            dep_score,
            anx_score,
            encoded[1],  # stress
            sleep,
            encoded[2],  # activity
            encoded[3],  # illness
            encoded[4],  # history
            encoded[5],  # treatment
            treatment_days,
            encoded[6]   # work
        ]])
        
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        return prediction, probability
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None





def display_solutions(risk_level):
    """Display personalized solutions based on risk level"""
    solution_data = SOLUTIONS_DB[risk_level]
    
    # Main result box
    st.markdown(f"""
    <div class='result-box result-{risk_level.lower()}'>
        <h2>{solution_data['title']}</h2>
        <p>{solution_data['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📋 Recommended Solutions & Action Plan")
    
    # Display solutions by category
    for solution in solution_data['solutions']:
        with st.expander(f"{solution['category']}", expanded=(risk_level == 'High')):
            for tip in solution['tips']:
                st.write(f"• {tip}")





def display_assessment_results(phq9_score, gad7_score, risk_level, model_confidence):
    """Display assessment results with visualizations"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("PHQ-9 Score", f"{phq9_score}/27", 
                 f"{get_depression_level(phq9_score)}", 
                 delta_color="inverse")
    with col2:
        st.metric("GAD-7 Score", f"{gad7_score}/21", 
                 f"{get_anxiety_level(gad7_score)}", 
                 delta_color="inverse")
    with col3:
        confidence_pct = max(model_confidence) * 100
        st.metric("Risk Level", f"{risk_level}", 
                 f"Confidence: {confidence_pct:.1f}%")
    
    # Visualizations
    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Score progression gauge
        fig_phq = go.Figure(go.Indicator(
            mode="gauge+number",
            value=phq9_score,
            title="Depression (PHQ-9)",
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 27]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 4], 'color': "lightgreen"},
                    {'range': [5, 9], 'color': "yellow"},
                    {'range': [10, 14], 'color': "orange"},
                    {'range': [15, 19], 'color': "salmon"},
                    {'range': [20, 27], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 20
                }
            }
        ))
        fig_phq.update_layout(height=300)
        st.plotly_chart(fig_phq, use_container_width=True)
    
    with col_chart2:
        # Anxiety gauge
        fig_gad = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gad7_score,
            title="Anxiety (GAD-7)",
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 21]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 4], 'color': "lightgreen"},
                    {'range': [5, 9], 'color': "yellow"},
                    {'range': [10, 14], 'color': "orange"},
                    {'range': [15, 21], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 15
                }
            }
        ))
        fig_gad.update_layout(height=300)
        st.plotly_chart(fig_gad, use_container_width=True)
        
        
        
st.markdown("""
<style>
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
}
.hero-sub {
    font-size: 1.3rem;
    text-align: center;
    color: #666;
}
.card-box {
    padding: 1.5rem;
    border-radius: 12px;
    background-color: #f8f9fa;
    border: 1px solid #e6e6e6;
}
</style>
""", unsafe_allow_html=True)
      

# ==================== MAIN APP ====================
def main():
    
    # Load models
    model, encoders, feature_cols = load_models()
    if "page" not in st.session_state:
        st.session_state.page = "🏠 Home"
    page = st.session_state.page

   
    if page == "🏠 Home":
        
        st.markdown("""
        <h2 style='text-align:left; color:#8b5cf6;'>MindScope-2025</h2>
        <p style='text-align:left; color:grey; font-size:14px; margin-top:-10px;'>AI-powered mental health assessment tool, Evidence-based depression and anxiety screening with personalized recommendations.</p>""", unsafe_allow_html=True)
        
        if st.button("Start My Screening Test", icon="📋"):
            st.session_state.page = "📋 Screening Test"
            st.rerun()
            
        st.markdown("""
        <div class="footer" style="background-color:black;color:#333;padding:18px;border-radius:12px;max-width:820px;margin:20px auto;text-align:center;font-family:Segoe UI, Tahoma, sans-serif;">
            <p style="margin:8px 0 0;color:#555;font-size:13px;line-height:1.4;text-align:left;">
            MindScope-2025 is an AI-driven mental health assessment tool that helps you understand your emotional well-being using clinically validated screening methods.
            <br><br>
            It uses:<br>
            PHQ-9 for depression screening<br>
            GAD-7 for anxiety assessment<br>
            Lifestyle factors like sleep, stress, and activity<br>
            Machine learning for risk prediction<br><br>
            How it works<br>
            1. Answer 16 short questions  <br>
            2. Provide lifestyle information  <br>
            3. AI analyzes your responses  <br>
            4. Receive personalized guidance  <br>
            </p>
            <hr style="border:none;border-top:1px solid #e6e6e6;margin:12px 0;">
            <p style="margin:0 0 12px;font-size:14px;">
                <a href="https://haroontrailblazer.vercel.app" target="_blank" style="color:#8b5cf6;text-decoration:none;margin:0 8px;">Developer</a>|
                <a href="https://www.instagram.com/hendrix__trailblazer?igsh=MTEyOTEycm9mMGxjaA==" target="_blank" style="color:#8b5cf6;text-decoration:none;margin:0 8px;">Instagram</a> |
                <a href="https://github.com/haroontrailblazer" target="_blank" style="color:#8b5cf6;text-decoration:none;margin:0 8px;">GitHub</a>
            </p>
        </div>
        <br>
        """, unsafe_allow_html=True)
                    


            
            
            
            
            
            
            
            
    
    # ==================== SCREENING TEST PAGE ====================
    
    
    elif page == "📋 Screening Test":
        st.markdown("""
        <h2 style='text-align:left; color:#fffdd0;'>Mental Health Screening Test</h2>""", unsafe_allow_html=True)
        st.caption("Please answer the following questions honestly. All responses are confidential.")
        
        st.markdown("---")
        
        # Session state to store responses
        if 'test_complete' not in st.session_state:
            st.session_state.test_complete = False
            
            
            
        
        # Basic Information Section
        st.markdown("""
        <h4 style='text-align:left; color:#fffdd0;'>👤 Basic Information</h4>""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=13, max_value=100, step=1)

        with col2:
            gender = st.selectbox(
                "Gender",
                ["Male", "Female", "Other", "Prefer not to say"],
                index=0
            )

        with col3:
            work_status = st.selectbox(
                "Work Status",
                ["Student", "Employed", "Unemployed", "Retired", "Other"]
            )

        
        st.markdown("---")
        
        
        
        
        
        # PHQ-9 Section
        st.markdown("""
        <h4 style='text-align:left; color:#fffdd0;'>😞 Depression Screening (PHQ-9)</h4>""", unsafe_allow_html=True)
        st.caption("Over the last **2 weeks**, how often have you been bothered by the following problems?")

        PHQ_OPTIONS = {
            0: "Not at all",
            1: "Several days",
            2: "More than half the days",
            3: "Nearly every day"
        }

        phq9_responses = []

        for i, question in enumerate(PHQ9_QUESTIONS):
            st.markdown(f"**{i+1}. {question}**")
            response = st.radio(
                label="",
                options=list(PHQ_OPTIONS.keys()),
                format_func=lambda x: f"{x} – {PHQ_OPTIONS[x]}",
                key=f"phq9_{i}",
                horizontal=False,
            )
            phq9_responses.append(response)
            st.markdown("""---""")

        phq9_score = sum(phq9_responses)
        st.info(f"PHQ-9 Score: **{phq9_score}/27** → **{get_depression_level(phq9_score)}**")
        
        st.markdown("---")
        
        
        
        
        
        # GAD-7 Section
        st.markdown("""
        <h4 style='text-align:left; color:#fffdd0;'>😰 Anxiety Screening (GAD-7)</h4>""", unsafe_allow_html=True)
        st.caption("Over the last **2 weeks**, how often have you been bothered by the following problems?")

        gad7_responses = []

        for i, question in enumerate(GAD7_QUESTIONS):
            st.markdown(f"**{i+1}. {question}**")
            response = st.radio(
                label="",
                options=list(PHQ_OPTIONS.keys()),
                format_func=lambda x: f"{x} – {PHQ_OPTIONS[x]}",
                key=f"gad7_{i}",
                horizontal=False
            )
            gad7_responses.append(response)
            st.markdown("---")

        gad7_score = sum(gad7_responses)
        st.info(f"GAD-7 Score: **{gad7_score}/21** → **{get_anxiety_level(gad7_score)}**")

        
        st.markdown("---")
        
        
        
        
        
        
        # Lifestyle Factors Section
        st.markdown("""
        <h4 style='text-align:left; color:#fffdd0;'>🏃 Lifestyle & Health Factors</h4>""", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sleep_hours = st.slider("Average sleep per night (hours)", 0.0, 12.0, 7.0, 0.5)
        with col2:
            physical_activity = st.selectbox(
                "Physical activity level",
                ["Low", "Moderate", "High"]
            )
        with col3:
            stress_level = st.selectbox(
                "Current stress level",
                ["Low", "Moderate", "High", "Severe"]
            )
        
        chronic_illness = st.radio("Do you have any chronic illness?", ["No", "Yes"])
        mental_health_history = st.radio("Do you have a history of mental health issues?", ["No", "Yes"])
        treatment = st.selectbox(
            "Current treatment",
            ["None", "Medication", "Therapy", "Both"]
        )
        treatment_days = st.slider("Days in current treatment", 0, 365, 0)
        
        st.markdown("---")
        
        
        
        
        # Submit button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit_btn = st.button("Analyze & Get Results", icon="📊", use_container_width=True)
        
        if submit_btn:
            
            with st.spinner("Analyzing your responses..."):
                prediction = predict_risk(
                    age, gender, phq9_score, gad7_score, stress_level, sleep_hours,
                    physical_activity, chronic_illness, mental_health_history, treatment,
                    treatment_days, work_status, model, encoders, feature_cols
                )
                if prediction:
                    risk_level, probabilities = prediction
                    st.session_state.test_complete = True
                    st.session_state.phq9_score = phq9_score
                    st.session_state.gad7_score = gad7_score
                    st.session_state.risk_level = risk_level
                    st.session_state.probabilities = probabilities
                    st.success("✅ Assessment complete! Proceed to view your results.")
                    st.session_state.page = "📈 Results & Solutions"
                    st.rerun()
    
    
    
    
    
    
    
    
    
    
    
    # ==================== RESULTS PAGE ====================
    elif page == "📈 Results & Solutions":
        if not st.session_state.get('test_complete', False):
            st.warning("Please complete the screening test first!",icon="⚠️")
            if st.button("Start My Screening Test", icon="📋"):
                st.session_state.page = "📋 Screening Test"
                st.rerun()
        else:
            st.markdown("""
            <h2 style='text-align:left; color:#fffdd0;'>Your Assessment Results</h2>""", unsafe_allow_html=True)
            st.caption("Based on your responses, here are your mental health assessment results and personalized recommendations.")
            
            # Display results
            display_assessment_results(
                st.session_state.phq9_score,
                st.session_state.gad7_score,
                st.session_state.risk_level,
                st.session_state.probabilities
            )
            
            st.markdown("---")
            
            # Display solutions
            display_solutions(st.session_state.risk_level)
            
            st.markdown("---")
            
            # Additional resources
            st.markdown("")
            
            # Reset button
            if st.button("Take Assessment Again", icon="🔄", use_container_width=True):
                st.session_state.page = "📋 Screening Test"
                st.rerun()
                
                
                
                
    
    # ==================== ABOUT PAGE ====================
    elif page == "ℹ️ About":
        st.markdown("""
        ### About MindScope-2025
        
        **MindScope-2025** is an educational tool designed to help individuals assess their mental health 
        using evidence-based screening tools combined with machine learning.
        
        #### Technology Stack
        - **Framework**: Streamlit
        - **ML Models**: Scikit-learn (Random Forest, Logistic Regression)
        - **Data**: Global Mental Health Dataset 2025 (synthetic)
        - **Assessment Tools**: PHQ-9 & GAD-7
        
        #### Clinical Scales Used
        
        **PHQ-9 (Patient Health Questionnaire)**
        - 9-item screening tool for depression
        - Scores: 0-27
        - Based on DSM-IV depression criteria
        - Widely used in clinical and research settings
        
        **GAD-7 (Generalized Anxiety Disorder)**
        - 7-item screening tool for anxiety
        - Scores: 0-21
        - Clinical diagnostic validity
        - Recommended by WHO
        
        #### Disclaimer
        This application is for educational and self-assessment purposes only and should not be used 
        for professional medical diagnosis or treatment. If you are experiencing a mental health crisis, 
        please contact:
        
        - **Emergency Services**: 911 (US) | 112 (India)
        - **Suicide Prevention**: 1-800-273-8255 (US)
        - **Crisis Text**: Text HOME to 741741
        
        #### Data Privacy
        All assessments are processed locally. No data is stored or transmitted to external servers.
        
        #### Disclaimer
        This tool is designed for educational purposes only and cannot replace professional 
        mental health assessment or treatment.
        """)

if __name__ == "__main__":
    main()
