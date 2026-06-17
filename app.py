"""
MindScope-2025 — AI-powered mental health screening (PHQ-9 + GAD-7).

UI: "Warm Wellness" design system — sage + clay on warm sand, serif (Fraunces)
display headings, organic soft cards, and a guided step-by-step assessment wizard.
Model & clinical logic unchanged.
"""

import numpy as np
import joblib
import streamlit as st
import plotly.graph_objects as go


# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="MindScope — Mental Wellbeing Assessment",
    page_icon="https://github.com/haroontrailblazer/haroontrailblazer/blob/main/Project%20Pngs/ico.png?raw=true",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# SEO + Open Graph / Twitter meta. Single source of truth in og_meta.html, which is
# also injected into Streamlit's served index.html at build time (see Dockerfile) so
# social crawlers — which don't run JS — can read the share-card tags.
try:
    with open("og_meta.html", encoding="utf-8") as _f:
        st.markdown(_f.read(), unsafe_allow_html=True)
except FileNotFoundError:
    pass


# ==================== DESIGN TOKENS (Warm Wellness) ====================
CANVAS   = "#FBF7F0"   # warm sand
SURFACE  = "#FFFFFF"
SAND     = "#F3EEE4"
INK      = "#2E2A24"   # warm near-black
BODY     = "#6B6258"
MUTED    = "#A89F92"
LINE     = "#ECE4D8"
SAGE     = "#7C9473"
SAGE_DK  = "#5F7A57"
CLAY     = "#C2703D"

RISK_THEME = {
    "Low":      {"bg": "#EEF3E9", "ring": "#7C9473", "ink": "#3F5238", "glyph": "check-circle"},
    "Moderate": {"bg": "#FBF0E6", "ring": "#C2703D", "ink": "#8A4B22", "glyph": "warning"},
    "High":     {"bg": "#FBE9E4", "ring": "#B5462C", "ink": "#7E2C1A", "glyph": "warning-octagon"},
}


def icon(name, fill=False):
    """Return a Phosphor icon <i> tag."""
    return f'<i class="{"ph-fill" if fill else "ph"} ph-{name}"></i>'


# ==================== GLOBAL STYLES ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Nunito+Sans:wght@400;500;600;700;800&display=swap');
@import url('https://cdn.jsdelivr.net/npm/@phosphor-icons/web@2.1.1/src/regular/style.css');
@import url('https://cdn.jsdelivr.net/npm/@phosphor-icons/web@2.1.1/src/fill/style.css');

:root { --serif: 'Fraunces', Georgia, 'Times New Roman', serif; }

html, body, [class*="css"], .stApp, [data-testid="stMarkdownContainer"], input, button, select, textarea {
    font-family: 'Nunito Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
[class^="ph"], [class*=" ph"] { line-height: 1; vertical-align: middle; }

/* ---------- Ambient background ---------- */
html, body, .stApp { overflow-x: hidden; }
.stApp {
    background-color: #FBF7F0;
    background-image:
        radial-gradient(circle at 14% 15%, rgba(124,148,115,0.16), transparent 36%),
        radial-gradient(circle at 86% 9%,  rgba(194,112,61,0.11),  transparent 38%),
        radial-gradient(circle at 83% 85%, rgba(124,148,115,0.14), transparent 40%),
        radial-gradient(circle at 15% 90%, rgba(194,112,61,0.09),  transparent 38%),
        radial-gradient(rgba(95,80,55,0.035) 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 100% 100%, 100% 100%, 26px 26px;
    background-position: 0 0, 0 0, 0 0, 0 0, 0 0;
    background-attachment: fixed;
}
/* soft organic blobs floating in the gutters, behind the content */
.stApp::before, .stApp::after {
    content: ""; position: fixed; z-index: 0; pointer-events: none;
    border-radius: 46% 54% 58% 42% / 52% 44% 56% 48%;
    filter: blur(16px);
}
.stApp::before {
    top: 130px; left: -50px; width: 230px; height: 230px;
    background: radial-gradient(circle, rgba(124,148,115,0.22), transparent 68%);
}
.stApp::after {
    bottom: 90px; right: -60px; width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(194,112,61,0.17), transparent 70%);
}

.block-container {
    position: relative;
    z-index: 1;
    max-width: 880px !important;
    padding-top: 1.2rem !important;
    padding-bottom: 3rem !important;
}

#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
.stActionButton { display: none; }

/* ---------- Brand header ---------- */
.ms-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 22px; margin: 0 0 6px;
    background: #FFFFFF; border: 1px solid #ECE4D8;
    border-radius: 20px; box-shadow: 0 10px 30px -22px rgba(95,80,55,.4);
}
.ms-brand { display: flex; align-items: center; gap: 13px; }
.ms-logo {
    width: 44px; height: 44px; border-radius: 50% 50% 50% 12px;
    background: linear-gradient(135deg, #93A985, #5F7A57);
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-size: 1.4rem; box-shadow: 0 8px 18px -6px rgba(95,122,87,.55);
}
.ms-brand-name { font-family: var(--serif); font-size: 1.34rem; font-weight: 600; color: #2E2A24; letter-spacing: -.01em; line-height: 1.05; }
.ms-brand-tag  { font-size: .68rem; font-weight: 700; color: #A89F92; letter-spacing: .18em; text-transform: uppercase; margin-top: 2px; }
.ms-nav-chip {
    font-size: .68rem; font-weight: 700; color: #5F7A57; letter-spacing: .06em;
    background: #EEF3E9; border: 1px solid #DCE7D4; padding: 8px 13px; border-radius: 999px;
    display: inline-flex; align-items: center; gap: 7px;
}

/* ---------- Wizard progress ---------- */
.wz-prog { display: flex; gap: 8px; margin: 8px 0 6px; }
.wz-seg { flex: 1; height: 9px; border-radius: 999px; background: #ECE4D8; transition: all .25s ease; }
.wz-seg.done { background: #7C9473; }
.wz-seg.active { background: #7C9473; box-shadow: 0 0 0 4px rgba(124,148,115,.18); }
.wz-meta { font-size: .76rem; font-weight: 800; color: #5F7A57; letter-spacing: .12em; text-transform: uppercase; margin-bottom: 12px; }

/* ---------- Hero ---------- */
.ms-hero {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, #5F7A57 0%, #7C9473 58%, #93A985 100%);
    border-radius: 30px; padding: 50px 44px; margin: 14px 0 18px;
    box-shadow: 0 30px 60px -30px rgba(95,122,87,.7);
}
.ms-hero::before {
    content: ""; position: absolute; right: -60px; bottom: -90px;
    width: 260px; height: 260px; border-radius: 47% 53% 60% 40%;
    background: rgba(194,112,61,.22);
}
.ms-hero::after {
    content: ""; position: absolute; right: -40px; top: -80px;
    width: 200px; height: 200px; border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,.2), transparent 70%);
}
.ms-hero-badge {
    position: relative; display: inline-flex; align-items: center; gap: 8px;
    font-size: .68rem; font-weight: 800; letter-spacing: .14em;
    color: #FBF7F0; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.28);
    padding: 7px 14px; border-radius: 999px; margin-bottom: 18px;
}
.ms-hero h1 { position: relative; font-family: var(--serif); color: #FFFFFF; font-size: 2.7rem; font-weight: 600; line-height: 1.08; letter-spacing: -.01em; margin: 0 0 14px; }
.ms-hero p  { position: relative; color: #EAF0E5; font-size: 1.04rem; line-height: 1.65; margin: 0; max-width: 600px; }

/* ---------- Section header ---------- */
.ms-sec { display: flex; align-items: center; gap: 13px; margin: 2px 0 6px; }
.ms-sec-ic {
    width: 42px; height: 42px; border-radius: 50% 50% 50% 14px; flex: none;
    background: #EEF3E9; color: #5F7A57; font-size: 1.25rem;
    display: flex; align-items: center; justify-content: center;
}
.ms-sec-tt { font-family: var(--serif); font-size: 1.3rem; font-weight: 600; color: #2E2A24; line-height: 1.15; }
.ms-sec-sb { font-size: .84rem; color: #A89F92; margin-top: 2px; }

/* feature grid */
.ms-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 4px 0 6px; }
.ms-feat { background: #FFFFFF; border: 1px solid #ECE4D8; border-radius: 22px; padding: 24px; box-shadow: 0 14px 30px -26px rgba(95,80,55,.5); }
.ms-feat-ic {
    width: 50px; height: 50px; border-radius: 50% 50% 50% 16px; font-size: 1.55rem;
    display: flex; align-items: center; justify-content: center; margin-bottom: 14px;
}
.ms-feat h4 { font-family: var(--serif); color: #2E2A24; font-size: 1.1rem; font-weight: 600; margin: 0 0 5px; }
.ms-feat p  { color: #6B6258; font-size: .86rem; line-height: 1.55; margin: 0; }

/* how it works */
.ms-steplist { display: grid; grid-template-columns: repeat(4, 1fr); gap: 13px; }
.ms-howstep { background: #FFFFFF; border: 1px solid #ECE4D8; border-radius: 18px; padding: 18px; }
.ms-howstep .b {
    width: 30px; height: 30px; border-radius: 50% 50% 50% 10px; background: #EEF3E9; color: #5F7A57;
    font-family: var(--serif); font-weight: 700; font-size: 1.05rem; display: flex; align-items: center; justify-content: center; margin-bottom: 11px;
}
.ms-howstep h5 { font-family: var(--serif); color: #2E2A24; font-size: .98rem; font-weight: 600; margin: 0 0 3px; }
.ms-howstep p  { color: #6B6258; font-size: .8rem; line-height: 1.5; margin: 0; }

/* risk banner */
.ms-risk { border-radius: 24px; padding: 30px 32px; display: flex; align-items: center; gap: 22px; }
.ms-risk-badge {
    width: 66px; height: 66px; border-radius: 50% 50% 50% 18px; flex: none; font-size: 2.1rem;
    display: flex; align-items: center; justify-content: center; background: rgba(255,255,255,.55);
}
.ms-risk-kicker { font-size: .7rem; font-weight: 800; letter-spacing: .14em; opacity: .85; text-transform: uppercase; }
.ms-risk h2 { font-family: var(--serif); font-size: 1.6rem; font-weight: 600; margin: 5px 0; letter-spacing: -.01em; }
.ms-risk p  { font-size: .92rem; line-height: 1.55; margin: 0; opacity: .92; }

/* recommendation cards */
.ms-rec-head { display: flex; align-items: center; gap: 12px; margin-bottom: 11px; }
.ms-rec-ic {
    width: 38px; height: 38px; border-radius: 50% 50% 50% 12px; flex: none; font-size: 1.15rem;
    background: #EEF3E9; color: #5F7A57; display: flex; align-items: center; justify-content: center;
}
.ms-rec-tt { font-family: var(--serif); font-size: 1.08rem; font-weight: 600; color: #2E2A24; }
.ms-tip { display: flex; align-items: flex-start; gap: 10px; font-size: .88rem; color: #6B6258; line-height: 1.55; padding: 5px 0; }
.ms-tip i { color: #7C9473; font-size: 1.1rem; margin-top: 1px; flex: none; }

/* disclaimer + footer */
.ms-disc {
    background: #FBF0E6; border: 1px solid #EAD7C2; border-radius: 18px;
    padding: 16px 20px; color: #8A4B22; font-size: .84rem; line-height: 1.55;
    display: flex; gap: 11px; align-items: flex-start;
}
.ms-disc i { font-size: 1.25rem; margin-top: 1px; flex: none; }
.ms-foot {
    background: #2E2A24; border-radius: 26px; padding: 32px 34px; margin-top: 24px; color: #D8CFC2;
}
.ms-foot .ttl { font-family: var(--serif); color: #FBF7F0; font-weight: 600; font-size: 1.25rem; display: flex; align-items: center; gap: 11px; }
.ms-foot p { font-size: .84rem; line-height: 1.6; color: #A89F92; margin: 10px 0 0; }
.ms-foot a { color: #C7D6BD; text-decoration: none; font-weight: 700; }
.ms-foot a:hover { color: #E3EBD9; }
.ms-foot hr { border: none; border-top: 1px solid #46403A; margin: 18px 0; }
.ms-foot .copy { font-size: .76rem; color: #7A7066; }

/* ---------- Streamlit widget polish ---------- */
div.stButton > button {
    border-radius: 14px; font-weight: 700; border: 1.5px solid #E2D9CB;
    background: #FFFFFF; color: #5F574C; transition: all .15s ease; padding: .6rem 1.1rem;
}
div.stButton > button:hover { border-color: #C7D6BD; color: #5F7A57; background: #FAFBF8; }
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7C9473, #5F7A57); color: #FFFFFF; border: none;
    font-weight: 800; box-shadow: 0 12px 26px -12px rgba(95,122,87,.85);
}
div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #6B8463, #557049);
    box-shadow: 0 16px 32px -12px rgba(95,122,87,1); transform: translateY(-1px);
}

/* inputs */
.stSelectbox label, .stNumberInput label, .stSlider label, .stRadio > label {
    font-weight: 700 !important; color: #5F574C !important; font-size: .87rem !important;
}
[data-baseweb="select"] > div, .stNumberInput input {
    border-radius: 12px !important; border-color: #E2D9CB !important; background: #FFFFFF !important;
}
[data-testid="stWidgetLabel"] p { font-weight: 700; color: #5F574C; }

/* radio group as graded chips: calm -> intense (sage -> sand -> tan -> clay) */
div[role="radiogroup"] { gap: 8px; flex-wrap: wrap; }
div[role="radiogroup"] input[type="radio"] { accent-color: #5F7A57; }
div[role="radiogroup"] label {
    border: 1.5px solid #E8DECF; border-radius: 999px;
    padding: 9px 16px; margin: 0 !important; font-weight: 700; color: #5F574C;
    transition: all .14s ease;
}
div[role="radiogroup"] label:nth-child(1) { background: #EFF4EA; border-color: #D7E3CE; }
div[role="radiogroup"] label:nth-child(2) { background: #F5F1E5; border-color: #E8DEC8; }
div[role="radiogroup"] label:nth-child(3) { background: #F7EAD8; border-color: #ECD5B6; }
div[role="radiogroup"] label:nth-child(4) { background: #F7E1D3; border-color: #EAC6AC; }
div[role="radiogroup"] label:hover { transform: translateY(-1px); border-color: #C7B79F; }
div[role="radiogroup"] label:has(input:checked) {
    color: #2E2A24; font-weight: 800; transform: translateY(-1px);
    box-shadow: inset 0 0 0 2px rgba(46,42,36,.28), 0 10px 20px -10px rgba(95,80,55,.55);
}

/* metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #ECE4D8; border-radius: 20px;
    padding: 20px 22px; box-shadow: 0 14px 30px -26px rgba(95,80,55,.5);
}
[data-testid="stMetricLabel"] p { color: #A89F92; font-weight: 700; font-size: .78rem; letter-spacing: .03em; }
[data-testid="stMetricValue"] { font-family: var(--serif); color: #2E2A24; font-weight: 600; }

/* bordered containers (wizard / cards) */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF; border-radius: 24px; border-color: #ECE4D8 !important;
    box-shadow: 0 18px 40px -30px rgba(95,80,55,.55);
}
[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 4px; }

/* alerts */
[data-testid="stAlert"] { border-radius: 14px; }

/* question label */
.ms-q { font-weight: 700; color: #3D372F; font-size: .93rem; margin: 4px 0 4px; }
.ms-q .qn { font-family: var(--serif); color: #5F7A57; font-weight: 700; margin-right: 7px; }

/* running-score card (PHQ / GAD) */
.ms-score {
    display: flex; align-items: center; gap: 16px;
    background: linear-gradient(135deg, #F3F7EF, #EEF3E9);
    border: 1px solid #DCE7D4; border-radius: 18px; padding: 16px 20px;
}
.ms-score-ic {
    width: 44px; height: 44px; border-radius: 50% 50% 50% 14px; flex: none; font-size: 1.3rem;
    background: #FFFFFF; color: #5F7A57; display: flex; align-items: center; justify-content: center;
    box-shadow: 0 6px 14px -8px rgba(95,122,87,.6);
}
.ms-score-lbl { font-size: .72rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; color: #7C9473; }
.ms-score-val { font-family: var(--serif); font-size: 1.45rem; font-weight: 600; color: #2E2A24; line-height: 1.1; }
.ms-score-val small { font-size: .9rem; color: #A89F92; font-weight: 600; }
.ms-score-band {
    margin-left: auto; font-size: .82rem; font-weight: 800; color: #2E2A24;
    background: #FFFFFF; border: 1px solid #DCE7D4; border-radius: 999px; padding: 7px 15px; white-space: nowrap;
}

/* ------------------------- Mobile ------------------------- */
@media (max-width: 640px) {
    .block-container { padding-top: 0.85rem !important; padding-bottom: 2rem !important; }

    /* brand header: hide the decorative chip, tighten */
    .ms-nav { padding: 13px 16px; border-radius: 16px; }
    .ms-brand-name { font-size: 1.2rem; }
    .ms-nav-chip { display: none; }

    /* hero scales down */
    .ms-hero { padding: 30px 22px; border-radius: 24px; }
    .ms-hero h1 { font-size: 1.95rem; }
    .ms-hero p  { font-size: .96rem; }

    /* multi-column grids collapse to one column */
    .ms-grid, .ms-steplist { grid-template-columns: 1fr; }

    /* section headers */
    .ms-sec-tt { font-size: 1.18rem; }
    .ms-sec-ic { width: 38px; height: 38px; font-size: 1.1rem; }

    /* comfortable inner card padding on narrow screens */
    [data-testid="stVerticalBlockBorderWrapper"] > div { padding: 1rem; }

    /* tighten stacked column gaps (top nav, inputs, metrics) */
    [data-testid="stHorizontalBlock"] { gap: 0.5rem; }

    /* answer chips: clean two-per-row, equal width, single line */
    div[role="radiogroup"] label { flex: 1 1 42%; min-width: 88px; padding: 8px 10px; font-size: 0.82rem; }
    div[role="radiogroup"] label p { font-size: 0.82rem; margin: 0; }

    /* running-score card wraps gracefully (long severity bands) */
    .ms-score { padding: 14px 16px; gap: 12px; flex-wrap: wrap; }

    /* risk banner: tighter on small screens */
    .ms-risk { padding: 22px 20px; gap: 16px; border-radius: 20px; }
    .ms-risk-badge { width: 54px; height: 54px; font-size: 1.7rem; }
    .ms-risk h2 { font-size: 1.35rem; }
    .ms-risk-kicker { font-size: .64rem; }

    /* recommendations + footer */
    .ms-tip { font-size: .85rem; }
    .ms-foot { padding: 24px 22px; border-radius: 20px; }
    .ms-foot .ttl { font-size: 1.1rem; }
}
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
        'title': 'Low Risk — Keep It Up',
        'description': 'Your responses suggest good mental wellbeing. Maintain your current healthy habits.',
        'solutions': [
            {'icon': 'barbell', 'category': 'Physical Activity', 'tips': [
                'Continue regular exercise (30 mins daily)',
                'Explore new physical activities you enjoy',
                'Consider outdoor activities for vitamin D exposure']},
            {'icon': 'moon', 'category': 'Sleep & Rest', 'tips': [
                'Maintain consistent sleep schedule (7-8 hours)',
                'Create a relaxing bedtime routine',
                'Avoid screens 1 hour before bed']},
            {'icon': 'flower-lotus', 'category': 'Mental Wellness', 'tips': [
                'Practice mindfulness or meditation (10 mins daily)',
                'Maintain social connections',
                'Engage in hobbies and creative activities']},
            {'icon': 'leaf', 'category': 'Lifestyle', 'tips': [
                'Eat balanced, nutritious meals',
                'Limit caffeine and sugar intake',
                'Spend time in nature regularly']},
        ]
    },
    'Moderate': {
        'title': 'Moderate Risk — Take Action',
        'description': 'You are experiencing moderate stress/depression. Consider lifestyle changes and support.',
        'solutions': [
            {'icon': 'barbell', 'category': 'Physical Activity', 'tips': [
                'Increase exercise to 45-60 minutes daily',
                'Try activities like yoga, walking, or swimming',
                'Join fitness classes or sports groups for social support']},
            {'icon': 'moon', 'category': 'Sleep & Rest', 'tips': [
                'Prioritize 7-9 hours of sleep nightly',
                'Develop relaxation techniques (breathing, progressive muscle relaxation)',
                'Keep bedroom cool, dark, and quiet']},
            {'icon': 'users-three', 'category': 'Social Support', 'tips': [
                'Spend quality time with family and friends',
                'Join community groups or clubs',
                'Consider peer support groups',
                'Reach out to trusted people about your feelings']},
            {'icon': 'brain', 'category': 'Stress Management', 'tips': [
                'Practice daily meditation (15-20 mins)',
                'Try journaling to process emotions',
                'Use stress-relief apps (Calm, Headspace, Insight Timer)',
                'Identify and reduce stressors where possible']},
            {'icon': 'stethoscope', 'category': 'Professional Help', 'tips': [
                'Consider speaking with a counselor or therapist',
                'Consult your doctor about mental health screening',
                'Explore cognitive behavioral therapy (CBT) resources']},
        ]
    },
    'High': {
        'title': 'High Risk — Seek Professional Help',
        'description': 'You are experiencing significant stress/depression. Professional support is strongly recommended.',
        'solutions': [
            {'icon': 'siren', 'category': 'Immediate Actions', 'tips': [
                'URGENT: Contact a mental health professional or counselor immediately',
                'Call your doctor and schedule an appointment',
                'Consider visiting a mental health clinic or hospital',
                'If in crisis, call emergency services or crisis helpline']},
            {'icon': 'hospital', 'category': 'Professional Treatment', 'tips': [
                'Consult a psychiatrist for comprehensive evaluation',
                'Consider therapy (CBT, DBT, or psychotherapy)',
                'Discuss medication options with doctor if appropriate',
                'Explore inpatient or outpatient treatment programs']},
            {'icon': 'lifebuoy', 'category': 'Crisis Support', 'tips': [
                'National Suicide Prevention Lifeline: 1-800-273-8255 (US)',
                'Crisis Text Line: Text HOME to 741741',
                'International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/',
                'AAMI (Befrienders India): 9152987821 (India)']},
            {'icon': 'sun', 'category': 'Daily Wellness (Supplementary)', 'tips': [
                'Engage in light physical activity (short walks)',
                'Maintain regular sleep schedule',
                'Eat nutritious meals at regular times',
                'Avoid alcohol and drugs',
                'Connect with supportive people']},
            {'icon': 'heart', 'category': 'Self-Care', 'tips': [
                'Create a safety plan with professional guidance',
                'Keep emergency contact numbers accessible',
                'Maintain daily routine and structure',
                'Track mood and symptoms in a journal']},
        ]
    }
}


# ==================== CLINICAL THRESHOLDS ====================
PHQ9_THRESHOLDS = {'Minimal': (0, 4), 'Mild': (5, 9), 'Moderate': (10, 14), 'Moderately Severe': (15, 19), 'Severe': (20, 27)}
GAD7_THRESHOLDS = {'Minimal': (0, 4), 'Mild': (5, 9), 'Moderate': (10, 14), 'Severe': (15, 21)}


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

PHQ_OPTIONS = {0: "Not really", 1: "Here & there", 2: "Quite a lot", 3: "Nearly always"}


# ==================== HELPER FUNCTIONS ====================
def get_depression_level(score):
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
    if score <= 4:
        return 'Minimal'
    elif score <= 9:
        return 'Mild'
    elif score <= 14:
        return 'Moderate'
    else:
        return 'Severe'


def get_risk_level(dep_score, anx_score):
    risk = (dep_score + anx_score) / 2
    if risk < 5:
        return 'Low'
    elif risk < 12:
        return 'Moderate'
    else:
        return 'High'


def _safe_transform(encoder, value):
    """Encode a single label, falling back to a known class if unseen.

    The saved encoders were trained on a subset of the UI options (e.g. gender
    only saw Male/Female). This keeps the inclusive dropdowns from crashing
    prediction when a not-seen-in-training option is chosen.
    """
    classes = list(encoder.classes_)
    if value in classes:
        return int(encoder.transform([value])[0])
    return int(encoder.transform([classes[0]])[0])


def encode_features(gender, stress, activity, illness, history, treatment, work, encoders):
    try:
        return [
            _safe_transform(encoders['gender'], gender),
            _safe_transform(encoders['stress'], stress),
            _safe_transform(encoders['activity'], activity),
            _safe_transform(encoders['illness'], illness),
            _safe_transform(encoders['history'], history),
            _safe_transform(encoders['treatment'], treatment),
            _safe_transform(encoders['work'], work),
        ]
    except Exception as e:
        st.error(f"Encoding error: {e}")
        return None


def predict_risk(age, gender, dep_score, anx_score, stress, sleep, activity,
                 illness, history, treatment, treatment_days, work, model, encoders, feature_cols):
    try:
        encoded = encode_features(gender, stress, activity, illness, history, treatment, work, encoders)
        if encoded is None:
            return None

        features = np.array([[
            age, encoded[0], dep_score, anx_score, encoded[1], sleep,
            encoded[2], encoded[3], encoded[4], encoded[5], treatment_days, encoded[6]
        ]])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        return prediction, probability
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# ==================== WIZARD STATE ====================
def fresh_answers():
    return {
        'age': 25, 'gender': 'Male', 'work': 'Student',
        'phq9': [0] * 9, 'gad7': [0] * 7,
        'sleep': 7.0, 'activity': 'Moderate', 'stress': 'Low',
        'chronic': 'No', 'history': 'No', 'treatment': 'None', 'tdays': 0,
    }


def reset_assessment():
    for k in list(st.session_state.keys()):
        if k.startswith('wz_'):
            del st.session_state[k]
    st.session_state.ans = fresh_answers()
    st.session_state.wiz_step = 1
    st.session_state.test_complete = False


def _seed(key, default):
    if key not in st.session_state:
        st.session_state[key] = default


# ==================== UI BUILDING BLOCKS ====================
def render_header():
    st.markdown(f"""
    <div class="ms-nav">
      <div class="ms-brand">
        <div class="ms-logo">{icon("leaf", fill=True)}</div>
        <div>
          <div class="ms-brand-name">MindScope</div>
          <div class="ms-brand-tag">Mental Wellbeing</div>
        </div>
      </div>
      <div class="ms-nav-chip">{icon("heartbeat")} PHQ-9 &nbsp;·&nbsp; GAD-7 &nbsp;·&nbsp; AI</div>
    </div>
    """, unsafe_allow_html=True)


def render_nav(active_page):
    c1, c2, c3, _ = st.columns([1.1, 1.4, 1, 4])
    with c1:
        if st.button("Home", use_container_width=True,
                     type="primary" if active_page == "home" else "secondary"):
            st.session_state.page = "home"
            st.rerun()
    with c2:
        if st.button("Assessment", use_container_width=True,
                     type="primary" if active_page == "screening" else "secondary"):
            st.session_state.page = "screening"
            st.rerun()
    with c3:
        if st.button("About", use_container_width=True,
                     type="primary" if active_page == "about" else "secondary"):
            st.session_state.page = "about"
            st.rerun()


def section_header(icon_name, title, subtitle=""):
    sub = f'<div class="ms-sec-sb">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="ms-sec">
      <div class="ms-sec-ic">{icon(icon_name)}</div>
      <div><div class="ms-sec-tt">{title}</div>{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def disclaimer(html):
    st.markdown(f'<div class="ms-disc">{icon("warning-circle", fill=True)}<div>{html}</div></div>',
                unsafe_allow_html=True)


def render_score_card(label, score, out_of, band, glyph):
    st.markdown(f"""
    <div class="ms-score">
      <div class="ms-score-ic">{icon(glyph, fill=True)}</div>
      <div>
        <div class="ms-score-lbl">{label}</div>
        <div class="ms-score-val">{score}<small> / {out_of}</small></div>
      </div>
      <div class="ms-score-band">{band}</div>
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    st.markdown(f"""
    <div class="ms-foot">
      <div class="ttl">{icon("leaf", fill=True)} MindScope-2025</div>
      <p>An AI-driven mental wellbeing assessment that helps you understand your emotional health using
         clinically validated screening — PHQ-9 for depression and GAD-7 for anxiety — combined with
         machine learning for personalized insight. Educational use only; not a medical diagnosis.</p>
      <hr>
      <p style="margin:0;">
        <a href="https://haroontrailblazer.vercel.app" target="_blank">Developer</a> &nbsp;·&nbsp;
        <a href="https://github.com/haroontrailblazer" target="_blank">GitHub</a> &nbsp;·&nbsp;
        <a href="https://www.instagram.com/hendrix__trailblazer?igsh=MTEyOTEycm9mMGxjaA==" target="_blank">Instagram</a>
      </p>
      <p class="copy" style="margin-top:14px;">© 2025 MindScope · Built by Haroon K M</p>
    </div>
    """, unsafe_allow_html=True)


def render_risk_banner(risk_level, confidence_pct):
    t = RISK_THEME[risk_level]
    data = SOLUTIONS_DB[risk_level]
    st.markdown(f"""
    <div class="ms-risk" style="background:{t['bg']}; border:1px solid {t['ring']}33;">
      <div class="ms-risk-badge" style="color:{t['ring']};">{icon(t['glyph'], fill=True)}</div>
      <div style="color:{t['ink']};">
        <div class="ms-risk-kicker">Assessed risk · {confidence_pct:.0f}% model confidence</div>
        <h2>{data['title']}</h2>
        <p>{data['description']}</p>
      </div>
    </div>
    """, unsafe_allow_html=True)


def display_solutions(risk_level):
    section_header("list-checks", "Recommended Actions",
                   "Personalized guidance based on your assessed risk level.")
    for solution in SOLUTIONS_DB[risk_level]['solutions']:
        with st.container(border=True):
            st.markdown(
                f'<div class="ms-rec-head"><div class="ms-rec-ic">{icon(solution["icon"])}</div>'
                f'<div class="ms-rec-tt">{solution["category"]}</div></div>',
                unsafe_allow_html=True,
            )
            tips_html = "".join(
                f'<div class="ms-tip">{icon("check-circle", fill=True)}<span>{tip}</span></div>'
                for tip in solution['tips']
            )
            st.markdown(tips_html, unsafe_allow_html=True)


def display_assessment_results(phq9_score, gad7_score, risk_level, model_confidence):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PHQ-9 · Depression", f"{phq9_score}/27",
                  f"{get_depression_level(phq9_score)}", delta_color="inverse")
    with col2:
        st.metric("GAD-7 · Anxiety", f"{gad7_score}/21",
                  f"{get_anxiety_level(gad7_score)}", delta_color="inverse")
    with col3:
        confidence_pct = max(model_confidence) * 100
        st.metric("Overall Risk", f"{risk_level}", f"Confidence {confidence_pct:.1f}%")

    col_chart1, col_chart2 = st.columns(2)

    def _style(fig):
        fig.update_layout(
            height=300, margin=dict(l=20, r=20, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Nunito Sans, sans-serif", color=INK),
        )
        return fig

    with col_chart1:
        fig_phq = go.Figure(go.Indicator(
            mode="gauge+number", value=phq9_score,
            title={"text": "Depression (PHQ-9)", "font": {"size": 15, "color": BODY}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 27], 'tickcolor': MUTED},
                'bar': {'color': SAGE}, 'borderwidth': 0,
                'steps': [
                    {'range': [0, 4], 'color': "#E3EBD9"},
                    {'range': [5, 9], 'color': "#F1E7D3"},
                    {'range': [10, 14], 'color': "#F0D8BD"},
                    {'range': [15, 19], 'color': "#E9C2A4"},
                    {'range': [20, 27], 'color': "#DFA98C"}
                ],
                'threshold': {'line': {'color': CLAY, 'width': 3}, 'thickness': 0.75, 'value': 20}
            }
        ))
        st.plotly_chart(_style(fig_phq), use_container_width=True)

    with col_chart2:
        fig_gad = go.Figure(go.Indicator(
            mode="gauge+number", value=gad7_score,
            title={"text": "Anxiety (GAD-7)", "font": {"size": 15, "color": BODY}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 21], 'tickcolor': MUTED},
                'bar': {'color': CLAY}, 'borderwidth': 0,
                'steps': [
                    {'range': [0, 4], 'color': "#E3EBD9"},
                    {'range': [5, 9], 'color': "#F1E7D3"},
                    {'range': [10, 14], 'color': "#F0D8BD"},
                    {'range': [15, 21], 'color': "#DFA98C"}
                ],
                'threshold': {'line': {'color': CLAY, 'width': 3}, 'thickness': 0.75, 'value': 15}
            }
        ))
        st.plotly_chart(_style(fig_gad), use_container_width=True)


# ==================== PAGES ====================
def page_home():
    st.markdown(f"""
    <div class="ms-hero">
      <span class="ms-hero-badge">{icon("sparkle", fill=True)} A CALMER WAY TO CHECK IN</span>
      <h1>Understand your mind,<br>gently and clearly</h1>
      <p>MindScope pairs the clinically validated PHQ-9 and GAD-7 screenings with a machine
         learning model to offer a confidential, evidence-based reflection on your wellbeing —
         in just a few quiet minutes.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, _ = st.columns([1.5, 1, 1.2])
    with c1:
        if st.button("Begin Assessment", type="primary", use_container_width=True):
            reset_assessment()
            st.session_state.page = "screening"
            st.rerun()
    with c2:
        if st.button("Learn More", use_container_width=True):
            st.session_state.page = "about"
            st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ms-grid">
      <div class="ms-feat">
        <div class="ms-feat-ic" style="background:#EEF3E9;color:#5F7A57;">{icon("stethoscope")}</div>
        <h4>Clinically Grounded</h4>
        <p>Built on PHQ-9 and GAD-7 — the same standardized tools trusted in clinical and research settings.</p>
      </div>
      <div class="ms-feat">
        <div class="ms-feat-ic" style="background:#FBF0E6;color:#C2703D;">{icon("cpu")}</div>
        <h4>Thoughtful AI</h4>
        <p>A trained model weighs your scores alongside lifestyle factors to estimate your overall risk.</p>
      </div>
      <div class="ms-feat">
        <div class="ms-feat-ic" style="background:#EEF3E9;color:#5F7A57;">{icon("hand-heart")}</div>
        <h4>Private &amp; Caring</h4>
        <p>Your answers are processed in-session for your result. Nothing is stored or shared.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    section_header("path", "How It Works", "From a few gentle questions to personalized guidance.")
    st.markdown("""
    <div class="ms-steplist">
      <div class="ms-howstep"><div class="b">1</div><h5>Answer</h5><p>Move through 16 short, validated questions about how you've felt recently.</p></div>
      <div class="ms-howstep"><div class="b">2</div><h5>Reflect</h5><p>Share a little about your sleep, activity and stress.</p></div>
      <div class="ms-howstep"><div class="b">3</div><h5>Analyze</h5><p>The model reviews your responses and finds your risk profile.</p></div>
      <div class="ms-howstep"><div class="b">4</div><h5>Guidance</h5><p>Receive a clear result with caring, actionable next steps.</p></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    disclaimer("<b>A gentle note:</b> MindScope is an educational self-assessment, not a substitute for "
               "professional diagnosis or treatment. If you are in crisis, please contact your local emergency services.")

    render_footer()


# ---------- Wizard steps ----------
def _step_basics(ans):
    section_header("user", "A Few Basics", "Tell us a little about yourself.")
    col1, col2, col3 = st.columns(3)
    with col1:
        _seed('wz_age', ans['age'])
        ans['age'] = st.number_input("Age", min_value=13, max_value=100, step=1, key='wz_age')
    with col2:
        _seed('wz_gender', ans['gender'])
        ans['gender'] = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"], key='wz_gender')
    with col3:
        _seed('wz_work', ans['work'])
        ans['work'] = st.selectbox("Work Status", ["Student", "Employed", "Unemployed", "Retired", "Other"], key='wz_work')


def _step_phq(ans):
    section_header("cloud-rain", "Depression · PHQ-9",
                   "Over the last 2 weeks, how much has each of these felt like you?")
    for i, question in enumerate(PHQ9_QUESTIONS):
        st.markdown(f'<div class="ms-q"><span class="qn">{i+1}.</span>{question}</div>', unsafe_allow_html=True)
        _seed(f'wz_phq9_{i}', ans['phq9'][i])
        ans['phq9'][i] = st.radio(
            label=f"phq9_{i}", options=list(PHQ_OPTIONS.keys()),
            format_func=lambda x: PHQ_OPTIONS[x],
            key=f'wz_phq9_{i}', horizontal=True, label_visibility="collapsed",
        )
        if i < len(PHQ9_QUESTIONS) - 1:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    score = sum(ans['phq9'])
    render_score_card("Your PHQ-9 so far", score, 27, get_depression_level(score), "cloud-rain")


def _step_gad(ans):
    section_header("wind", "Anxiety · GAD-7",
                   "Over the last 2 weeks, how much has each of these felt like you?")
    for i, question in enumerate(GAD7_QUESTIONS):
        st.markdown(f'<div class="ms-q"><span class="qn">{i+1}.</span>{question}</div>', unsafe_allow_html=True)
        _seed(f'wz_gad7_{i}', ans['gad7'][i])
        ans['gad7'][i] = st.radio(
            label=f"gad7_{i}", options=list(PHQ_OPTIONS.keys()),
            format_func=lambda x: PHQ_OPTIONS[x],
            key=f'wz_gad7_{i}', horizontal=True, label_visibility="collapsed",
        )
        if i < len(GAD7_QUESTIONS) - 1:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    score = sum(ans['gad7'])
    render_score_card("Your GAD-7 so far", score, 21, get_anxiety_level(score), "wind")


def _step_lifestyle(ans):
    section_header("heartbeat", "Lifestyle & Health", "A few factors that shape wellbeing.")
    col1, col2, col3 = st.columns(3)
    with col1:
        _seed('wz_sleep', ans['sleep'])
        ans['sleep'] = st.slider("Average sleep per night (hrs)", 0.0, 12.0, step=0.5, key='wz_sleep')
    with col2:
        _seed('wz_activity', ans['activity'])
        ans['activity'] = st.selectbox("Physical activity level", ["Low", "Moderate", "High"], key='wz_activity')
    with col3:
        _seed('wz_stress', ans['stress'])
        ans['stress'] = st.selectbox("Current stress level", ["Low", "High", "Severe"], key='wz_stress')

    col4, col5 = st.columns(2)
    with col4:
        _seed('wz_chronic', ans['chronic'])
        ans['chronic'] = st.radio("Any chronic illness?", ["No", "Yes"], horizontal=True, key='wz_chronic')
    with col5:
        _seed('wz_history', ans['history'])
        ans['history'] = st.radio("History of mental health issues?", ["No", "Yes"], horizontal=True, key='wz_history')

    col6, col7 = st.columns(2)
    with col6:
        _seed('wz_treatment', ans['treatment'])
        ans['treatment'] = st.selectbox("Current treatment", ["None", "Medication", "Therapy", "Both"], key='wz_treatment')
    with col7:
        _seed('wz_tdays', ans['tdays'])
        ans['tdays'] = st.slider("Days in current treatment", 0, 365, step=1, key='wz_tdays')


WIZARD_STEPS = [
    ("Basics", _step_basics),
    ("Depression", _step_phq),
    ("Anxiety", _step_gad),
    ("Lifestyle", _step_lifestyle),
]


def page_screening(model, encoders, feature_cols):
    if 'ans' not in st.session_state:
        st.session_state.ans = fresh_answers()
    if 'wiz_step' not in st.session_state:
        st.session_state.wiz_step = 1

    # Re-commit off-screen widget values so Streamlit doesn't garbage-collect
    # the answers from steps that aren't currently rendered. (Runs before any
    # widget is created this rerun, so it acts purely as a seed.)
    for k in list(st.session_state.keys()):
        if k.startswith('wz_'):
            st.session_state[k] = st.session_state[k]

    ans = st.session_state.ans
    step = st.session_state.wiz_step
    total = len(WIZARD_STEPS)
    label, render_step = WIZARD_STEPS[step - 1]

    # progress
    segs = "".join(
        f'<div class="wz-seg {"done" if i < step else ("active" if i == step else "")}"></div>'
        for i in range(1, total + 1)
    )
    st.markdown(f'<div class="wz-prog">{segs}</div>'
                f'<div class="wz-meta">Step {step} of {total} · {label}</div>', unsafe_allow_html=True)

    with st.container(border=True):
        render_step(ans)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # navigation
    c1, c2 = st.columns(2)
    with c1:
        if step > 1:
            if st.button("Back", use_container_width=True):
                st.session_state.wiz_step = step - 1
                st.rerun()
        else:
            if st.button("Cancel", use_container_width=True):
                st.session_state.page = "home"
                st.rerun()
    with c2:
        if step < total:
            if st.button("Continue", type="primary", use_container_width=True):
                st.session_state.wiz_step = step + 1
                st.rerun()
        else:
            if st.button("Analyze & Get Results", type="primary", use_container_width=True):
                with st.spinner("Reflecting on your responses..."):
                    phq9_score = sum(ans['phq9'])
                    gad7_score = sum(ans['gad7'])
                    prediction = predict_risk(
                        ans['age'], ans['gender'], phq9_score, gad7_score, ans['stress'],
                        ans['sleep'], ans['activity'], ans['chronic'], ans['history'],
                        ans['treatment'], ans['tdays'], ans['work'], model, encoders, feature_cols
                    )
                    if prediction:
                        risk_level, probabilities = prediction
                        st.session_state.test_complete = True
                        st.session_state.phq9_score = phq9_score
                        st.session_state.gad7_score = gad7_score
                        st.session_state.risk_level = risk_level
                        st.session_state.probabilities = probabilities
                        st.session_state.page = "results"
                        st.rerun()


def page_results():
    if not st.session_state.get('test_complete', False):
        st.warning("Please complete the assessment first.")
        if st.button("Begin Assessment", type="primary"):
            reset_assessment()
            st.session_state.page = "screening"
            st.rerun()
        return

    section_header("chart-line-up", "Your Results",
                   "A summary of your screening scores with personalized recommendations.")

    confidence_pct = max(st.session_state.probabilities) * 100
    render_risk_banner(st.session_state.risk_level, confidence_pct)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    display_assessment_results(
        st.session_state.phq9_score, st.session_state.gad7_score,
        st.session_state.risk_level, st.session_state.probabilities,
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    display_solutions(st.session_state.risk_level)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    disclaimer("<b>Reminder:</b> These results are for educational self-awareness only and do not constitute "
               "a medical diagnosis. Please consult a qualified professional for clinical evaluation.")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("Take Assessment Again", use_container_width=True):
        reset_assessment()
        st.session_state.page = "screening"
        st.rerun()

    render_footer()


def page_about():
    section_header("info", "About MindScope-2025",
                   "An educational, evidence-based mental wellbeing tool.")

    with st.container(border=True):
        st.markdown("""
**MindScope-2025** helps individuals reflect on their mental health using evidence-based screening
combined with machine learning. It is designed for awareness and education — not clinical diagnosis.

**Technology Stack**
- **Framework:** Streamlit
- **ML Models:** Scikit-learn (Random Forest, Logistic Regression)
- **Data:** Global Mental Health Dataset 2025 (synthetic)
- **Assessment Tools:** PHQ-9 & GAD-7
        """)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("""
**PHQ-9 — Patient Health Questionnaire**
- 9-item screening tool for depression
- Score range: 0–27
- Based on DSM-IV depression criteria
- Widely used in clinical & research settings
            """)
    with col2:
        with st.container(border=True):
            st.markdown("""
**GAD-7 — Generalized Anxiety Disorder**
- 7-item screening tool for anxiety
- Score range: 0–21
- Established clinical diagnostic validity
- Recommended by the WHO
            """)

    disclaimer("<b>Disclaimer:</b> This application is for educational and self-assessment purposes only and "
               "should not be used for professional medical diagnosis or treatment. If you are experiencing a "
               "mental health crisis, please contact: <b>Emergency Services</b> 911 (US) / 112 (India) · "
               "<b>Suicide Prevention</b> 1-800-273-8255 (US) · <b>Crisis Text</b> HOME to 741741.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    render_footer()


# ==================== MAIN APP ====================
def main():
    model, encoders, feature_cols = load_models()

    if "page" not in st.session_state:
        st.session_state.page = "home"
    page = st.session_state.page

    render_header()
    render_nav(page)

    if page == "home":
        page_home()
    elif page == "screening":
        page_screening(model, encoders, feature_cols)
    elif page == "results":
        page_results()
    elif page == "about":
        page_about()


if __name__ == "__main__":
    main()
