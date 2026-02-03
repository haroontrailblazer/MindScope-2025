# Model Training Script for MindScope-2025
# Trains ML models to predict depression and anxiety levels

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA LOADING & CLEANING ====================
print("Loading and cleaning data...")
df = pd.read_csv(r'Data/Global_Mental_Health_Dataset_2025.csv')

# Derive depression and anxiety levels based on clinical thresholds
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

# Create target columns
df['Depression_Level'] = df['Depression_Score'].apply(get_depression_level)
df['Anxiety_Level'] = df['Anxiety_Score'].apply(get_anxiety_level)

# Create combined risk level
def get_risk_level(dep_score, anx_score):
    risk = (dep_score + anx_score) / 2
    if risk < 5:
        return 'Low'
    elif risk < 12:
        return 'Moderate'
    else:
        return 'High'

df['Risk_Level'] = df.apply(lambda row: get_risk_level(row['Depression_Score'], row['Anxiety_Score']), axis=1)

# Handle missing values
df['Sleep_Hours'].fillna(df['Sleep_Hours'].median(), inplace=True)
df['Physical_Activity'].fillna(df['Physical_Activity'].mode()[0], inplace=True)
df['Treatment'].fillna('None', inplace=True)

# Encode categorical variables
le_gender = LabelEncoder()
le_stress = LabelEncoder()
le_activity = LabelEncoder()
le_illness = LabelEncoder()
le_history = LabelEncoder()
le_treatment = LabelEncoder()
le_work = LabelEncoder()

df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])
df['Stress_Level_Encoded'] = le_stress.fit_transform(df['Stress_Level'])
df['Physical_Activity_Encoded'] = le_activity.fit_transform(df['Physical_Activity'])
df['Chronic_Illness_Encoded'] = le_illness.fit_transform(df['Chronic_Illness'])
df['Mental_Health_History_Encoded'] = le_history.fit_transform(df['Mental_Health_History'])
df['Treatment_Encoded'] = le_treatment.fit_transform(df['Treatment'])
df['Work_Status_Encoded'] = le_work.fit_transform(df['Work_Status'])

# ==================== FEATURE SELECTION ====================
feature_cols = [
    'Age', 
    'Gender_Encoded',
    'Depression_Score',
    'Anxiety_Score',
    'Stress_Level_Encoded',
    'Sleep_Hours',
    'Physical_Activity_Encoded',
    'Chronic_Illness_Encoded',
    'Mental_Health_History_Encoded',
    'Treatment_Encoded',
    'Days_of_Treatment',
    'Work_Status_Encoded'
]

X = df[feature_cols]
y_depression = df['Depression_Level']
y_anxiety = df['Anxiety_Level']
y_risk = df['Risk_Level']

# ==================== TRAIN TEST SPLIT ====================
X_train, X_test, y_risk_train, y_risk_test = train_test_split(
    X, y_risk, test_size=0.2, random_state=42, stratify=y_risk
)

# ==================== MODEL TRAINING ====================
print("\nTraining models for Risk Level prediction...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train
    model.fit(X_train, y_risk_train)
    trained_models[name] = model
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    train_acc = accuracy_score(y_risk_train, y_pred_train)
    test_acc = accuracy_score(y_risk_test, y_pred_test)
    precision = precision_score(y_risk_test, y_pred_test, average='weighted', zero_division=0)
    recall = recall_score(y_risk_test, y_pred_test, average='weighted', zero_division=0)
    f1 = f1_score(y_risk_test, y_pred_test, average='weighted', zero_division=0)
    
    results[name] = {
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

# ==================== SELECT BEST MODEL ====================
best_model_name = max(results, key=lambda x: results[x]['Test Accuracy'])
best_model = trained_models[best_model_name]

print(f"\n{'='*50}")
print(f"BEST MODEL: {best_model_name}")
print(f"Test Accuracy: {results[best_model_name]['Test Accuracy']:.4f}")
print(f"{'='*50}")

# ==================== SAVE MODELS & ENCODERS ====================
print("\nSaving models and encoders...")

joblib.dump(best_model, 'models/best_risk_model.pkl')
joblib.dump(trained_models['Random Forest'], 'models/random_forest_model.pkl')

# Save encoders for later use
encoders = {
    'gender': le_gender,
    'stress': le_stress,
    'activity': le_activity,
    'illness': le_illness,
    'history': le_history,
    'treatment': le_treatment,
    'work': le_work
}

joblib.dump(encoders, 'models/encoders.pkl')

# Save feature columns for app
joblib.dump(feature_cols, 'models/feature_cols.pkl')

print("\n Models trained and saved successfully!")
print("Models saved in 'models/' directory")
