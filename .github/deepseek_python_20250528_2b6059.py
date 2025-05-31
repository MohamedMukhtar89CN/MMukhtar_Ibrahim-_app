# app.py

streamlit run app.py
pip install streamlit

! pip install streamlit
! pip install -r requirements.txt

from flask import Flask, request, jsonify
from flask import 
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
classification_features_full = features
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np



# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load the model and preprocessing objects
@st.cache_resource
def load_model_artifacts():
    try:
        # Try to load pre-trained models if they exist
        artifacts = {
            'loan_approval_model': joblib.load('C:\Users\mmukh\Documents\FlaskAPI.py\loan_approval_model (7).pkl'),
            'classification_model': joblib.load('C:\Users\mmukh\Documents\FlaskAPI.py\classification_model (7).pkl'),
            'scaler': joblib.load('C:\Users\mmukh\Documents\FlaskAPI.py\scaler (10).pkl'),
            'label_encoder': joblib.load('C:\Users\mmukh\Documents\FlaskAPI.py\label_encoder (10).pkl')
        }
    except:
        # If models don't exist, train new ones
        st.warning("Pre-trained models not found. Training new models...")
        
        # Load your dataset
        df = pd.read_csv('Loan approval prediction (1).csv')
        
        # Prepare features and target
        features = [
            "person_age", "person_income", "person_home_ownership",
            "person_emp_length", "loan_intent", "loan_grade", "loan_amnt",
            "loan_int_rate", "loan_percent_income", "cb_person_default_on_file",
            "cb_person_cred_hist_length"
        ]
        
        # Separate features and target
        X = df[features]
        y_regression = df['loan_status']  # For regression
        y_classification = df['loan_status']  # For classification
        
        # Preprocessing
        scaler = StandardScaler()
        label_encoder = LabelEncoder()
        
        # Encode categorical variables
        categorical_columns = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
        for column in categorical_columns:
            X[column] = label_encoder.fit_transform(X[column])
        
        # Scale features
        X_scaled = scaler.fit_transform(X)
        
        # Train models
        loan_approval_model = LinearRegression()
        loan_approval_model.fit(X_scaled, y_regression)
        
        classification_model = RandomForestClassifier(n_estimators=100, random_state=42)
        classification_model.fit(X_scaled, y_classification)
        
        # Save models for future use
        joblib.dump(loan_approval_model, 'loan_approval_model.pkl')
        joblib.dump(classification_model, 'classification_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(label_encoder, 'label_encoder.pkl')
        
        artifacts = {
            'loan_approval_model': loan_approval_model,
            'classification_model': classification_model,
            'scaler': scaler,
            'label_encoder': label_encoder
        }
    
    return artifacts






app = Flask(__name__, template_folder = "."



streamlit run "C:\Users\YourUser\deepseek_python_20250528_2b6059.py"
# Load the model and preprocessing objects
@st.cache_resource
def load_model_artifacts():
    # Load your trained model and preprocessing objects here
    C:\Users\mmukh\Documents\FlaskAPI.py\loan_approval_model (7).pkl
    C:\Users\mmukh\Documents\FlaskAPI.py\scaler (5).pkl
    C:\Users\mmukh\Documents\FlaskAPI.py\label_encoder (4).pkl
    C:\Users\mmukh\Documents\FlaskAPI.py\classification_model (2).pkl
    C:\Users\mmukh\Documents\FlaskAPI.py\UI.HTML.html
    C:\Users\mmukh\Documents\FlaskAPI.py\requirements (6).txt
    classification_model (2).pkl
    # For this example, we'll create them from scratch as placeholders
    # In production, you would load your actual trained models
   artifacts
    # Create placeholder models
    loan_approval_model = RandomForestRegressor()
    classification_model = XGBClassifier(n_estimators=100, random_state=42)
    
    # Create placeholder scaler and encoders
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    
    return {
        'loan_approval_model': loan_approval_model,
        'classification_model': classification_model,
        'scaler': scaler,
        'label_encoder': label_encoder
    }

# Preprocess input data
def preprocess_input(input_data, artifacts):
    # Create a DataFrame from the input data
    df = pd.DataFrame([input_data])
    
    # Encode categorical variables (same encoding as during training)
    categorical_columns = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
    
    for column in categorical_columns:
        df[column] = artifacts['label_encoder'].fit_transform(df[column])
    
    # Select features in the correct order
    features = [
        "person_age", "person_income", "person_home_ownership",
        "person_emp_length", "loan_intent", "loan_grade", "loan_amnt",
        "loan_int_rate", "loan_percent_income", "cb_person_default_on_file",
        "cb_person_cred_hist_length"
    ]
    
    X = df[features]
    
    # Scale the features
    X_scaled = artifacts['scaler'].fit_transform(X)
    
    return X_scaled

# Make predictions
def make_predictions(input_data, artifacts):
    # Preprocess the input
    X_scaled = preprocess_input(input_data, artifacts)
    
    # Make regression prediction
    regression_pred = artifacts['loan_approval_model'].predict(X_scaled)
    regression_value = np.expm1(regression_pred[0])  # Undo log transformation
    
    # Make classification prediction
    class_pred = classification_model.predict(X_scaled)[0]
    approval_status = 'Approved' if class_pred == 1 else 'Disapproved'
    



    return {
        'regression_value': regression_value,
        'approval_probability': class_proba,
        'approval_status': class_pred
    }

# Main Streamlit app
def main():
    st.title("Loan Approval Prediction")
    st.write("""
    This app predicts the likelihood of loan approval based on applicant information.
    """)
    
    # Load model artifacts
    artifacts = load_model_artifacts()
    
    # Create input form
    with st.form("loan_form"):
        st.header("Applicant Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
            person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
            person_home_ownership = st.selectbox(
                "Home Ownership",
                ["RENT", "OWN", "MORTGAGE", "OTHER"]
            )
            person_emp_length = st.number_input(
                "Employment Length (years)", 
                min_value=0.0, 
                max_value=50.0, 
                value=5.0, 
                step=0.5
            )
            loan_intent = st.selectbox(
                "Loan Intent",
                ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
            )
        
        with col2:
            loan_grade = st.selectbox(
                "Loan Grade",
                ["A", "B", "C", "D", "E", "F", "G"]
            )
            loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000)
            loan_int_rate = st.number_input(
                "Interest Rate (%)", 
                min_value=0.0, 
                max_value=30.0, 
                value=7.5, 
                step=0.1
            )
            loan_percent_income = st.number_input(
                "Loan Percentage of Income", 
                min_value=0.0, 
                max_value=100.0, 
                value=20.0, 
                step=0.1
            )
            cb_person_default_on_file = st.selectbox(
                "Previous Default",
                ["Y", "N"]
            )
            cb_person_cred_hist_length = st.number_input(
                "Credit History Length (years)", 
                min_value=0, 
                value=5
            )
        
        submitted = st.form_submit_button("Predict Loan Approval")
    
    # When form is submitted
    if submitted:
        # Create input dictionary
        input_data = {
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length
        }
        
        # Make predictions
        with st.spinner('Making predictions...'):
            predictions = make_predictions(input_data, artifacts)
        
        # Display results
        st.header("Prediction Results")
        
        # Approval status with color
        if predictions['approval_status'] == 'Approved':
            st.success(f"Loan Status: {predictions['approval_status']}")
        else:
            st.error(f"Loan Status: {predictions['approval_status']}")
        
        # Probability meter
        st.write(f"Approval Probability: {predictions['approval_probability']*100:.2f}%")
        st.progress(predictions['approval_probability'])
        
        # Regression value
        st.write(f"Predicted Loan Status Value: {predictions['regression_value']:.4f}")
        
        # Explanation
        if predictions['approval_status'] == 'Approved':
            st.info("""
            **Congratulations!** Based on the information provided, your loan is likely to be approved.
            """)
        else:
            st.warning("""
            **Note:** Based on the information provided, your loan application may not meet 
            the current approval criteria. Consider improving your credit score, reducing debt-to-income ratio, 
            or applying for a smaller loan amount.
            """)

if __name__ == "__main__":
    main()