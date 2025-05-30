# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Load models and preprocessing objects
@st.cache_resource
def load_model_artifacts():
    try:
        # Try to load pre-trained models
        artifacts = {
            'loan_approval_model': joblib.load('loan_approval_model.pkl'),
            'classification_model': joblib.load('classification_model.pkl'),
            'scaler': joblib.load('scaler.pkl'),
            'label_encoders': joblib.load('label_encoders.pkl')
        }
    except FileNotFoundError:
        st.warning("Pre-trained models not found! Training new models...")
        
        # Load dataset
        df = pd.read_csv("Loan approval prediction (1).csv")

        # Define feature columns
        features = [
            "person_age", "person_income", "person_home_ownership",
            "person_emp_length", "loan_intent", "loan_grade", "loan_amnt",
            "loan_int_rate", "loan_percent_income", "cb_person_default_on_file",
            "cb_person_cred_hist_length"
        ]

        # Define categorical feature mappings
        CATEGORIES = {
            "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "OTHER"],
            "loan_intent": ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
            "loan_grade": ["A", "B", "C", "D", "E", "F", "G"],
            "cb_person_default_on_file": ["Y", "N"]
        }

        # Apply encoding
        label_encoders = {}
        for col, cats in CATEGORIES.items():
            le = LabelEncoder()
            le.fit(cats)  # Fit on all possible categories
            df[col] = le.transform(df[col])
            label_encoders[col] = le

        # Extract features and target variables
        X = df[features]
        y_regression = df["loan_status"]
        y_classification = (df["loan_status"] > 0.5).astype(int)

        # Split dataset
        X_train, X_test, y_train_reg, y_test_reg = train_test_split(
            X, y_regression, test_size=0.2, random_state=42
        )

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train models
        loan_approval_model = RandomForestRegressor(random_state=42)
        loan_approval_model.fit(X_train_scaled, y_train_reg)

        classification_model = RandomForestClassifier(n_estimators=100, random_state=42)
        classification_model.fit(X_train_scaled, y_classification[y_train_reg.index])

        # Save models
        joblib.dump(loan_approval_model, "loan_approval_model.pkl")
        joblib.dump(classification_model, "classification_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(label_encoders, "label_encoders.pkl")

        artifacts = {
            "loan_approval_model": loan_approval_model,
            "classification_model": classification_model,
            "scaler": scaler,
            "label_encoders": label_encoders
        }
    
    return artifacts

def preprocess_input(input_data, artifacts):
    # Create DataFrame from input
    df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    for col, le in artifacts['label_encoders'].items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    
    # Select features in correct order
    features = [
        "person_age", "person_income", "person_home_ownership",
        "person_emp_length", "loan_intent", "loan_grade", "loan_amnt",
        "loan_int_rate", "loan_percent_income", "cb_person_default_on_file",
        "cb_person_cred_hist_length"
    ]
    
    X = df[features]
    
    # Scale features
    X_scaled = artifacts['scaler'].transform(X)
    
    return X_scaled

def make_predictions(input_data, artifacts):
    # Preprocess the input
    X_scaled = preprocess_input(input_data, artifacts)
    
    # Make regression prediction
    regression_pred = artifacts['loan_approval_model'].predict(X_scaled)[0]
    
    # Make classification prediction
    class_proba = artifacts['classification_model'].predict_proba(X_scaled)[0][1]
    class_pred = 'Approved' if class_proba >= 0.5 else 'Disapproved'
    
    return {
        'regression_value': regression_pred,
        'approval_probability': class_proba,
        'approval_status': class_pred
    }

# Main Streamlit app
def main():
    st.title("Loan Approval Prediction")
    
    # Load model artifacts
    artifacts = load_model_artifacts()
    
    if artifacts is None:
        st.error("Failed to load or train models")
        return
    
    # Get category options from encoders
    cat_options = {
        'person_home_ownership': list(artifacts['label_encoders']['person_home_ownership'].classes_),
        'loan_intent': list(artifacts['label_encoders']['loan_intent'].classes_),
        'loan_grade': list(artifacts['label_encoders']['loan_grade'].classes_),
        'cb_person_default_on_file': list(artifacts['label_encoders']['cb_person_default_on_file'].classes_)
    }

    # Create input form
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
            person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
            person_home_ownership = st.selectbox(
                "Home Ownership",
                options=cat_options['person_home_ownership']
            )
            person_emp_length = st.number_input(
                "Employment Length (years)", 
                min_value=0.0, 
                max_value=50.0, 
                value=5.0, 
                step=0.5
            )
            loan_intent = st.selectbox(
                "Loan Purpose",
                options=cat_options['loan_intent']
            )
        
        with col2:
            loan_grade = st.selectbox(
                "Loan Grade",
                options=cat_options['loan_grade']
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
                options=cat_options['cb_person_default_on_file']
            )
            cb_person_cred_hist_length = st.number_input(
                "Credit History Length (years)", 
                min_value=0, 
                value=5
            )
        
        submitted = st.form_submit_button("Predict Loan Approval")
    
    if submitted:
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
        
        with st.spinner('Making predictions...'):
            predictions = make_predictions(input_data, artifacts)
        
        st.header("Prediction Results")
        
        if predictions['approval_status'] == 'Approved':
            st.success(f"Loan Status: {predictions['approval_status']}")
        else:
            st.error(f"Loan Status: {predictions['approval_status']}")
        
        st.write(f"Approval Probability: {predictions['approval_probability']*100:.2f}%")
        st.progress(predictions['approval_probability'])
        
        st.write(f"Predicted Loan Status Value: {predictions['regression_value']:.4f}")
        
        if predictions['approval_status'] == 'Approved':
            st.info("**Congratulations!** Based on the information provided, your loan is likely to be approved.")
        else:
            st.warning("""
            **Note:** Based on the information provided, your loan application may not meet 
            the current approval criteria. Consider improving your credit score, reducing debt-to-income ratio, 
            or applying for a smaller loan amount.
            """)

if __name__ == "__main__":
    main()