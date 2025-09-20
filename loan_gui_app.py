import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import streamlit as st

# Load dataset
df = pd.read_csv("dataset.csv")

# Handle categorical columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)  # Handle missing values in categorical columns
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Encode categorical columns
    label_encoders[col] = le

# Handle numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
imputer = SimpleImputer(strategy='mean')
df[num_cols] = imputer.fit_transform(df[num_cols])  # Impute missing values in numerical columns

# Features and target
X = df.drop(["Loan_Status", "Loan_ID"], axis=1)  # Drop 'Loan_Status' and 'Loan_ID' columns
y = df["Loan_Status"]

# Feature Scaling for SVM
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVC model with Grid Search for hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}  # Define a dictionary with hyperparameters for tuning
grid_search = GridSearchCV(SVC(), param_grid, cv=5)  # Set up the GridSearchCV with the Support Vector Classifier (SVC), the parameter grid, and 5-fold cross-validation
grid_search.fit(X_train, y_train)  # Train the model with different hyperparameters and cross-validation
model = grid_search.best_estimator_  # Get the best model based on the grid search results

# Streamlit App

def predict_loan(input_data):
    # Scale the input data
    input_data_scaled = scaler.transform([input_data])

    # Prediction
    prediction = model.predict(input_data_scaled)
    return "Loan Approved!" if prediction[0] == 1 else "Loan Rejected"


# Custom CSS for styling
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        .header {
            text-align: center;
            margin-top: 20px;
        }
        .stTextInput, .stSelectbox, .stButton {
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #2980B9;
            color:white;
            border-radius: 5px;
            font-size: 1.1em;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: white;
            color:black;
        }
        
        .stSelectbox>div, .stTextInput>div {
            font-size: 1.1em;
        }
        
    </style>
""", unsafe_allow_html=True)

# Streamlit form layout
st.markdown("<h1 class='header'>Loan Approval Prediction</h1>", unsafe_allow_html=True)

# Create input fields for the user to enter feature values
gender = st.selectbox("Gender (0=Female, 1=Male):", options=["Select", "Male", "Female"])
married = st.selectbox("Married (0=No, 1=Yes):", options=["Select", "Yes", "No"])
dependents = st.text_input("Dependents (0, 1, 2, 3):", value="")
education = st.selectbox("Education (0=Graduate, 1=Not Graduate):", options=["Select", "Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed (0=No, 1=Yes):", options=["Select", "Yes", "No"])
applicant_income = st.text_input("Applicant Income:", value="")
coapplicant_income = st.text_input("Coapplicant Income:", value="")
loan_amount = st.text_input("Loan Amount:", value="")
loan_term = st.text_input("Loan Amount Term (in months):", value="")
credit_history = st.selectbox("Credit History (0=Poor, 1=Good):", options=["Select", "Good", "Poor"])
property_area = st.selectbox("Property Area (0=Rural, 1=Semiurban, 2=Urban):", options=["Select", "Rural", "Semiurban", "Urban"])

# Convert the inputs to appropriate numerical values
if gender != "Select":
    gender = 1 if gender == "Male" else 0
if married != "Select":
    married = 1 if married == "Yes" else 0
if education != "Select":
    education = 0 if education == "Graduate" else 1
if self_employed != "Select":
    self_employed = 1 if self_employed == "Yes" else 0
if credit_history != "Select":
    credit_history = 1 if credit_history == "Good" else 0
if property_area != "Select":
    property_area = {"Rural": 0, "Semiurban": 1, "Urban": 2}.get(property_area, 0)

# Button to make prediction
if st.button("Predict Loan Status"):
    if all(value != "" for value in [gender, married, dependents, education, self_employed, applicant_income, coapplicant_income, loan_amount, loan_term, credit_history, property_area]):
        input_data = [
            gender, married, int(dependents), education, self_employed,
            float(applicant_income), float(coapplicant_income), float(loan_amount), int(loan_term),
            credit_history, property_area
        ]

        result = predict_loan(input_data)
        st.success(result)
    else:
        st.error("Please fill all the fields.")
