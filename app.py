from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import Optional
import os

app = FastAPI(title="LoanTap Default Prediction API")

# CORS Middleware
# Allows all origins, all methods, all headers for simplicity.
# For production, you might want to restrict origins.
origins = ["*"] # Or specify your frontend's origin e.g., "http://localhost:xxxx", "null" (for file://)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and preprocessor
try:
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    preprocessor_path = os.path.join(os.path.dirname(__file__), 'preprocessor.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        raise FileNotFoundError("Model files not found. Please run train_model.py first.")
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    print("Model and preprocessor loaded successfully!")
except Exception as e:
    print(f"Error loading model files: {str(e)}")
    raise

class LoanApplication(BaseModel):
    loan_amnt: float
    term: str
    int_rate: float
    installment: float
    grade: str
    emp_title: str
    emp_length: str
    home_ownership: str
    annual_inc: float
    verification_status: str
    dti: float
    earliest_cr_line: str
    open_acc: int
    pub_rec: int
    revol_bal: float
    revol_util: float
    total_acc: int
    mort_acc: int
    pub_rec_bankruptcies: int
    address: str
    # Added fields based on error: "['purpose', 'initial_list_status', 'application_type'] not in index"
    purpose: Optional[str] = "debt_consolidation" # Assuming str, added default
    initial_list_status: Optional[str] = "w" # Assuming str, added default (w or f)
    application_type: Optional[str] = "Individual" # Assuming str, added default

    class Config:
        schema_extra = {
            "example": {
                "loan_amnt": 10000,
                "term": "36 months",
                "int_rate": 10.5,
                "installment": 325.0,
                "grade": "B",
                "emp_title": "Software Engineer",
                "emp_length": "5 years",
                "home_ownership": "MORTGAGE",
                "annual_inc": 75000,
                "verification_status": "Verified",
                "dti": 15.5,
                "earliest_cr_line": "2010-01-01",
                "open_acc": 5,
                "pub_rec": 0,
                "revol_bal": 5000,
                "revol_util": 30.5,
                "total_acc": 10,
                "mort_acc": 1,
                "pub_rec_bankruptcies": 0,
                "address": "123 Main St, CA",
                "purpose": "debt_consolidation",
                "initial_list_status": "w",
                "application_type": "Individual"
            }
        }

def preprocess_input(data: dict) -> pd.DataFrame:
    """Preprocess input data to match training data format"""
    df = pd.DataFrame([data])
    
    # Convert term to numeric
    df['term'] = df['term'].apply(lambda x: int(x.split()[0]))
    
    # Convert emp_length to numeric
    def parse_emp_length(length):
        if pd.isna(length):
            return 0
        elif '< 1 year' in length:
            return 0
        elif '10+ years' in length:
            return 10
        else:
            return int(length.split()[0])
    
    df['emp_length'] = df['emp_length'].apply(parse_emp_length)
    
    # Convert earliest_cr_line to credit history length
    df['earliest_cr_line_dt'] = pd.to_datetime(df['earliest_cr_line'], format='mixed', errors='coerce')
    reference_year = 2017
    df['credit_history_length'] = reference_year - df['earliest_cr_line_dt'].dt.year
    df.drop(columns=['earliest_cr_line', 'earliest_cr_line_dt'], inplace=True)
    
    # Extract state from address
    df['state'] = df['address'].str.extract(r'([A-Z]{2})\s+\d{5}$')
    df['state'].fillna('Missing', inplace=True)
    df.drop(columns=['address'], inplace=True)
    
    # Create binary flags
    for col in ['pub_rec', 'mort_acc', 'pub_rec_bankruptcies']:
        flag_col_name = f'{col}_flag'
        df[flag_col_name] = df[col].apply(lambda x: 1 if pd.notna(x) and x > 0 else 0)
    
    return df

@app.get("/")
def read_root():
    return {
        "message": "Welcome to LoanTap Default Prediction API",
        "endpoints": {
            "/predict": "POST - Predict loan default probability",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

@app.post("/predict")
async def predict(loan_application: LoanApplication):
    try:
        # Convert input to DataFrame and preprocess
        input_data = preprocess_input(loan_application.dict())
        
        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        return {
            "prediction": "Charged Off" if prediction[0] == 1 else "Fully Paid",
            "probability": float(probability[0][1]),  # Probability of default
            "confidence": float(max(probability[0])),
            "features_used": model.named_steps['preprocessor'].get_feature_names_out().tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
