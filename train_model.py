import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import joblib
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = 'logistic_regression.csv'
TARGET_VARIABLE = 'loan_status'
POSITIVE_CLASS = 'Charged Off'
NEGATIVE_CLASS = 'Fully Paid'

# Load data
print("Loading data...")
df = pd.read_csv(DATA_FILE)

# Feature Engineering
print("Performing feature engineering...")
# Drop less useful columns
cols_to_drop = ['title', 'sub_grade', 'issue_d']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Convert 'term' to numeric
df['term'] = df['term'].apply(lambda term: int(term.split()[0]))

# Convert 'emp_length' to numeric
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

# Convert 'earliest_cr_line' to years of credit history
df['earliest_cr_line_dt'] = pd.to_datetime(df['earliest_cr_line'], format='mixed', errors='coerce')
df.dropna(subset=['earliest_cr_line_dt'], inplace=True)
df['earliest_cr_line_year'] = df['earliest_cr_line_dt'].dt.year
reference_year = 2017
df['credit_history_length'] = reference_year - df['earliest_cr_line_year']
df.drop(columns=['earliest_cr_line', 'earliest_cr_line_dt', 'earliest_cr_line_year'], inplace=True)

# Extract State from Address
df['state'] = df['address'].str.extract(r'([A-Z]{2})\s+\d{5}$')
df['state'].fillna('Missing', inplace=True)
df.drop(columns=['address'], inplace=True)

# Create binary flags
for col in ['pub_rec', 'mort_acc', 'pub_rec_bankruptcies']:
    flag_col_name = f'{col}_flag'
    df[flag_col_name] = df[col].apply(lambda x: 1 if pd.notna(x) and x > 0 else 0)

# Encode Target Variable
df[TARGET_VARIABLE] = df[TARGET_VARIABLE].apply(lambda x: 1 if x == POSITIVE_CLASS else 0)

# Separate features and target
X = df.drop(TARGET_VARIABLE, axis=1)
y = df[TARGET_VARIABLE]

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Create preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# Create model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
print("Training model...")
model_pipeline.fit(X_train, y_train)

# Save model and preprocessor
print("Saving model and preprocessor...")
joblib.dump(model_pipeline, 'model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

print("Model and preprocessor saved successfully!") 