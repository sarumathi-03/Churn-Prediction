import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from tpot import TPOTClassifier


data = pd.read_csv(r"E:\Machine Learning\archive\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Handle missing values
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna(subset=['TotalCharges'])

# Define categorical labels mapping
labels = {
    "Male": 1, "Female": 0,
    "Yes": 1, "No": 0,
    "No phone service": 0,
    "Fiber optic": 1, "DSL": 2,
    "No internet service": 0,
    "Month-to-month": 1, "Two year": 2, "One year": 3,
    "Electronic check": 1, "Mailed check": 2,
    "Bank transfer (automatic)": 3, "Credit card (automatic)": 4
}

# Map categorical variables to numerical values
columns_to_convert = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'Churn'
]

for column in columns_to_convert:
    data[column] = data[column].map(labels)

# Split features and target variable
X = data.drop('Churn', axis=1)
y = data['Churn']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y
)

# Define preprocessing steps
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in X.columns if col not in numeric_features]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Apply preprocessing to numeric features
X_train[numeric_features] = numeric_transformer.fit_transform(X_train[numeric_features])
X_test[numeric_features] = numeric_transformer.transform(X_test[numeric_features])

# Apply preprocessing to categorical features
X_train_cat = categorical_transformer.fit_transform(X_train[categorical_features])
X_test_cat = categorical_transformer.transform(X_test[categorical_features])

# Concatenate preprocessed numeric and categorical features
X_train = np.hstack((X_train[numeric_features].values, X_train_cat.toarray()))
X_test = np.hstack((X_test[numeric_features].values, X_test_cat.toarray()))

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train_smote, y_train_smote)

train_accuracy = tpot.score(X_train_smote, y_train_smote)
print("Training accuracy: ", train_accuracy)

test_accuracy = tpot.score(X_test, y_test)
print("Test accuracy: ", test_accuracy)