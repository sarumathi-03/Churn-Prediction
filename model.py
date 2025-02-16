import numpy as np
import pandas as pd
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Data Ingestion
data_path = r"datset.csv"
data = pd.read_csv(data_path)

# Step 2: Exploratory Data Analysis
logging.info("Initial Data Info:")
logging.info(data.info())
logging.info("\nMissing Values:")
logging.info(data.isnull().sum())
logging.info("\nDataset Summary:\n%s", data.describe())

# Only calculate correlation for numeric columns
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(12,6))
sns.heatmap(data[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 3: Data Preprocessing
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.dropna(subset=['TotalCharges'], inplace=True)

label_encoder = LabelEncoder()
categorical_columns = data.select_dtypes(include=['object']).columns
data[categorical_columns] = data[categorical_columns].apply(lambda col: label_encoder.fit_transform(col))

# Step 4: Feature Engineering
X = data.drop('Churn', axis=1)
y = data['Churn']
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in X.columns if col not in numeric_features]

scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# All categorical features are already label encoded from Step 3
X_final = X.values

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Handling Class Imbalance
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Step 6: Model Training & Hyperparameter Tuning
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

models = {'RandomForest': rf, 'XGBoost': xgb}
best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train_balanced, y_train_balanced)
    train_acc = model.score(X_train_balanced, y_train_balanced)
    test_acc = model.score(X_test, y_test)
    logging.info(f"{name} - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    if test_acc > best_score:
        best_score = test_acc
        best_model = model

# Step 7: Model Evaluation
y_pred_test = best_model.predict(X_test)
logging.info("\nClassification Report:\n%s", classification_report(y_test, y_pred_test))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, best_model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Save Model & Preprocessing Objects
with open("best_model.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

logging.info("All models and preprocessing objects saved successfully!")
