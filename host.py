from flask import Flask, request, render_template, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import joblib
import pandas as pd
import os
from pymongo import MongoClient

app = Flask(__name__)
app.secret_key = 'your_secret_key'
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'home'

# MongoDB connection setup
client = MongoClient('localhost',27017)
db = client['churn_db']
collection = db['predictions']

# Load the model
model = joblib.load(os.path.join('C:\\', 'Users', 'sec21', 'OneDrive', 'Desktop', 'churn_prediction', 'flask_app', 'xgb_model.pkl'))

# Simple user store
users = {'sarumathi03': {'password': 'saru2003'}}

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            user = User(username)
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials')
    return redirect(url_for('home'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/index')
@login_required
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    form_data = request.form
    input_data = {
        'SeniorCitizen': int(form_data['SeniorCitizen']),
        'gender': form_data['gender'],
        'Partner': form_data['Partner'],
        'Dependents': form_data['Dependents'],
        'tenure': float(form_data['tenure']),
        'PhoneService': form_data['PhoneService'],
        'MultipleLines': form_data['MultipleLines'],
        'InternetService': form_data['InternetService'],
        'OnlineSecurity': form_data['OnlineSecurity'],
        'OnlineBackup': form_data['OnlineBackup'],
        'DeviceProtection': form_data['DeviceProtection'],
        'TechSupport': form_data['TechSupport'],
        'StreamingTV': form_data['StreamingTV'],
        'StreamingMovies': form_data['StreamingMovies'],
        'Contract': form_data['Contract'],
        'PaperlessBilling': form_data['PaperlessBilling'],
        'PaymentMethod': form_data['PaymentMethod'],
        'MonthlyCharges': float(form_data['MonthlyCharges']),
        'TotalCharges': float(form_data['TotalCharges'])
    }
    
    # Convert input_data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure the model has a preprocessor and classifier
    if 'preprocessor' in model.named_steps and 'classifier' in model.named_steps:
        # Apply the same preprocessing as was done during model training
        input_df = model.named_steps['preprocessor'].transform(input_df)

        # Make prediction
        prediction = model.named_steps['classifier'].predict(input_df)
        prediction_prob = model.named_steps['classifier'].predict_proba(input_df)
    else:
        # If model does not have named steps, assume it's a direct classifier
        prediction = model.predict(input_df)
        prediction_prob = model.predict_proba(input_df)
    
    # Extract churn probability (probability of class 1)
    churn_probability = float(prediction_prob[0][1])  # Convert to native Python float
    
    # Format churn_probability to 2 decimal places
    formatted_probability = f"{churn_probability:.2f}"
    
    # Store data in MongoDB
    input_data['prediction'] = int(prediction[0])
    input_data['probability'] = churn_probability
    input_data['username'] = current_user.id
    collection.insert_one(input_data)
    
    return render_template('result.html', prediction=prediction[0], probability=formatted_probability)

@app.route('/history')
@login_required
def history():
    user_predictions = collection.find({'username': current_user.id})
    records = [record for record in user_predictions]
    return render_template('history.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)
