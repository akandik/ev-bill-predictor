from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Initialize Flask app
app = Flask(__name__)

# Load and train model
file_path = "ev_charging_patterns.csv"  
df = pd.read_csv(file_path)

features = [
    "Energy Consumed (kWh)", 
    "Charging Duration (hours)", 
    "Charging Rate (kW)", 
    "Charger Type", 
    "Time of Day", 
    "User Type", 
    "Temperature (°C)"
]
target = "Charging Cost (USD)"

df_ml = df[features + [target]].dropna()
X = df_ml[features]
y = df_ml[target]

categorical = ["Charger Type", "Time of Day", "User Type"]
numeric = ["Energy Consumed (kWh)", "Charging Duration (hours)", "Charging Rate (kW)", "Temperature (°C)"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", "passthrough", numeric)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Prediction function
def predict_ev_bill(energy_kwh, duration_hr, rate_kw, charger_type, time_of_day, user_type, temperature_c):
    input_data = pd.DataFrame([{
        "Energy Consumed (kWh)": energy_kwh,
        "Charging Duration (hours)": duration_hr,
        "Charging Rate (kW)": rate_kw,
        "Charger Type": charger_type,
        "Time of Day": time_of_day,
        "User Type": user_type,
        "Temperature (°C)": temperature_c
    }])
    predicted_cost = model.predict(input_data)[0]
    return round(predicted_cost, 2)


# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        energy = float(request.form['energy'])
        duration = float(request.form['duration'])
        rate = float(request.form['rate'])
        temp = float(request.form['temperature'])
        charger = request.form['charger']
        time_day = request.form['time']
        user = request.form['user']

        bill = predict_ev_bill(energy, duration, rate, charger, time_day, user, temp)
        return render_template('result.html', bill=bill)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
