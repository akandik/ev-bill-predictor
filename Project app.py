import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load dataset and train model
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

# Clean dataset
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

# Prediction Function

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

# GUI Application (Tkinter)
def calculate_bill():
    try:
        energy = float(entry_energy.get())
        duration = float(entry_duration.get())
        rate = float(entry_rate.get())
        temp = float(entry_temp.get())
        
        charger = combo_charger.get()
        time_day = combo_time.get()
        user = combo_user.get()
        
        bill = predict_ev_bill(energy, duration, rate, charger, time_day, user, temp)
        messagebox.showinfo("Predicted Bill", f" Estimated Charging Cost: ${bill}")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# Create main window
root = tk.Tk()
root.title("EV Self-Charging Bill Calculator (AI Powered)")
root.geometry("400x450")
root.resizable(False, False)

# Labels and Inputs
tk.Label(root, text="Energy Consumed (kWh):").pack(pady=5)
entry_energy = tk.Entry(root)
entry_energy.pack()

tk.Label(root, text="Charging Duration (hours):").pack(pady=5)
entry_duration = tk.Entry(root)
entry_duration.pack()

tk.Label(root, text="Charging Rate (kW):").pack(pady=5)
entry_rate = tk.Entry(root)
entry_rate.pack()

tk.Label(root, text="Temperature (°C):").pack(pady=5)
entry_temp = tk.Entry(root)
entry_temp.pack()

tk.Label(root, text="Charger Type:").pack(pady=5)
combo_charger = ttk.Combobox(root, values=["Level 1", "Level 2", "DC Fast Charger"])
combo_charger.pack()

tk.Label(root, text="Time of Day:").pack(pady=5)
combo_time = ttk.Combobox(root, values=["Morning", "Afternoon", "Evening", "Night"])
combo_time.pack()

tk.Label(root, text="User Type:").pack(pady=5)
combo_user = ttk.Combobox(root, values=["Commuter", "Casual Driver", "Long-Distance Traveler"])
combo_user.pack()

tk.Button(root, text="Calculate Bill", command=calculate_bill, bg="green", fg="white").pack(pady=20)

root.mainloop()
