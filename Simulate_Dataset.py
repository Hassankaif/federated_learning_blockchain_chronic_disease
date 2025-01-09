import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Generate dataset
def generate_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "Age": np.random.randint(30, 80, num_samples),
        "Gender": np.random.choice([0, 1], num_samples),  # 0 = Male, 1 = Female
        "BMI": np.round(np.random.uniform(18, 35, num_samples), 2),
        "Blood_Pressure_Systolic": np.random.randint(110, 180, num_samples),
        "Blood_Pressure_Diastolic": np.random.randint(70, 110, num_samples),
        "Glucose_Level": np.random.randint(70, 200, num_samples),
        "Cholesterol_Level": np.random.randint(150, 300, num_samples),
        "Smoking_Status": np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),  # 0 = No, 1 = Yes
        "Physical_Activity": np.random.choice([0, 1, 2], num_samples, p=[0.4, 0.4, 0.2]),  # 0 = Low, 1 = Moderate, 2 = High
        "Has_Chronic_Disease": np.random.choice([0, 1], num_samples, p=[0.6, 0.4])  # 0 = No, 1 = Yes
    }
    return pd.DataFrame(data)

# Create the dataset
dataset = generate_data()

# Split dataset for each client
client_1_data = dataset[["BMI", "Glucose_Level", "Has_Chronic_Disease"]]
client_2_data = dataset[["Blood_Pressure_Systolic", "Blood_Pressure_Diastolic", "Cholesterol_Level", "Has_Chronic_Disease"]]
client_3_data = dataset[["Age", "Physical_Activity", "Smoking_Status", "Has_Chronic_Disease"]]

# Save each client's data as a CSV
client_1_data.to_csv("client_1_data.csv", index=False)
client_2_data.to_csv("client_2_data.csv", index=False)
client_3_data.to_csv("client_3_data.csv", index=False)
dataset.to_csv('simulated_health_data.csv', index=False)
print("Datasets created and saved for Client 1, Client 2, and Client 3.")
