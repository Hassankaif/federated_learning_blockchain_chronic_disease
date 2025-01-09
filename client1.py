import flwr as fl
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load the simulated dataset
def load_simulated_data(client_id):
    # Load the full dataset
    df = pd.read_csv("simulated_health_data.csv")

    # Split data based on client-specific features
    if client_id == 1:
        df = df[["glucose", "BMI", "chronic_condition"]]
    elif client_id == 2:
        df = df[["blood_pressure", "cholesterol", "chronic_condition"]]
    elif client_id == 3:
        df = df[["age", "exercise_frequency", "chronic_condition"]]

    # Separate features (X) and labels (y)
    X = df.drop("Has_Chronic_Disease", axis=1).values
    y = df["Has_Chronic_Disease"].values

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Build a model suitable for the chronic disease dataset
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")  # Binary classification for chronic disease
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Load client-specific data
client_id = int(sys.argv[1])  # Pass client ID as an argument
x_train, x_test, y_train, y_test = load_simulated_data(client_id)

# Initialize model
model = build_model(x_train.shape[1])

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=0)
        print("Fit history:", history.history)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("Eval accuracy:", accuracy)
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_client(
    server_address="localhost:" + str(sys.argv[1]),
    client=FlowerClient(),
    grpc_max_message_length=1024 * 1024 * 1024
)
