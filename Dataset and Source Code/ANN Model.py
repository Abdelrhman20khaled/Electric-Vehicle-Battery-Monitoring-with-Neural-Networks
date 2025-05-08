# -*- coding: utf-8 -*-
"""
ANN Model for SOC Prediction (Loads & Saves to Google Drive)

This script mounts Google Drive, loads battery data from a specified path,
preprocesses it, builds, trains, evaluates, and saves the complete
Artificial Neural Network (ANN) model using TensorFlow/Keras directly
to Google Drive. It predicts the State of Charge (SOC) based on
voltage and current, and also shows predictions on the test set.

**Version with reduced batch size to potentially prevent RAM crashes.**
"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import os
# Import joblib to save the scaler
import joblib

# --- NEW: Mount Google Drive ---
from google.colab import drive
try:
    # Attempt to mount, but don't force remount if already mounted
    if not os.path.exists('/content/drive/MyDrive'):
      drive.mount('/content/drive')
      print("Google Drive mounted successfully.")
    else:
      print("Google Drive already mounted.")
except Exception as e:
    print(f"Error mounting Google Drive: {e}")
    print("Please ensure you are running this in a Google Colab environment.")
    exit()

# --- Define File Paths ---
# IMPORTANT: Change this path to the EXACT location of your CSV file in Google Drive
DRIVE_FILE_PATH = '/content/drive/MyDrive/EV_Project/Cleaning_Data.csv' # *** CHANGE THIS PATH ***

# --- NEW: Define Model and Scaler Save Paths in Google Drive ---
# IMPORTANT: Change this path to where you want to save the model in your Drive
# Make sure the folder path (e.g., /content/drive/MyDrive/MyModels/) exists in your Drive
DRIVE_SAVE_MODEL_PATH = '/content/drive/MyDrive/soc_model_complete.keras' # *** CHANGE THIS PATH ***
# Define path to save the scaler object in Drive
DRIVE_SAVE_SCALER_PATH = '/content/drive/MyDrive/soc_scaler.joblib' # *** CHANGE THIS PATH ***


# --- 1. Load Data ---
# Load the dataset from the specified path in Google Drive
print(f"\nLoading data from: {DRIVE_FILE_PATH}")
if not os.path.exists(DRIVE_FILE_PATH):
    print(f"Error: File not found at '{DRIVE_FILE_PATH}'.")
    print("Please check the file path variable in the script and ensure the file exists in your Drive.")
    exit()

try:
    df = pd.read_csv(DRIVE_FILE_PATH)
    print("Data loaded successfully from Google Drive.")
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    exit()

# --- 2. Initial Data Exploration & Preprocessing ---
print("\nOriginal Data Info:")
# Add a check if df exists before calling info()
if 'df' in locals():
    df.info()
    print("\nFirst 5 rows of original data:")
    print(df.head().to_markdown(index=False, numalign="left", stralign="left"))
else:
    print("DataFrame 'df' was not loaded. Exiting.")
    exit()


# Select relevant columns
relevant_cols = ['soc', 'voltage', 'current']
df_model = df[relevant_cols].copy()

# Handle missing values
print(f"\nData shape before dropping NaN: {df_model.shape}")
initial_rows = len(df_model)
df_model.dropna(inplace=True)
rows_after_drop = len(df_model)
print(f"Data shape after dropping NaN: {df_model.shape}")
print(f"Number of rows removed due to NaN: {initial_rows - rows_after_drop}")

if df_model.empty:
    print("\nError: No data remaining after removing missing values. Cannot proceed.")
    exit()

# Define features and target
X = df_model[['voltage', 'current']]
y = df_model['soc']

print("\nFeatures (X) head:")
print(X.head().to_markdown(index=False, numalign="left", stralign="left"))
print("\nTarget (y) head:")
print(y.head().to_markdown(index=False, numalign="left", stralign="left"))

# --- 3. Data Splitting ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining data shape: X={X_train.shape}, y={y_train.shape}")
print(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")

# --- 4. Feature Scaling ---
# Initialize the scaler
scaler = StandardScaler()
# Fit the scaler ONLY on training data and transform it
X_train_scaled = scaler.fit_transform(X_train)
# Transform the test data using the FITTED scaler
X_test_scaled = scaler.transform(X_test)
print("\nFirst 5 rows of scaled training features (voltage, current):")
print(pd.DataFrame(X_train_scaled, columns=X.columns).head().to_markdown(index=False, numalign="left", stralign="left"))

# --- NEW: Save the Scaler ---
print(f"\nSaving the scaler object to: {DRIVE_SAVE_SCALER_PATH}")
try:
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(DRIVE_SAVE_SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, DRIVE_SAVE_SCALER_PATH)
    print("Scaler saved successfully to Google Drive.")
except Exception as e:
    print(f"Error saving scaler: {e}")

# --- 5. Build the ANN Model ---
model = keras.Sequential(
    [
        layers.Dense(32, activation="relu", input_shape=[X_train_scaled.shape[1]], name="hidden_layer_1"),
        layers.Dense(16, activation="relu", name="hidden_layer_2"),
        layers.Dense(8, activation="relu", name="hidden_layer_3"),
        layers.Dense(1, name="output_layer")
    ]
)
print("\nModel Summary:")
model.summary()

# --- 6. Compile the Model ---
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])
print("\nModel compiled successfully.")

# --- 7. Train the Model ---
print("\nStarting model training...")
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# *** MODIFIED: Reduced batch_size from 64 to 32 ***
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32, # Reduced batch size
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)
print("Model training finished.")

# --- 8. Evaluate the Model ---
print("\nEvaluating model on test data...")
loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Set Performance:")
print(f"Mean Squared Error (MSE): {loss:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Interpretation: On average, the model's SOC prediction is off by approximately {mae:.2f} SOC percentage points on the unseen test data.")

# --- 9. Save the Trained Model to Google Drive ---
print(f"\nSaving the trained model to Google Drive at: {DRIVE_SAVE_MODEL_PATH}...")
try:
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(DRIVE_SAVE_MODEL_PATH), exist_ok=True)
    # Save the entire model (architecture, weights, optimizer state)
    model.save(DRIVE_SAVE_MODEL_PATH)
    print(f"Model saved successfully to Google Drive as {DRIVE_SAVE_MODEL_PATH}")
    # Verify if the file exists
    if os.path.exists(DRIVE_SAVE_MODEL_PATH):
        print(f"Verified: File '{DRIVE_SAVE_MODEL_PATH}' exists in Google Drive.")
    else:
        print(f"Warning: File '{DRIVE_SAVE_MODEL_PATH}' was not found after saving.")

except Exception as e:
    print(f"Error saving model to Google Drive: {e}")

# --- 10. Make Predictions on the Test Set ---
print("\nMaking predictions on the test set...")
test_predictions = model.predict(X_test_scaled).flatten()

# --- 11. Compare Predictions with Actual Values (Test Set) ---
comparison_df = pd.DataFrame({'Actual SOC': y_test.reset_index(drop=True),
                              'Predicted SOC': test_predictions})
print("\nComparison of Actual vs. Predicted SOC on the Test Set (First 15 samples):")
print(comparison_df.head(15).to_markdown(index=False, numalign="left", stralign="left"))

# --- 12. Visualize Training History (Optional) ---
# (Plotting code remains the same)
print("\nPlotting training history...")
history_df = pd.DataFrame(history.history)
history_df['epoch'] = history.epoch

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_df['epoch'], history_df['mean_absolute_error'], label='Train MAE')
plt.plot(history_df['epoch'], history_df['val_mean_absolute_error'], label = 'Val MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error (SOC %)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_df['epoch'], history_df['loss'], label='Train Loss (MSE)')
plt.plot(history_df['epoch'], history_df['val_loss'], label = 'Val Loss (MSE)')
plt.title('Training and Validation Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# --- 13. Example Prediction with Saved Model from Drive (Demonstration) ---
print("\n--- Loading and Using Saved Model & Scaler from Drive (Demonstration) ---")

# Check if both model and scaler files exist in Drive
model_exists = os.path.exists(DRIVE_SAVE_MODEL_PATH)
scaler_exists = os.path.exists(DRIVE_SAVE_SCALER_PATH)

if model_exists and scaler_exists:
    print(f"Loading model from: {DRIVE_SAVE_MODEL_PATH}")
    print(f"Loading scaler from: {DRIVE_SAVE_SCALER_PATH}")
    try:
        # Load the model
        loaded_model = tf.keras.models.load_model(DRIVE_SAVE_MODEL_PATH)
        # Load the scaler
        loaded_scaler = joblib.load(DRIVE_SAVE_SCALER_PATH)
        print("Model and scaler loaded successfully from Google Drive.")

        # Prepare sample data
        sample_data = np.array([
            [55.0, -1.0],
            [50.0, -10.0],
            [56.0, 0.0]
        ])

        # Scale the sample data using the LOADED scaler
        sample_data_scaled = loaded_scaler.transform(sample_data)

        # Predict using the loaded model
        loaded_predictions = loaded_model.predict(sample_data_scaled)
        print("\nPredictions using the loaded model and scaler:")
        for i in range(len(sample_data)):
            print(f"Input (Voltage, Current): {sample_data[i]} -> Predicted SOC: {loaded_predictions[i][0]:.2f}%")

    except Exception as e:
        print(f"Error loading or using the saved model/scaler from Drive: {e}")
else:
    if not model_exists:
        print(f"Saved model file '{DRIVE_SAVE_MODEL_PATH}' not found in Google Drive.")
    if not scaler_exists:
        print(f"Saved scaler file '{DRIVE_SAVE_SCALER_PATH}' not found in Google Drive.")
    print("Skipping loading demonstration.")

print("\nScript finished.")
