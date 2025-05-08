# 🔋 Electric Vehicle Battery SoC Prediction Using ANN

This project is an educational machine learning pipeline designed to predict the **State of Charge (SoC)** of an electric vehicle battery using artificial neural networks (ANN). The model is trained on a dataset containing electrical parameters such as **current**, **voltage**, and **timestamps**, simulating real-world EV battery data.

## 📚 Project Overview

This project follows a complete data science lifecycle:
1. **Data Collection**
2. **Data Preprocessing & Cleaning**
3. **Feature Engineering**
4. **Model Building with ANN**
5. **Model Evaluation**
6. **Prediction & Visualization**

---

## 📁 Dataset Description

The dataset includes:
- `Timestamp`: Time of data recording
- `Voltage (V)`: Battery voltage over time
- `Current (A)`: Current flow at each timestamp
- (Optional) `Temperature`: Environmental or internal cell temperature
- `State of Charge (SoC)`: The target variable (label)

The dataset simulates battery operation under different loading conditions.

---

## 🧹 Data Preprocessing

- Handled missing values
- Removed outliers using standard deviation thresholds
- Normalized current and voltage features
- Converted timestamps to
