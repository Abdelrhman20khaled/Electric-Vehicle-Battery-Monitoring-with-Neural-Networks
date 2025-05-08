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
- Converted timestamps to datetime format and extracted time-based features (if applicable)

---

## 🧠 Model: Artificial Neural Network (ANN)

- Input features: Voltage, Current, and Time-derived metrics
- Output: Predicted SoC (0–100%)
- Architecture:
  - Input layer with normalized features
  - 2–3 hidden layers with ReLU activation
  - Output layer with linear activation for regression

---

## 📊 Evaluation Metrics

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**
- Learning curves and SoC prediction.

---

## 🎯 Project Goals

- Learn end-to-end machine learning workflow in a real-world scenario
- Apply ANN for a regression problem in the energy domain
- Build intuition around battery behavior and SoC estimation

---

## 🔧 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib / Seaborn

---

## 📌 Future Work

- Integrate additional sensor data such as temperature, current, and voltage for improved prediction accuracy

- Design a battery monitoring hardware circuit to collect real-time data

- Develop a graphical user interface (GUI) using Tkinter in Python for user-friendly interaction and visualizatio

---

## 🧑‍💻 Author

**Abdelrahman Khaled**  
[LinkedIn]([https://linkedin.com/in/YOUR-LINKEDIN](https://www.linkedin.com/in/abdelrahman-khaled-12a8b6242/))

---

> ⚠️ _This project is for educational purposes only and does not represent a commercial or production-grade battery monitoring solution._
