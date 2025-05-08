"""
File Name: Data_Mining_Code.py
Developers: Group 1 Members
Date: 01/03/2025
Task: Data Cleaning for EV Features

Description:
This script is designed to clean and preprocess a dataset containing electrical measurements related to 
Electric Vehicles (EV), such as State of Charge (SOC), Voltage, and Current. The cleaning process is crucial 
for ensuring the dataset's consistency and reliability before performing any analysis or modeling. This includes 
the removal of duplicates, handling missing values, filtering out invalid SOC values, and ensuring proper 
formatting of timestamps.

Main Features:
- Verifies the existence of all required columns (timestamp, SOC, voltage, and current)
- Removes duplicate rows based on timestamp to ensure each data point represents a unique moment
- Filters out invalid SOC values (SOC must be in the range of 0–100)
- Fills missing values in numerical columns (SOC, voltage, current) using the column mean
- Saves the cleaned dataset for further analysis or predictive modeling
- Plots the relationship between SOC and voltage using a scatter plot with a trend line for visual insight into the data

Usage:
1. Place the dataset (CSV format) in the same directory as the script or update the file path.
2. Run this script to clean the dataset and prepare it for further analysis or modeling.
3. The cleaned dataset will be saved as 'Cleaning_Data.csv' for future use.
4. The script also generates a plot showing the linear relationship between SOC and Voltage, with a trend line.

Note:
Make sure the input CSV contains columns named 'timestamp', 'soc', 'voltage', and 'current'. 
If any required columns are missing, an error will be raised, and the script will terminate.

"""
import pandas as pd
import matplotlib.pyplot as plot_data
import numpy as np

# Load the dataset from the CSV file
Raw_Data = pd.read_csv('Dataset_Conf.csv')

# Drop index column if it exists ('Unnamed: 0' from previously saved CSV files)
if 'Unnamed: 0' in Raw_Data.columns:
    # Drop unnecessary index column
    Raw_Data.drop(columns=['Unnamed: 0'], inplace=True)  

# Validate that required columns are present in the dataset
required_cols = ['timestamp', 'soc', 'voltage', 'current']
for col in required_cols:
    # Raise an error if any required column is missing
    if col not in Raw_Data.columns:
        raise ValueError(f"Missing required column: '{col}' in the dataset.")  

# Display the features (columns) available in the dataset
print('---------------------------- Features in Dataset --------------------------------')
# List all columns in the dataset
print("Features in Dataset:", Raw_Data.columns.tolist())  
print('----------------------------------------------------------------------------------\n')

# Print initial row count before any cleaning
print('-------------------------- Initial Number of Rows -------------------------------')
# Print number of rows before cleaning
print("Rows in Dataset (before trimming):", Raw_Data.shape[0])  
print('----------------------------------------------------------------------------------\n')

# Convert 'timestamp' to datetime format and drop any rows where conversion fails
# Convert 'timestamp' to datetime, invalid rows become NaT
Raw_Data['timestamp'] = pd.to_datetime(Raw_Data['timestamp'], errors='coerce') 
# Remove rows with invalid timestamps
Raw_Data.dropna(subset=['timestamp'], inplace=True)  

# Remove rows with duplicate combinations of timestamp, SOC, voltage, and current
print('------------------------ Duplicate Rows in Dataset ------------------------------')
duplicates_total = Raw_Data.duplicated().sum()  # Count total duplicates in the dataset
print("Duplicates found in Dataset:", duplicates_total)  # Print the number of duplicates
# Drop exact duplicates based on all four columns
Raw_Data.drop_duplicates(subset=['timestamp', 'soc', 'voltage', 'current'], inplace=True) 
print("Duplicates removed from Dataset.")
print('----------------------------------------------------------------------------------\n')

# Filter out rows where SOC is outside the valid range (0 to 100)
print('-------------------------- SOC Value Filtering ----------------------------------')
before_soc_filter = Raw_Data.shape[0]  # Record the number of rows before filtering
Raw_Data = Raw_Data[(Raw_Data['soc'] >= 0) & (Raw_Data['soc'] <= 100)]  # Keep only valid SOC values
after_soc_filter = Raw_Data.shape[0]  # Record the number of rows after filtering
# Print the number of rows removed due to invalid SOC
print(f"Rows removed due to invalid SOC values: {before_soc_filter - after_soc_filter}") 
print('----------------------------------------------------------------------------------\n')

# Fill any missing values in numeric columns with the column mean
print('------------------------ Missing Values in Dataset ------------------------------')
 # Print total number of missing values before filling
print("Missing values in Dataset before fill:", Raw_Data.isnull().sum().sum()) 
 # Fill missing values in numeric columns with the column mean
Raw_Data.fillna(Raw_Data.mean(numeric_only=True), inplace=True) 
print("Missing values filled with column mean.")
print('----------------------------------------------------------------------------------\n')

# Save the cleaned dataset to a new CSV file for future analysis
Raw_Data.reset_index(drop=True, inplace=True)  # Reset the index after cleaning
Raw_Data.to_csv("Cleaning_Data.csv", index=False)  # Save the cleaned data to 'Cleaning_Data.csv'

# Print the final number of rows after cleaning
print('------------------- Final Number of Rows After Cleaning -------------------------')
print("Rows in Cleaned Dataset:", Raw_Data.shape[0])  # Print the number of rows after cleaning
print('----------------------------------------------------------------------------------\n')

# Plot a scatter plot to visualize the relationship between SOC and Voltage with a trend line
if 'soc' in Raw_Data.columns and 'voltage' in Raw_Data.columns:
    plot_data.figure(figsize=(10, 6))  # Create a figure with specified size
    plot_data.scatter(Raw_Data['soc'], Raw_Data['voltage'], color='blue', alpha=0.6, s=15, label='Data Points')  # Scatter plot of SOC vs Voltage

    # Add a trend line using linear regression
    z = np.polyfit(Raw_Data['soc'], Raw_Data['voltage'], 1)  # Fit a line to the data (linear regression)
    p = np.poly1d(z)  # Create a polynomial object for the trend line
    plot_data.plot(Raw_Data['soc'], p(Raw_Data['soc']), color='red', linestyle='--', linewidth=3, label='Trend Line')  # Plot the trend line

    # Add labels and title to the plot
    plot_data.xlabel('State of Charge (SOC) [%]', fontsize=14)  # X-axis label
    plot_data.ylabel('Voltage [V]', fontsize=14)  # Y-axis label
    plot_data.title('SOC vs Voltage Relationship', fontsize=16)  # Title of the plot
    plot_data.grid(True, linestyle='--', linewidth=0.7)  # Grid settings
    plot_data.legend(loc='best', fontsize=12)  # Add legend
    plot_data.tight_layout()  # Adjust the layout to prevent clipping
    plot_data.show()  # Display the plot
else:
    print("Columns 'soc' and 'voltage' not found in cleaned data.")  # In case the required columns are not found

# Print a summary message indicating the cleaning process is complete
print("\n Dataset cleaning completed successfully. Cleaned file saved as 'Cleaning_Data.csv'.")
