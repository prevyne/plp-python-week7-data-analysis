#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests # To download the zip file
import zipfile # To handle the zip file
import io      # To handle the downloaded bytes in memory
import ssl     # Sometimes needed for fetching URL data

print("--- Loading Data from UCI Repository (using ZIP archive) ---")

# --- Task 1: Load and Explore the Dataset ---

# Allow fetching data from HTTPS sources 
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass # Legacy Python versions may not have this attribute
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Define the URL for the ZIP file containing the datasets
zip_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip'
csv_file_in_zip = 'student-mat.csv' # We want the Math performance data

print(f"Attempting to download ZIP archive from: {zip_url}")

try:
    # Download the ZIP file content
    response = requests.get(zip_url)
    response.raise_for_status() # Check if the download was successful (status code 200)
    print("ZIP archive downloaded successfully.")

    # Open the ZIP file from the downloaded bytes
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        print(f"Extracting '{csv_file_in_zip}' from archive...")
        # Check if the specific CSV file exists in the zip archive
        if csv_file_in_zip in z.namelist():
            # Read the specific CSV file from the ZIP archive using pandas
            # Ensure to specify the separator ';'
            with z.open(csv_file_in_zip) as csvfile:
                 df = pd.read_csv(csvfile, sep=';')
            print(f"Successfully loaded '{csv_file_in_zip}' into DataFrame.")
            print("Using the 'student-mat.csv' (Math course performance) dataset.")
        else:
            print(f"Error: '{csv_file_in_zip}' not found inside the downloaded ZIP archive.")
            print(f"Files found: {z.namelist()}")
            exit()

except requests.exceptions.RequestException as e:
    print(f"An error occurred while downloading the ZIP file: {e}")
    print("Please check your internet connection and the URL.")
    exit()
except zipfile.BadZipFile:
    print("Error: Failed to open the downloaded file as a ZIP archive. It might be corrupted or incomplete.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    exit()

print("\n--- Task 1: Explore ---")

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Explore the structure: data types and non-null counts
print("\nDataset Information (Data Types, Non-Null Counts):")
df.info()

# Check for missing values before cleaning
print("\nMissing values per column (before cleaning):")
print(df.isnull().sum())

# --- Data Cleaning ---
print(f"\nShape before dropping NA (if any): {df.shape}")
df_cleaned = df.dropna()
print(f"Shape after dropping NA: {df_cleaned.shape}")

# Check missing values again after cleaning
print("\nMissing values per column (after cleaning):")
print(df_cleaned.isnull().sum())

# Use the cleaned dataframe onwards
df = df_cleaned

# --- Task 2: Basic Data Analysis ---

print("\n--- Task 2: Basic Data Analysis ---")

# Compute basic statistics for numerical columns
print("\nBasic statistics for numerical columns:")
print(df.select_dtypes(include=np.number).describe())

# Perform groupings
categorical_col_name = 'Mjob' # Mother's job
numerical_col_name = 'G3'     # Final grade

group_means = None # Initialize group_means
if categorical_col_name in df.columns and numerical_col_name in df.columns:
    if pd.api.types.is_numeric_dtype(df[numerical_col_name]):
        print(f"\nMean of '{numerical_col_name}' (Final Grade) grouped by '{categorical_col_name}' (Mother's Job):")
        group_means = df.groupby(categorical_col_name)[numerical_col_name].mean()
        print(group_means)
        print("\nPotential Findings:")
        print(f"* Examine the descriptive statistics for grades (G1, G2, G3), absences, study time etc.")
        print(f"* Does the average final grade (G3) seem to differ based on the mother's job?")
    else:
         print(f"\nWarning: Column '{numerical_col_name}' is not numeric. Cannot calculate mean for grouping.")
else:
    print(f"\nWarning: Cannot perform grouping. Check if '{categorical_col_name}' and '{numerical_col_name}' exist.")


# --- Task 3: Data Visualization ---

print("\n--- Task 3: Data Visualization ---")
print("Generating plots...")
sns.set_style("whitegrid")

# Plot 0: Line Chart - SKIPPED
print("\nSkipped Line Chart: because it does not have a suitable time-series column.")

# Plot 1: Bar Chart (Avg Final Grade per Mother's Job)
if group_means is not None:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=group_means.index, y=group_means.values, palette='viridis', order=group_means.sort_values(ascending=False).index)
    plt.title(f'Average Final Grade (G3) by Mother\'s Job (Mjob)')
    plt.xlabel("Mother's Job")
    plt.ylabel('Average Final Grade (G3)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print(f"\nSkipping bar chart: Grouping by '{categorical_col_name}' did not produce results.")

# Plot 2: Histogram (Distribution of Final Grades)
hist_col = 'G3'
if hist_col in df.columns:
    if pd.api.types.is_numeric_dtype(df[hist_col]):
        plt.figure(figsize=(8, 5))
        sns.histplot(df[hist_col], kde=True, bins=10)
        plt.title(f'Distribution of Final Grades (G3)')
        plt.xlabel('Final Grade (G3)')
        plt.ylabel('Number of Students')
        plt.tight_layout()
        plt.show()
    else:
        print(f"\nSkipping histogram: Column '{hist_col}' is not numeric.")
else:
    print(f"\nSkipping histogram: Check if '{hist_col}' exists.")

# Plot 3: Scatter Plot (Study Time vs Final Grade)
scatter_col_1 = 'studytime' # Weekly study time
scatter_col_2 = 'G3'        # Final grade
if scatter_col_1 in df.columns and scatter_col_2 in df.columns:
    if pd.api.types.is_numeric_dtype(df[scatter_col_1]) and pd.api.types.is_numeric_dtype(df[scatter_col_2]):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=scatter_col_1, y=scatter_col_2, data=df, alpha=0.6)
        plt.title(f'Relationship between Study Time and Final Grade (G3)')
        plt.xlabel('Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)')
        plt.ylabel('Final Grade (G3)')
        plt.xticks([1, 2, 3, 4])
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print(f"\nSkipping scatter plot: One or both columns ('{scatter_col_1}', '{scatter_col_2}') are not numeric.")
else:
    print(f"\nSkipping scatter plot: Check if '{scatter_col_1}' and '{scatter_col_2}' exist.")

# PLot 4: An additional plot: Box Plot (Grades by Internet Access)
box_cat_col = 'internet' # Categorical: yes/no
box_num_col = 'G3'       # Numerical: Final grade
if box_cat_col in df.columns and box_num_col in df.columns:
     if pd.api.types.is_numeric_dtype(df[box_num_col]):
        plt.figure(figsize=(7, 5))
        sns.boxplot(x=box_cat_col, y=box_num_col, data=df, palette='pastel')
        plt.title(f'Final Grades (G3) by Home Internet Access')
        plt.xlabel('Internet Access at Home')
        plt.ylabel('Final Grade (G3)')
        plt.tight_layout()
        plt.show()
     else:
         print(f"\nSkipping box plot: Numerical column '{box_num_col}' is not numeric.")
else:
    print(f"\nSkipping box plot: Check if '{box_cat_col}' and '{box_num_col}' exist.")

print("\nAnalysis and Visualization Complete.")