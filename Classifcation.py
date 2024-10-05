from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import csv
from Scalers import *  # Importing the custom scaler functions
from sklearn.impute import SimpleImputer
def main():
    # Read the .data file
    file_path = "./breast_cancer/breast-cancer.data"
    column_names = [
        "recurrence_status", "age", "menopause", "tumor_size", "inv_nodes",
        "node_caps", "deg_malig", "breast", "breast_quad", "irradiation"
    ]
    data = []
    with open(file_path, "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        for row in csv_reader:
            if row:
                data.append(row)

    # Encode Data
    encodings = {
        "recurrence_status": {"no-recurrence-events": 0, "recurrence-events": 1},
        "age": {"10-19": 14.5, "20-29": 24.5, "30-39": 34.5, "40-49": 44.5, "50-59": 54.5, "60-69": 64.5, "70-79": 74.5},
        "menopause": {"lt40": 0, "ge40": 1, "premeno": 2},
        "tumor_size": {"0-4": 0, "5-9": 1, "10-14": 2, "15-19": 3, "20-24": 4, "25-29": 5, "30-34": 6, "35-39": 7, "40-44": 8, "45-49": 9, "50-54": 10},
        "inv_nodes": {"0-2": 0, "3-5": 1, "6-8": 2, "9-11": 3, "12-14": 4, "15-17": 5, "18-20": 6, "21-23": 7, "24-26": 8},
        "node_caps": {"no": 0, "yes": 1, "?": None},  # Use None for missing values
        "deg_malig": {"1": 1, "2": 2, "3": 3},
        "breast": {"left": 0, "right": 1},
        "breast_quad": {"left_up": 0, "left_low": 1, "right_up": 2, "right_low": 3, "central": 4, "?": None},  # Use None
        "irradiation": {"no": 0, "yes": 1}
    }

    # Ensure value is valid before applying .strip()
    for row in data:
        for key, value in zip(column_names, row):
            if value is not None:  # Check if value is not None before stripping
                row[column_names.index(key)] = encodings[key].get(value.strip(), None)  # Use None for invalid entries

    # Convert to Lists of Features and Labels
    X = [row[1:] for row in data]
    y = [row[0] for row in data]

    # Step 1: Handle Missing Values Using Imputer
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = imputer.fit_transform(X)

    # Standard Scaler: Subtract the mean and divide by standard deviation
    def apply_standard_scaler(feature):
        mean = sum(feature) / len(feature)
        variance = sum((x - mean) ** 2 for x in feature) / len(feature)
        std_dev = math.sqrt(variance)
        return [(x - mean) / std_dev for x in feature]

    # Min-Max Scaler: Rescale the data to range between 0 and 1
    def apply_min_max_scaler(feature):
        min_val = min(feature)
        max_val = max(feature)
        return [(x - min_val) / (max_val - min_val) for x in feature]

    # Robust Scaler: Rescale the data using median and IQR (interquartile range)
    def apply_robust_scaler(feature):
        med = median(feature)
        q1 = median([x for x in feature if x < med])
        q3 = median([x for x in feature if x > med])
        iqr = q3 - q1
    
    def apply_scalers(X, column_names):
        scaled_X = []
        for i, feature in enumerate(zip(*X)):  # Transpose to get features column-wise
            if column_names[i] in ['age', 'tumor_size', 'inv_nodes']:
                print(f"Applying Standard Scaler to feature: {column_names[i]}")
                scaled_feature = apply_standard_scaler(feature)
            elif column_names[i] in ['deg_malig', 'irradiation']:
                print(f"Applying Min-Max Scaler to feature: {column_names[i]}")
                scaled_feature = apply_min_max_scaler(feature)
            elif column_names[i] in ['recurrence_status', 'node_caps', 'breast_quad']:
                print(f"Applying Robust Scaler to feature: {column_names[i]}")
                scaled_feature = apply_robust_scaler(feature)
            else:
                scaled_feature = feature  # No scaling applied
            scaled_X.append(scaled_feature)
    
    # Transpose back to match the original input structure
        return list(zip(*scaled_X)) 

    X_scaled = apply_scalers(X, column_names[1:])
    