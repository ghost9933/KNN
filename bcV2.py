import math
import time
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import csv
from CustomKNN import CustomKNN
from Scalers import *  # Custom scaling functions

def replace_nan_and_question(X):
    for i in range(len(X)):
        for j in range(len(X[i])):
            if X[i][j] == '?' or (isinstance(X[i][j], float) and math.isnan(X[i][j])):
                X[i][j] = None
    return X

def replace_none_with_most_frequent(X):
    X_transposed = list(zip(*X))
    for i, feature in enumerate(X_transposed):
        non_none_values = [x for x in feature if x is not None]
        most_frequent_value = max(set(non_none_values), key=non_none_values.count)
        X_transposed[i] = [most_frequent_value if x is None else x for x in feature]
    return list(zip(*X_transposed))

def apply_custom_scaling(X):
    age = [row[1] for row in X]
    tumor_size = [row[3] for row in X]
    inv_nodes = [row[4] for row in X]
    deg_malig = [row[6] for row in X]

    age_tumor_size = [[age[i], tumor_size[i]] for i in range(len(X))]
    inv_nodes_deg_malig = [[inv_nodes[i], deg_malig[i]] for i in range(len(X))]

    min_vals, max_vals = min_max_scaler_fit(age_tumor_size)
    age_tumor_size_scaled = min_max_scaler_transform(age_tumor_size, min_vals, max_vals)

    mean, std = standard_scaler_fit(inv_nodes_deg_malig)
    inv_nodes_deg_malig_scaled = standard_scaler_transform(inv_nodes_deg_malig, mean, std)

    X_scaled = []
    for i in range(len(X)):
        unscaled_part = [X[i][0], X[i][2], X[i][5], X[i][7], X[i][8], X[i][9]]
        scaled_part = list(age_tumor_size_scaled[i]) + list(inv_nodes_deg_malig_scaled[i])
        X_scaled.append([unscaled_part[0], scaled_part[0], unscaled_part[1], scaled_part[1], unscaled_part[2], unscaled_part[3], unscaled_part[4], unscaled_part[5]])

    return X_scaled

def load_and_encode_data(file_path, encodings, column_names):
    data = []
    with open(file_path, "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        for row in csv_reader:
            if row:
                encoded_row = [encodings[key][value.strip()] for key, value in zip(column_names, row)]
                data.append(encoded_row)
    return data

def main():
    column_names = [
        "recurrence_status", "age", "menopause", "tumor_size", "inv_nodes",
        "node_caps", "deg_malig", "breast", "breast_quad", "irradiation"
    ]
    encodings = {
        "recurrence_status": {"no-recurrence-events": 0, "recurrence-events": 1},
        "age": {"10-19": 0, "20-29": 1, "30-39": 2, "40-49": 3, "50-59": 4, "60-69": 5, "70-79": 6},
        "menopause": {"lt40": 0, "ge40": 1, "premeno": 2},
        "tumor_size": {"0-4": 0, "5-9": 1, "10-14": 2, "15-19": 3, "20-24": 4, "25-29": 5, "30-34": 6, "35-39": 7, "40-44": 8, "45-49": 9, "50-54": 10},
        "inv_nodes": {"0-2": 0, "3-5": 1, "6-8": 2, "9-11": 3, "12-14": 4, "15-17": 5, "18-20": 6, "21-23": 7, "24-26": 8},
        "node_caps": {"no": 0, "yes": 1, "?": None},
        "deg_malig": {"1": 1, "2": 2, "3": 3},
        "breast": {"left": 0, "right": 1},
        "breast_quad": {"left_up": 0, "left_low": 1, "right_up": 2, "right_low": 3, "central": 4, "?": None},
        "irradiation": {"no": 0, "yes": 1}
    }

    data = load_and_encode_data("./breast_cancer/breast-cancer.data", encodings, column_names)
    X = [row[1:] for row in data]
    y = [row[0] for row in data]

    X_cleaned = replace_nan_and_question(X)
    X_imputed = replace_none_with_most_frequent(X_cleaned)
    X_scaled = apply_custom_scaling(X_imputed)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_neighbors': list(range(1, 21)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'chebyshev']
    }
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_knn = grid_search.best_estimator_

    y_pred = best_knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best Parameters (scikit-learn KNN): {grid_search.best_params_}")
    print(f"Accuracy of Best scikit-learn KNN model: {accuracy:.2f}")
    print("\nClassification Report (scikit-learn KNN):")
    print(classification_report(y_test, y_pred))

    start_time = time.time()
    knn_custom = CustomKNN(n=3, task='classification', weights='distance', metric='euclidean')
    knn_custom.fit(X_train, y_train)
    y_pred_custom = knn_custom.predict(X_test)
    custom_knn_time = time.time() - start_time
    accuracy_custom = accuracy_score(y_test, y_pred_custom)
    print(f"\nAccuracy of CustomKNN model: {accuracy_custom:.2f}")
    print(f"CustomKNN Time Taken: {custom_knn_time:.2f} seconds")
    print("\nClassification Report (CustomKNN):")
    print(classification_report(y_test, y_pred_custom))

    print("\n--- Comparison of scikit-learn KNN vs CustomKNN ---")
    print(f"scikit-learn KNN Accuracy: {accuracy:.2f}")
    print(f"CustomKNN Accuracy: {accuracy_custom:.2f}")
    print(f"CustomKNN Time Taken: {custom_knn_time:.2f} seconds")

if __name__ == "__main__":
    main()
