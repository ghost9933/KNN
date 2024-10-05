from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import time
import psutil
from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import csv
from myCustomKNN import CustomKNN 
from Scalers import * # Importing the CustomKNN class

def main():
    # Fetch Dataset
    breast_cancer = fetch_ucirepo(id=14)
    X = breast_cancer.data.features
    y = breast_cancer.data.targets

    # Read the .data File
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
        "age": {"10-19": 0, "20-29": 1, "30-39": 2, "40-49": 3, "50-59": 4, "60-69": 5, "70-79": 6},
        "menopause": {"lt40": 0, "ge40": 1, "premeno": 2},
        "tumor_size": {"0-4": 0, "5-9": 1, "10-14": 2, "15-19": 3, "20-24": 4, "25-29": 5, "30-34": 6, "35-39": 7, "40-44": 8, "45-49": 9, "50-54": 10},
        "inv_nodes": {"0-2": 0, "3-5": 1, "6-8": 2, "9-11": 3, "12-14": 4, "15-17": 5, "18-20": 6, "21-23": 7, "24-26": 8},
        "node_caps": {"no": 0, "yes": 1, "?": np.nan},
        "deg_malig": {"1": 1, "2": 2, "3": 3},
        "breast": {"left": 0, "right": 1},
        "breast_quad": {"left_up": 0, "left_low": 1, "right_up": 2, "right_low": 3, "central": 4, "?": np.nan},
        "irradiation": {"no": 0, "yes": 1}
    }
    for row in data:
        for key, value in zip(column_names, row):
            row[column_names.index(key)] = encodings[key][value.strip()]

    # Convert to Lists of Features and Labels
    X = [row[1:] for row in data]
    y = [row[0] for row in data]


    def replace_nan_and_question(X):
        for i in range(len(X)):
            for j in range(len(X[i])):
                if X[i][j] == '?' or (isinstance(X[i][j], float) and math.isnan(X[i][j])):
                    X[i][j] = None
        return X

    # Function to replace None with the most frequent value in each column
    def replace_none_with_most_frequent(X):
        # Transpose the dataset to work column-wise (each feature)
        X_transposed = list(zip(*X))
        
        # Iterate through each column (feature)
        for i, feature in enumerate(X_transposed):
            # Find the most frequent value, ignoring None
            non_none_values = [x for x in feature if x is not None]
            most_frequent_value = max(set(non_none_values), key=non_none_values.count)

            # Replace None values with the most frequent value
            X_transposed[i] = [most_frequent_value if x is None else x for x in feature]
        
        # Transpose back to the original structure
        return list(zip(*X_transposed))

    # Step 1: Replace NaN and ? with None
    X_cleaned = replace_nan_and_question(X)

    # Step 2: Replace None values with the most frequent value in each column
    X_imputed = replace_none_with_most_frequent(X_cleaned)
    X_imputed = replace_none_with_most_frequent(X)

    # Step 2: Scale Features Using Standard Scaler
    mean, std = standard_scaler_fit(X_imputed)  # Fit the scaler (calculate mean and std for each column)
    X_scaled = standard_scaler_transform(X_imputed, mean, std)
    minV,maxV=min_max_scaler_fit(X_imputed)
    X_scaled=min_max_scaler_transform(X_imputed,min_vals=minV,max_vals=maxV)


    

   

    # Step 4: Split Data into Training and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # # Step 5: Hyperparameter Tuning for KNN (scikit-learn)
    param_grid = {
        'n_neighbors': list(range(1, 21)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'chebyshev','hamming','cosine']
    }
    
    print("Tuning Scikit-learn KNN with GridSearchCV...")
    grid_search_sklearn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    start_time_sklearn = time.time()
    grid_search_sklearn.fit(X_train, y_train)
    end_time_sklearn = time.time() - start_time_sklearn
    best_knn_sklearn = grid_search_sklearn.best_estimator_

    # Evaluate the best Scikit-learn KNN model
    y_pred_sklearn = best_knn_sklearn.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

    print(f"\nBest Parameters (scikit-learn KNN): {grid_search_sklearn.best_params_}")
    print(f"Accuracy of Best scikit-learn KNN model: {accuracy_sklearn:.2f}")
    print(f"Time taken by Scikit-learn KNN: {end_time_sklearn:.2f} seconds")
    print("\nClassification Report (Scikit-learn KNN):")
    print(classification_report(y_test, y_pred_sklearn))

    # Step 2: Hyperparameter Tuning for CustomKNN using Grid Search (manual implementation)
    def custom_knn_grid_search(X_train, y_train, X_test, y_test, param_grid):
        best_params = None
        best_accuracy = 0
        best_model = None
        best_time = 0

        for n_neighbors in param_grid['n_neighbors']:
            for weights in param_grid['weights']:
                for metric in param_grid['metric']:
                    knn_custom = CustomKNN(n=n_neighbors, task='classification', weights=weights, metric=metric)
                    start_time = time.time()
                    knn_custom.fit(X_train, y_train)
                    y_pred_custom = knn_custom.predict(X_test)
                    end_time = time.time() - start_time

                    accuracy_custom = accuracy_score(y_test, y_pred_custom)

                    if accuracy_custom > best_accuracy:
                        best_accuracy = accuracy_custom
                        best_params = {'n_neighbors': n_neighbors, 'weights': weights, 'metric': metric}
                        best_model = knn_custom
                        best_time = end_time

        return best_model, best_params, best_accuracy, best_time

    # Run grid search for CustomKNN
    print("\nTuning CustomKNN...")
    best_knn_custom, best_params_custom, accuracy_custom, custom_knn_time = custom_knn_grid_search(X_train, y_train, X_test, y_test, param_grid)

    print(f"\nBest Parameters (CustomKNN): {best_params_custom}")
    print(f"Accuracy of Best CustomKNN model: {accuracy_custom:.2f}")
    print(f"Time taken by CustomKNN: {custom_knn_time:.2f} seconds")
    y_pred_custom = best_knn_custom.predict(X_test)
    print("\nClassification Report (CustomKNN):")
    print(classification_report(y_test, y_pred_custom))

    print('Sikits for same parameters')
    n_neighbors = best_params_custom['n_neighbors']
    weights = best_params_custom['weights']
    metric = best_params_custom['metric']
    print(f"\nTesting Scikit-learn KNN with CustomKNN best parameters: {best_params_custom}")
    start_time_sklearn = time.time()

    # Initialize Scikit-learn's KNeighborsClassifier with the best parameters from CustomKNN
    knn_sklearn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)

    # Fit the model using the same training data
    knn_sklearn.fit(X_train, y_train)

    # Predict on the test set
    y_pred_sklearn = knn_sklearn.predict(X_test)

    # Calculate the runtime
    end_time_sklearn = time.time() - start_time_sklearn

    # Calculate accuracy and other metrics for Scikit-learn KNN
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"\nAccuracy (Scikit-learn KNN with CustomKNN best parameters): {accuracy_sklearn:.2f}")
    print(f"Time taken (Scikit-learn KNN): {end_time_sklearn:.2f} seconds")

    # Classification report for Scikit-learn KNN
    print("\nClassification Report (Scikit-learn KNN with CustomKNN best parameters):")
    print(classification_report(y_test, y_pred_sklearn))


    # Step 3: Compare the performance of Scikit-learn KNN and CustomKNN
    print("\n--- Performance Comparison: Scikit-learn KNN vs CustomKNN ---")
    print(f"Scikit-learn KNN Accuracy: {accuracy_sklearn:.2f}")
    print(f"CustomKNN Accuracy: {accuracy_custom:.2f}")
    print(f"Time taken by Scikit-learn KNN: {end_time_sklearn:.2f} seconds")
    print(f"Time taken by CustomKNN: {custom_knn_time:.2f} seconds")


if __name__ == "__main__":
    main()