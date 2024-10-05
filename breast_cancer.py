import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from myCustomKNN import CustomKNN
from Scalers import *  # Assuming you have the Scalers defined for standard scaling and min-max scaling
import csv
from ucimlrepo import fetch_ucirepo

def process_dataset(file_path, column_names, encodings):
    data = []
    with open(file_path, "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        for row in csv_reader:
            if row:
                data.append(row)

    # Encode Data
    for row in data:
        for key, value in zip(column_names, row):
            row[column_names.index(key)] = encodings[key][value.strip()]

    # Convert to lists of features (X) and labels (y)
    X = [row[1:] for row in data]
    y = [row[0] for row in data]

    return X, y


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


def main(file_path, test_size=0.2, random_state=42, param_grid=None):
    if param_grid is None:
        param_grid = {
            'n_neighbors': list(range(1, 21)),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'chebyshev', 'hamming', 'cosine']
        }

    

    # Split Data into Training and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

   
    # CustomKNN Grid Search
    print("\nTuning CustomKNN...")
    best_knn_custom, best_params_custom, accuracy_custom, custom_knn_time = custom_knn_grid_search(
        X_train, y_train, X_test, y_test, param_grid
    )

    print(f"\nBest Parameters (CustomKNN): {best_params_custom}")
    print(f"Accuracy of Best CustomKNN model: {accuracy_custom:.2f}")
    print(f"Time taken by CustomKNN: {custom_knn_time:.2f} seconds")
    y_pred_custom = best_knn_custom.predict(X_test)
    print("\nClassification Report (CustomKNN):")
    print(classification_report(y_test, y_pred_custom))


    # Extract the best parameters from CustomKNN
    n_neighbors, weights, metric = best_params_custom.values()

    # Test Scikit-learn KNN with the same hyperparameters as CustomKNN
    print(f"\nTesting Scikit-learn KNN with CustomKNN best parameters: {best_params_custom}")
    start_time_sklearn = time.time()

    # Initialize and fit Scikit-learn KNN
    knn_sklearn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
    knn_sklearn.fit(X_train, y_train)

    # Predict and evaluate
    y_pred_sklearn = knn_sklearn.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    end_time_sklearn = time.time() - start_time_sklearn

    # Display results
    print(f"\nAccuracy (Scikit-learn KNN with CustomKNN best parameters): {accuracy_sklearn:.2f}")
    print(f"Time taken (Scikit-learn KNN): {end_time_sklearn:.2f} seconds")
    print("\nClassification Report (Scikit-learn KNN):")
    print(classification_report(y_test, y_pred_sklearn))

    # Performance comparison
    print("\n--- Performance Comparison ---")
    print(f"CustomKNN Accuracy: {accuracy_custom:.2f}")
    print(f"Scikit-learn KNN Accuracy: {accuracy_sklearn:.2f}")
    print(f"CustomKNN Time Taken: {custom_knn_time:.2f} seconds")
    print(f"Scikit-learn KNN Time Taken: {end_time_sklearn:.2f} seconds")


    # Step 3: Compare the performance of Scikit-learn KNN and CustomKNN
    print("\n--- Performance Comparison: Scikit-learn KNN vs CustomKNN ---")
    print(f"Scikit-learn KNN Accuracy: {accuracy_sklearn:.2f}")
    print(f"CustomKNN Accuracy: {accuracy_custom:.2f}")
    print(f"Time taken by Scikit-learn KNN: {end_time_sklearn:.2f} seconds")
    print(f"Time taken by CustomKNN: {custom_knn_time:.2f} seconds")


if __name__ == "__main__":
    file_path = "./breast_cancer/breast-cancer.data"

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

    X, y = process_dataset(file_path, column_names, encodings)
    X_cleaned = replace_nan_and_question(X)
    X_imputed = replace_none_with_most_frequent(X_cleaned)

    mean, std = standard_scaler_fit(X_imputed)
    X_scaled = standard_scaler_transform(X_imputed, mean, std)

    main(X_scaled)
