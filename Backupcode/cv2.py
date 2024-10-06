import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import csv

# Importing my code 
from myCustomKNN import CustomKNN
from Scalers import *
from kFold import *
from cleanHelpers import *
from hyperParamTune import *
from splitdata import *
from hypoTesting import *

# Function to process dataset and perform encoding
def process_dataset(file_path, column_names, encodings):
    data = []
    with open(file_path, "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        for row in csv_reader:
            if row:
                data.append(row)

    for row in data:
        for key, value in zip(column_names, row):
            value = value.strip()
            if value in encodings[key]:
                row[column_names.index(key)] = encodings[key][value]
            else:
                row[column_names.index(key)] = None

    # Split into features and labels
    X = [row[:-1] for row in data]  # Features
    y = [row[-1] for row in data]   # Labels

    return X, y


# Main function to evaluate both CustomKNN and Scikit-learn KNN
def main(X_scaled, y, k=10):
    # Parameter grid for KNN hyperparameters
    param_grid = {
        'n_neighbors': list(range(1, 21)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'chebyshev', 'cosine']
    }

    print("\nPerforming 75-25 split to find the best hyperparameters...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, stratify=y, random_state=42)

    print("\nTuning CustomKNN using the 70-25 split...")
    best_knn_custom, best_params_custom, accuracy_custom = custom_knn_grid_search(X_train, y_train, X_test, y_test, param_grid)

    print(f"\nBest Parameters (CustomKNN): {best_params_custom}")
    print(f"Accuracy on 70-25 split: {accuracy_custom:.2f}")

    knn_custom = CustomKNN(n=best_params_custom['n_neighbors'],
                           task='classification',
                           weights=best_params_custom['weights'],
                           metric=best_params_custom['metric'])

    knn_sklearn = KNeighborsClassifier(n_neighbors=best_params_custom['n_neighbors'],
                                       weights=best_params_custom['weights'],
                                       metric=best_params_custom['metric'])

    print("\nTesting both CustomKNN and Scikit-learn KNN with K-Fold Cross-Validation...")

    start_time = time.time()
    accuracy_custom_kfold, accuracy_sklearn_kfold, accuracyCusArr, accuracySkArr = k_fold_cross_validation(X_scaled, y, k, knn_custom, knn_sklearn)
    total_time = time.time() - start_time

    print(f"\nCustomKNN Mean Accuracy (K-Fold): {accuracy_custom_kfold:.2f}")
    print(f"Scikit-learn KNN Mean Accuracy (K-Fold): {accuracy_sklearn_kfold:.2f}")
    print(f"Time taken by both models with K-Fold: {total_time:.2f} seconds")

    # Compare the performance of Scikit-learn KNN and CustomKNN
    print("\n--- Performance Comparison: Scikit-learn KNN vs CustomKNN ---")
    print(f"CustomKNN Mean Accuracy: {accuracy_custom_kfold:.2f}")
    print(f"Scikit-learn KNN Mean Accuracy: {accuracy_sklearn_kfold:.2f}")
    print(f"Time taken by both models with K-Fold: {total_time:.2f} seconds")

    # Perform hypothesis testing
    hypothesis_testing(accuracyCusArr, accuracySkArr)


# Running the script
if __name__ == "__main__":
    file_path = "./car_evaluation/car.data"
    column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

    encodings = {
        'buying': {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0},
        'maint': {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0},
        'doors': {'2': 0, '3': 1, '4': 2, '5more': 3},
        'persons': {'2': 0, '4': 1, 'more': 2},
        'lug_boot': {'small': 0, 'med': 1, 'big': 2},
        'safety': {'low': 0, 'med': 1, 'high': 2},
        'class': {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
    }

    # Data Preprocessing
    X, y = process_dataset(file_path, column_names, encodings)
    X_cleaned = replace_nan_and_question(X)
    X_imputed = replace_none_with_most_frequent(X_cleaned)
    X_imputed = [[float(value) for value in row] for row in X_imputed]

    # Standardize the features
    mean, std = standard_scaler_fit(X_imputed)
    X_scaled = standard_scaler_transform(X_imputed, mean, std)

    # Run the main function
    main(X_scaled, y, k=10)
