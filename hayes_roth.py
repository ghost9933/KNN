import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import csv


# Importing my code 
from myCustomKNN import CustomKNN
from Scalers import *
from kFold import *
from cleanHelpers import *



def process_dataset(file_path, column_names):
    data = []
    with open(file_path, "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header if present
        for row in csv_reader:
            if row:
                data.append(row)
    X = [row[1:-1] for row in data]  # Exclude 'name' (ID) and 'class' columns
    y = [row[-1] for row in data]    # Target is the 'class' column
    return X, y

def main(X_scaled, y, k=10):
    print("\nTesting CustomKNN with K-Fold Cross-Validation...")
    best_params_custom = {'n_neighbors': 5, 'weights': 'distance', 'metric': 'euclidean'}
    knn_custom = CustomKNN(n=best_params_custom['n_neighbors'],
                           task='classification',
                           weights=best_params_custom['weights'],
                           metric=best_params_custom['metric'])
    start_time_custom = time.time()
    accuracy_custom = k_fold_cross_validation(X_scaled, y, k, knn_custom, custom=True)
    end_time_custom = time.time() - start_time_custom
    print(f"\nCustomKNN Mean Accuracy: {accuracy_custom:.2f}")
    print(f"Time taken by CustomKNN: {end_time_custom:.2f} seconds")
    print("\nTesting Scikit-learn KNN with K-Fold Cross-Validation...")
    knn_sklearn = KNeighborsClassifier(n_neighbors=best_params_custom['n_neighbors'],
                                       weights=best_params_custom['weights'],
                                       metric=best_params_custom['metric'])
    start_time_sklearn = time.time()
    accuracy_sklearn = k_fold_cross_validation(X_scaled, y, k, knn_sklearn, custom=False)
    end_time_sklearn = time.time() - start_time_sklearn
    print(f"\nScikit-learn KNN Mean Accuracy: {accuracy_sklearn:.2f}")
    print(f"Time taken by Scikit-learn KNN: {end_time_sklearn:.2f} seconds")
    print("\n--- Performance Comparison: Scikit-learn KNN vs CustomKNN ---")
    print(f"Scikit-learn KNN Accuracy: {accuracy_sklearn:.2f}")
    print(f"CustomKNN Accuracy: {accuracy_custom:.2f}")
    print(f"Time taken by Scikit-learn KNN: {end_time_sklearn:.2f} seconds")
    print(f"Time taken by CustomKNN: {end_time_custom:.2f} seconds")

if __name__ == "__main__":
    file_path = "hayes_roth\hayes-roth.data"  # Replace with your dataset file path
    column_names = ['name', 'hobby', 'age', 'educational_level', 'marital_status', 'class']
    X, y = process_dataset(file_path, column_names)
    X, y = replace_nan_and_question(X, y)
    X = replace_none_with_most_frequent(X)
    y = replace_none_with_mode(y)
    X = [[int(value) for value in row] for row in X]
    y = [int(float(value)) for value in y]
    mean, std = standard_scaler_fit(X)
    X_scaled = standard_scaler_transform(X, mean, std)
    main(X_scaled, y, k=5)
