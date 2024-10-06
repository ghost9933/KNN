import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import csv

# Importing custom code
from myCustomKNN import CustomKNN
from Scalers import *
from kFold import *
from cleanHelpers import *
from hyperParamTune import *
from splitdata import *
from hypoTesting import *

# Function to process the dataset and return X (features) and y (labels)
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

# Main function to evaluate both CustomKNN and Scikit-learn KNN
def main(X_scaled, y, k=10):
    # Parameter grid for tuning hyperparameters
    param_grid = {
        'n_neighbors': list(range(1, 21)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'chebyshev', 'cosine']
    }

    print("\nPerforming 75-25 split to find the best hyperparameters...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, stratify=y, random_state=42)

    print("\nTuning CustomKNN using grid search on the 75-25 split...")
    best_knn_custom, best_params_custom, accuracy_custom = custom_knn_grid_search(X_train, y_train, X_test, y_test, param_grid)
    
    print(f"\nBest Parameters (CustomKNN): {best_params_custom}")
    print(f"Accuracy on 75-25 split: {accuracy_custom:.2f}")

    # Initialize both models with the best parameters
    knn_custom = CustomKNN(n=best_params_custom['n_neighbors'],
                           task='classification',
                           weights=best_params_custom['weights'],
                           metric=best_params_custom['metric'])

    knn_sklearn = KNeighborsClassifier(n_neighbors=best_params_custom['n_neighbors'],
                                       weights=best_params_custom['weights'],
                                       metric=best_params_custom['metric'])

    # K-Fold Cross-Validation comparison
    print("\nTesting both CustomKNN and Scikit-learn KNN with K-Fold Cross-Validation...")

    start_time = time.time()
    accuracy_custom_kfold, accuracy_sklearn_kfold, accuracyCusArr, accuracySkArr = k_fold_cross_validation(X_scaled, y, k, knn_custom, knn_sklearn)
    total_time = time.time() - start_time

    print(f"\nCustomKNN Mean Accuracy (K-Fold): {accuracy_custom_kfold:.2f}")
    print(f"Scikit-learn KNN Mean Accuracy (K-Fold): {accuracy_sklearn_kfold:.2f}")
    print(f"Time taken by both models with K-Fold: {total_time:.2f} seconds")

    # Performance comparison
    print("\n--- Performance Comparison: Scikit-learn KNN vs CustomKNN ---")
    print(f"CustomKNN Mean Accuracy: {accuracy_custom_kfold:.2f}")
    print(f"Scikit-learn KNN Mean Accuracy: {accuracy_sklearn_kfold:.2f}")
    print(f"Time taken by both models with K-Fold: {total_time:.2f} seconds")

    # Perform hypothesis testing
    hypothesis_testing(accuracyCusArr, accuracySkArr)

# Running the script
if __name__ == "__main__":

    file_path = "hayes_roth/hayes-roth.data"  # Replace with your dataset file path
    column_names = ['name', 'hobby', 'age', 'educational_level', 'marital_status', 'class']

    # Data Preprocessing
    X, y = process_dataset(file_path, column_names)
    X, y = replace_nan_and_question(X, y)
    X = replace_none_with_most_frequent(X)
    y = replace_none_with_mode(y)

    # Convert features and labels to numeric
    X = [[int(value) for value in row] for row in X]
    y = [int(float(value)) for value in y]

    # Standardize the features
    mean, std = standard_scaler_fit(X)
    X_scaled = standard_scaler_transform(X, mean, std)

    # Run the main function
    main(X_scaled, y, k=10)
