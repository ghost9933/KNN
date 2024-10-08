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
from DBknn import dbKNN

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

# Function to perform k-fold cross-validation
def k_fold_cross_validation(X, y, k, model):
    n = len(X)
    fold_size = n // k
    indices = list(range(n))
    random.shuffle(indices)  # Shuffle indices for random selection of folds

    accuracies = []
    for i in range(k):
        test_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = indices[:i * fold_size] + indices[(i + 1) * fold_size:]

        X_train = [X[idx] for idx in train_indices]
        X_test = [X[idx] for idx in test_indices]
        y_train = [y[idx] for idx in train_indices]
        y_test = [y[idx] for idx in test_indices]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    mean_accuracy = sum(accuracies) / k
    return mean_accuracy, model

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

    # Further split the test set into validation and final test set (50-50 split)
    X_validation, X_final_test, y_validation, y_final_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=42)

    print("\nTuning CustomKNN using grid search on the 75-25 split...")
    best_knn_custom, best_params_custom, accuracy_custom = custom_knn_grid_search(X_train, y_train, X_validation, y_validation, param_grid)
    
    print(f"\nBest Parameters (CustomKNN): {best_params_custom}")
    print(f"Accuracy on 75-25 split: {accuracy_custom:.2f}")

    # Initialize both models with the best parameters from the validation set
    knn_custom = dbKNN(n=best_params_custom['n_neighbors'],
                           weights=best_params_custom['weights'],
                           metric=best_params_custom['metric'])
    
    knn_sklearn = KNeighborsClassifier(n_neighbors=best_params_custom['n_neighbors'],
                                       weights=best_params_custom['weights'],
                                       metric=best_params_custom['metric'])

    # K-Fold Cross-Validation comparison
    print("\nTesting both CustomKNN and Scikit-learn KNN with K-Fold Cross-Validation...")

    start_time = time.time()
    accuracy_custom_kfold, best_custom_model = k_fold_cross_validation(X_scaled, y, k, knn_custom)
    accuracy_sklearn_kfold, best_sklearn_model = k_fold_cross_validation(X_scaled, y, k, knn_sklearn)
    total_time = time.time() - start_time

    print(f"\nCustomKNN Mean Accuracy (K-Fold): {accuracy_custom_kfold:.2f}")
    print(f"Scikit-learn KNN Mean Accuracy (K-Fold): {accuracy_sklearn_kfold:.2f}")
    print(f"Time taken by both models with K-Fold: {total_time:.2f} seconds")

    # Evaluate best models on the validation set
    y_pred_custom = best_custom_model.predict(X_validation)
    accuracy_custom_validation = accuracy_score(y_validation, y_pred_custom)

    y_pred_sklearn = best_sklearn_model.predict(X_validation)
    accuracy_sklearn_validation = accuracy_score(y_validation, y_pred_sklearn)

    print("\n--- Performance on Validation Set ---")
    print(f"CustomKNN Accuracy on Validation Set: {accuracy_custom_validation:.2f}")
    print(f"Scikit-learn KNN Accuracy on Validation Set: {accuracy_sklearn_validation:.2f}")

    # Perform hypothesis testing
    hypothesis_testing(accuracy_custom_validation, accuracy_sklearn_validation)

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
    main(X_scaled, y, k=20)
