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

def process_dataset(file_path, column_names, encodings):
    data = []
    with open(file_path, "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        for row in csv_reader:
            if row:
                data.append(row)
    for row in data:
        for key, value in zip(column_names, row):
            row[column_names.index(key)] = encodings[key][value.strip()]

    X = [row[1:] for row in data]
    y = [row[0] for row in data]

    return X, y


def main(X_scaled, y, k=10, ):
    
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
    accuracy_custom_kfold, accuracy_sklearn_kfold, accuracyCusArr,accuracySkArr= k_fold_cross_validation(X_scaled, y, k, knn_custom, knn_sklearn)
    total_time = time.time() - start_time

    print(f"\nCustomKNN Mean Accuracy (K-Fold): {accuracy_custom_kfold:.2f}")
    print(f"Scikit-learn KNN Mean Accuracy (K-Fold): {accuracy_sklearn_kfold:.2f}")
    print(f"Time taken by both models with K-Fold: {total_time:.2f} seconds")

    # Step 5: Compare the performance of Scikit-learn KNN and CustomKNN
    print("\n--- Performance Comparison: Scikit-learn KNN vs CustomKNN ---")
    print(f"CustomKNN Mean Accuracy: {accuracy_custom_kfold:.2f}")
    print(f"Scikit-learn KNN Mean Accuracy: {accuracy_sklearn_kfold:.2f}")
    print(f"Time taken by both models with K-Fold: {total_time:.2f} seconds")

    hypothesis_testing(accuracyCusArr, accuracySkArr)



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

    main(X_scaled,y, k=10)
