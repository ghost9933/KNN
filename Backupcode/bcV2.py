import time
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

# Import CustomKNN and necessary functions
from myCustomKNN import CustomKNN
from kFold import k_fold_cross_validation


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


def main(X_scaled, y, k=10, param_grid=None):
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Define parameter grid if not provided
    if param_grid is None:
        param_grid = {
            'n_neighbors': list(range(1, 21)),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'chebyshev', 'hamming', 'cosine']
        }

    # CustomKNN Grid Search
    print("\nTuning CustomKNN...")
    best_params_custom, best_accuracy_custom = custom_knn_grid_search(X_train, y_train, X_test, y_test, param_grid)
    print(f"Best Parameters (CustomKNN): {best_params_custom}")
    print(f"Accuracy of Best CustomKNN model: {best_accuracy_custom:.2f}")

    # Initialize CustomKNN with best parameters and evaluate using K-Fold Cross-Validation
    knn_custom = CustomKNN(n=best_params_custom['n_neighbors'],
                           weights=best_params_custom['weights'],
                           metric=best_params_custom['metric'])

    accuracy_custom_kfold = k_fold_cross_validation(X_scaled, y, k, knn_custom, custom=True)
    print(f"\nCustomKNN Mean Accuracy (K-Fold): {accuracy_custom_kfold:.2f}")

    # Scikit-learn KNN Grid Search
    print("\nTuning Scikit-learn KNN...")
    knn_sklearn = KNeighborsClassifier()
    grid_search_sklearn = GridSearchCV(knn_sklearn, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search_sklearn.fit(X_train, y_train)

    best_params_sklearn = grid_search_sklearn.best_params_
    best_accuracy_sklearn = grid_search_sklearn.best_score_
    print(f"Best Parameters (Scikit-learn KNN): {best_params_sklearn}")
    print(f"Accuracy of Best Scikit-learn KNN model: {best_accuracy_sklearn:.2f}")

    # Evaluate Scikit-learn KNN using K-Fold Cross-Validation
    accuracy_sklearn_kfold = k_fold_cross_validation(X_scaled, y, k, grid_search_sklearn.best_estimator_, custom=False)
    print(f"\nScikit-learn KNN Mean Accuracy (K-Fold): {accuracy_sklearn_kfold:.2f}")

    # Compare the performance of CustomKNN and Scikit-learn KNN
    print("\n--- Performance Comparison ---")
    print(f"CustomKNN Mean Accuracy (K-Fold): {accuracy_custom_kfold:.2f}")
    print(f"Scikit-learn KNN Mean Accuracy (K-Fold): {accuracy_sklearn_kfold:.2f}")


# CustomKNN Grid Search for finding best hyperparameters using 70-30 split
def custom_knn_grid_search(X_train, y_train, X_test, y_test, param_grid):
    best_params = None
    best_accuracy = 0
    best_model = None

    for n_neighbors in param_grid['n_neighbors']:
        for weights in param_grid['weights']:
            for metric in param_grid['metric']:
                knn_custom = CustomKNN(n=n_neighbors, task='classification', weights=weights, metric=metric)
                start_time = time.time()
                knn_custom.fit(X_train, y_train)
                y_pred_custom = knn_custom.predict(X_test)
                accuracy_custom = accuracy_score(y_test, y_pred_custom)

                if accuracy_custom > best_accuracy:
                    best_accuracy = accuracy_custom
                    best_params = {'n_neighbors': n_neighbors, 'weights': weights, 'metric': metric}
                    best_model = knn_custom

    return best_params, best_accuracy


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
    main(X, y)
