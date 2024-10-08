import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import csv


# Importing my code 
from myCustomKNN import CustomKNN
from DBknn import dbKNN
from Scalers import *
from kFold import *
from cleanHelpers import *
from hyperParamTune import *
from splitdata import *
from hypoTesting import *

from mainDriver import main

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


# def main(X_scaled, y, k=10, ):
    

#     param_grid = {
#         'n_neighbors': list(range(1, 9)),
#         'weights': ['uniform', 'distance'],
#         'metric': ['euclidean', 'manhattan', 'cosine'],
#     }
#     print("\nPerforming 75-25 split to find the best hyperparameters...")
#     X_train, X_test, y_train, y_test = dataSplit(X_scaled, y, test_size=0.25)

#     print("\nTuning CustomKNN using the 70-25 split...")
#     best_knn_custom, best_params_custom, accuracy_custom = custom_knn_grid_search(X_train, y_train, X_test, y_test, param_grid)
     
#     print(f"\nBest Parameters (CustomKNN): {best_params_custom}")
#     print(f"Accuracy on 70-25 split: {accuracy_custom:.2f}")

#     print("\nTuning DB_KNN using the 70-25 split...")
#     best_knn_DB, best_params_DB, accuracy_DB = dbKNN_knn_grid_search(X_train, y_train, X_test, y_test, param_grid)
     
#     print(f"\nBest Parameters (DB_KNN): {best_params_DB}")
#     print(f"Accuracy on 70-25 split: {accuracy_DB:.2f}")

#     knn_custom = CustomKNN(n=best_params_custom['n_neighbors'],
#                            task='classification',
#                            weights=best_params_custom['weights'],
#                            metric=best_params_custom['metric'])
    
   
    
#     knn_DB=dbKNN(n=best_params_DB['n_neighbors'],
#                            task='classification',
#                            weights=best_params_DB['weights'],
#                            metric=best_params_DB['metric'])
    
    

#     knn_sklearn = KNeighborsClassifier(n_neighbors=best_params_custom['n'],
#                                        weights=best_params_custom['weights'],
#                                        metric=best_params_custom['metric'])

#     models={'knn_custom':knn_custom,'knn_DB':knn_DB,'knn_sklearn':knn_sklearn}

#     print("\nTesting all models with K-Fold Cross-Validation...")

#     start_time = time.time()
#     mean_accuracies, accuracies = kFoldCV3(X_scaled, y, k, models)
#     total_time = time.time() - start_time

    
#     print("\n--- Cross-Validation Results ---")
#     for model_name, mean_acc in mean_accuracies.items():
#         print(f"{model_name} Mean Accuracy: {mean_acc:.2f}")

#     print(f"Time taken by all models with K-Fold: {total_time:.2f} seconds")

#     custom_model_names = ['knn_custom', 'knn_DB']
#     best_custom_model_name = max(custom_model_names, key=lambda name: mean_accuracies[name])
#     best_custom_model_accuracy = mean_accuracies[best_custom_model_name]
#     print(f"\nBest Custom Model: {best_custom_model_name} with Mean Accuracy: {best_custom_model_accuracy:.2f}")

#     accuracy_best_custom = accuracies[best_custom_model_name]
#     accuracy_sklearn = accuracies['knn_sklearn']

#     print("\n--- Performance Comparison: Scikit-learn KNN vs Best Custom Model ---")
#     print(f"{best_custom_model_name} Mean Accuracy: {best_custom_model_accuracy:.2f}")
#     print(f"Scikit-learn KNN Mean Accuracy: {mean_accuracies['knn_sklearn']:.2f}")
#     print(f"Time taken by both models with K-Fold: {total_time:.2f} seconds")

#     print("\n--- Hypothesis Testing: Scikit-learn KNN vs Best Custom Model ---")
#     hypothesis_testing(accuracy_best_custom, accuracy_sklearn)




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
