import time
from sklearn.neighbors import KNeighborsClassifier


from myCustomKNN import CustomKNN
from DBknn import dbKNN
from Scalers import *
from kFold import *
from cleanHelpers import *
from hyperParamTune import *
from splitdata import *
from hypoTesting import *

def main(X_scaled, y, k=10, ):
    

    param_grid = {
        'n_neighbors': list(range(1, 8)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'cosine'],
    }
    print("\nPerforming 75-25 split to find the best hyperparameters...")
    X_train, X_test, y_train, y_test = dataSplit(X_scaled, y, test_size=0.25)

    print("\nTuning CustomKNN using the 70-25 split...")
    best_knn_custom, best_params_custom, accuracy_custom = custom_knn_grid_search(X_train, y_train, X_test, y_test, param_grid)
     
    print(f"\nBest Parameters (CustomKNN): {best_params_custom}")
    print(f"Accuracy on 70-25 split: {accuracy_custom:.2f}")

    print("\nTuning DB_KNN using the 70-25 split...")
    best_knn_DB, best_params_DB, accuracy_DB = dbKNN_knn_grid_search(X_train, y_train, X_test, y_test, param_grid)
     
    print(f"\nBest Parameters (DB_KNN): {best_params_DB}")
    print(f"Accuracy on 70-25 split: {accuracy_DB:.2f}")

    knn_custom = CustomKNN(n=best_params_custom['n_neighbors'],
                           task='classification',
                           weights=best_params_custom['weights'],
                           metric=best_params_custom['metric'])
    
   
    
    knn_DB=dbKNN(n=best_params_DB['n_neighbors'],
                           task='classification',
                           weights=best_params_DB['weights'],
                           metric=best_params_DB['metric'])
    
    

    knn_sklearn = KNeighborsClassifier(n_neighbors=best_params_custom['n_neighbors'],
                                       weights=best_params_custom['weights'],
                                       metric=best_params_custom['metric'])

    models={'knn_custom':knn_custom,'knn_DB':knn_DB,'knn_sklearn':knn_sklearn}

    print("\nTesting all models with K-Fold Cross-Validation...")

    start_time = time.time()
    mean_accuracies, accuracies = kFoldCV3(X_scaled, y, k, models)
    total_time = time.time() - start_time

    
    print("\n--- Cross-Validation Results ---")
    for model_name, mean_acc in mean_accuracies.items():
        print(f"{model_name} Mean Accuracy: {mean_acc:.2f}")

    print(f"Time taken by all models with K-Fold: {total_time:.2f} seconds")

    custom_model_names = ['knn_custom', 'knn_DB']
    best_custom_model_name = max(custom_model_names, key=lambda name: mean_accuracies[name])
    best_custom_model_accuracy = mean_accuracies[best_custom_model_name]
    print(f"\nBest Custom Model: {best_custom_model_name} with Mean Accuracy: {best_custom_model_accuracy:.2f}")

    accuracy_best_custom = accuracies[best_custom_model_name]
    accuracy_sklearn = accuracies['knn_sklearn']

    print("\n--- Performance Comparison: Scikit-learn KNN vs Best Custom Model ---")
    print(f"{best_custom_model_name} Mean Accuracy: {best_custom_model_accuracy:.2f}")
    print(f"Scikit-learn KNN Mean Accuracy: {mean_accuracies['knn_sklearn']:.2f}")
    print(f"Time taken by both models with K-Fold: {total_time:.2f} seconds")

    print("\n--- Hypothesis Testing: Scikit-learn KNN vs Best Custom Model ---")
    hypothesis_testing(accuracy_best_custom, accuracy_sklearn)
