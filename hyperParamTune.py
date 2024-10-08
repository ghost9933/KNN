

# custom import
from myCustomKNN import CustomKNN

from enhancedKNN import CustomKNN as EKNN
from sklearn.metrics import accuracy_score


from DBknn import dbKNN




import random
from itertools import product

def hyperParamTuneKFold(X, y, k, model_class, param_grid, random_seed=None):
    """
    Performs k-fold cross-validation with hyperparameter tuning using grid search for a single model.
    
    Parameters:
    - X: Features dataset
    - y: Labels dataset
    - k: Number of folds
    - model_class: The model class to instantiate (e.g., CustomKNN)
    - param_grid: Dictionary of hyperparameters to search over
    - random_seed: Seed for random number generator
    
    Returns:
    - mean_accuracy: Mean accuracy across all folds
    - accuracies: List of accuracies per fold
    - best_params_overall: Best parameters found over all folds
    """
    if random_seed is not None:
        random.seed(random_seed)  # Set random seed for reproducibility

    n_samples = len(X)
    fold_size = n_samples // k
    accuracies = []
    best_params_list = []
    param_performance = {}  # To store performance per param set

    fold_indices = list(range(n_samples))
    random.shuffle(fold_indices)  # Shuffle the indices to randomize the folds

    for i in range(k):
        # Create the test and train indices for this fold
        test_indices = fold_indices[i * fold_size:(i + 1) * fold_size]
        train_indices = fold_indices[:i * fold_size] + fold_indices[(i + 1) * fold_size:]

        X_train = [X[j] for j in train_indices]
        y_train = [y[j] for j in train_indices]
        X_test = [X[j] for j in test_indices]
        y_test = [y[j] for j in test_indices]

        print(f"Fold {i + 1}:")
        print(f"  Tuning hyperparameters...")

        # Generate all combinations of hyperparameters
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = [dict(zip(param_names, values)) for values in product(*param_values)]

        best_fold_accuracy = -1
        best_fold_params = None

        for params in param_combinations:
            # Instantiate the model with given parameters
            model = model_class(**params)
            # Train and evaluate the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = sum([1 if pred == actual else 0 for pred, actual in zip(y_pred, y_test)]) / len(y_test)

            params_key = tuple(sorted(params.items()))
            if params_key not in param_performance:
                param_performance[params_key] = []
            param_performance[params_key].append(accuracy)

            if accuracy > best_fold_accuracy:
                best_fold_accuracy = accuracy
                best_fold_params = params

        accuracies.append(best_fold_accuracy)
        best_params_list.append(best_fold_params)
        print(f"    Best Accuracy = {best_fold_accuracy:.2f} with params {best_fold_params}")

    mean_accuracy = sum(accuracies) / len(accuracies)

    param_avg_performance = {}
    for params_key, perf_list in param_performance.items():
        avg_perf = sum(perf_list) / len(perf_list)
        param_avg_performance[params_key] = avg_perf

    best_params_key = max(param_avg_performance, key=param_avg_performance.get)
    best_params_overall = dict(best_params_key)
    print(f"\nBest overall parameters: {best_params_overall} with average accuracy {param_avg_performance[best_params_key]:.2f}")

    return mean_accuracy, accuracies, best_params_overall



def custom_knn_grid_search(X_train, y_train, X_test, y_test, param_grid):
    best_params = None
    best_accuracy = 0
    best_model = None

    for n_neighbors in param_grid['n_neighbors']:
        for weights in param_grid['weights']:
            for metric in param_grid['metric']:
                knn_custom = CustomKNN(n=n_neighbors, task='classification', weights=weights, metric=metric)
                knn_custom.fit(X_train, y_train)
                y_pred_custom = knn_custom.predict(X_test)
                accuracy_custom = accuracy_score(y_test, y_pred_custom)

                if accuracy_custom > best_accuracy:
                    best_accuracy = accuracy_custom
                    best_params = {'n_neighbors': n_neighbors, 'weights': weights, 'metric': metric}
                    best_model = knn_custom

    return best_model, best_params, best_accuracy


def dbKNN_knn_grid_search(X_train, y_train, X_test, y_test, param_grid):
    best_params = None
    best_accuracy = 0
    best_model = None

    for n_neighbors in param_grid['n_neighbors']:
        for weights in param_grid['weights']:
            for metric in param_grid['metric']:
                knn_custom = dbKNN(n=n_neighbors, task='classification', weights=weights, metric=metric)
                knn_custom.fit(X_train, y_train)
                y_pred_custom = knn_custom.predict(X_test)
                accuracy_custom = accuracy_score(y_test, y_pred_custom)

                if accuracy_custom > best_accuracy:
                    best_accuracy = accuracy_custom
                    best_params = {'n_neighbors': n_neighbors, 'weights': weights, 'metric': metric}
                    best_model = knn_custom

    return best_model, best_params, best_accuracy

