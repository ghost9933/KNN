import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from myCustomKNN import CustomKNN
import csv

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
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]
    return X, y

def replace_nan_and_question(X):
    for i in range(len(X)):
        for j in range(len(X[i])):
            if X[i][j] == '?' or X[i][j] is None:
                X[i][j] = None
    return X

def replace_none_with_most_frequent(X):
    X_transposed = list(zip(*X))
    for i, feature in enumerate(X_transposed):
        non_none_values = [x for x in feature if x is not None]
        most_frequent_value = max(set(non_none_values), key=non_none_values.count)
        X_transposed[i] = [most_frequent_value if x is None else x for x in feature]
    return [list(row) for row in zip(*X_transposed)]

def standard_scaler_fit(X):
    mean = []
    std = []
    for i in range(len(X[0])):
        col = [row[i] for row in X]
        mean_i = sum(col) / len(col)
        std_i = (sum((x - mean_i) ** 2 for x in col) / len(col)) ** 0.5
        mean.append(mean_i)
        std.append(std_i)
    return mean, std

def standard_scaler_transform(X, mean, std):
    X_scaled = []
    for row in X:
        scaled_row = [(x - m) / s if s != 0 else 0 for x, m, s in zip(row, mean, std)]
        X_scaled.append(scaled_row)
    return X_scaled

def k_fold_cross_validation(X, y, k, model, custom=False):
    fold_size = len(X) // k
    accuracies = []
    fold_indices = list(range(len(X)))
    for i in range(k):
        test_indices = fold_indices[i * fold_size:(i + 1) * fold_size]
        train_indices = fold_indices[:i * fold_size] + fold_indices[(i + 1) * fold_size:]
        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = sum([1 if pred == actual else 0 for pred, actual in zip(y_pred, y_test)]) / len(y_test)
        accuracies.append(accuracy)
        print(f"Fold {i + 1}: Accuracy = {accuracy:.2f}")
    return sum(accuracies) / len(accuracies)

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
    X, y = process_dataset(file_path, column_names, encodings)
    X_cleaned = replace_nan_and_question(X)
    X_imputed = replace_none_with_most_frequent(X_cleaned)
    X_imputed = [[float(value) for value in row] for row in X_imputed]
    mean, std = standard_scaler_fit(X_imputed)
    X_scaled = standard_scaler_transform(X_imputed, mean, std)
    main(X_scaled, y, k=5)
