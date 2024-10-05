from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import time
import psutil

# Loading the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class CustomKNN:
    valid_metrics = {'euclidean', 'chebyshev', 'manhattan', 'hamming', 'cosine'}

    def __init__(self, n=5, metric='euclidean', task='classification', weights='uniform', adaptive_k=False, scaler=None):
        self.neighbors = n
        self.metric = metric.lower()
        self.weights = weights
        self.adaptive_k = adaptive_k
        self.scaler = scaler
        self.task = task
        self.X_train = None
        self.y_train = None
        self.min_vals = None
        self.max_vals = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

        if self.scaler == 'minmax':
            self.min_vals = [min(col) for col in zip(*X)]
            self.max_vals = [max(col) for col in zip(*X)]
            self.X_train = [
                [(x_i - min_v) / (max_v - min_v) if max_v != min_v else 0 for x_i, min_v, max_v in zip(row, self.min_vals, self.max_vals)]
                for row in X
            ]

    def calculateDistance(self, x1, x2):
        if self.metric == 'euclidean':
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))
        elif self.metric == 'manhattan':
            return sum(abs(a - b) for a, b in zip(x1, x2))
        elif self.metric == 'chebyshev':     
            return max(abs(a - b) for a, b in zip(x1, x2))
        elif self.metric == 'hamming':
            return sum(a != b for a, b in zip(x1, x2))
        elif self.metric == 'cosine':
            dot_product = sum(a * b for a, b in zip(x1, x2))
            magnitude_x1 = math.sqrt(sum(a ** 2 for a in x1))
            magnitude_x2 = math.sqrt(sum(b ** 2 for b in x2))
            return 1 - (dot_product / (magnitude_x1 * magnitude_x2))
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def predictSingle(self, x):
        if self.scaler == 'minmax' and self.min_vals is not None and self.max_vals is not None:
            x = [(x_i - min_v) / (max_v - min_v) if max_v != min_v else 0 for x_i, min_v, max_v in zip(x, self.min_vals, self.max_vals)]
        
        distances = [(self.calculateDistance(x, self.X_train[i]), self.y_train[i]) for i in range(len(self.X_train))]
        distances = sorted(distances, key=lambda x: x[0])[:self.neighbors]
        
        if self.weights == 'distance':
            distances = [(1 / (dist + 1e-5), label) if dist != 0 else (1.0, label) for dist, label in distances]

        if self.task == 'classification':
            if self.weights == 'uniform':
                labels = [neighbor[1] for neighbor in distances]
            else:
                labels = [neighbor[1] for neighbor in distances for _ in range(int(neighbor[0] * 100))]
            most_common_label = Counter(labels).most_common(1)[0][0]
            return most_common_label
        elif self.task == 'regression':
            if self.weights == 'uniform':
                values = [neighbor[1] for neighbor in distances]
                return sum(values) / len(values)
            else:
                weighted_sum = sum(weight * label for weight, label in distances)
                total_weight = sum(weight for weight, _ in distances)
                return weighted_sum / total_weight

    def predict(self, X):
        with ThreadPoolExecutor() as executor:
            predictions = list(executor.map(self.predictSingle, X))
        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        if self.task == 'classification':
            correct = sum(1 for pred, actual in zip(predictions, y) if pred == actual)
            return correct / len(y)
        elif self.task == 'regression':
            ss_total = sum((actual - sum(y) / len(y)) ** 2 for actual in y)
            ss_residual = sum((actual - pred) ** 2 for pred, actual in zip(predictions, y))
            return 1 - (ss_residual / ss_total)

# Create an advanced KNN classifier with distance weighting and feature scaling
start_time = time.time()
knn = CustomKNN(n=3, task='classification', weights='distance', scaler='minmax')
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
custom_knn_time = time.time() - start_time
custom_knn_cpu = psutil.cpu_percent(interval=1)
print(f"Enhanced Custom KNN Predictions: {predictions}")
accuracy = knn.score(X_test, y_test)
print(f"Enhanced Custom KNN Accuracy: {accuracy:.2f}")
print(f"Enhanced Custom KNN Time Taken: {custom_knn_time:.2f} seconds")
print(f"Enhanced Custom KNN CPU Usage: {custom_knn_cpu:.2f}%")

# Original KNN using sklearn
start_time = time.time()
knn_original = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean')
knn_original.fit(X_train, y_train)
predictions_original = knn_original.predict(X_test)
original_knn_time = time.time() - start_time
original_knn_cpu = psutil.cpu_percent(interval=1)
accuracy_original = accuracy_score(y_test, predictions_original)
print(f"Original KNN Accuracy: {accuracy_original:.2f}")
print(f"Original KNN Time Taken: {original_knn_time:.2f} seconds")
print(f"Original KNN CPU Usage: {original_knn_cpu:.2f}%")




# Print comparison of accuracies and performance
print("\n--- Comparison of Custom KNN vs Original KNN ---")
print(f"Original KNN Accuracy: {accuracy_original:.2f}")
print(f"Enhanced Custom KNN Accuracy: {accuracy:.2f}")
print(f"Original KNN Time Taken: {original_knn_time:.2f} seconds")
print(f"Enhanced Custom KNN Time Taken: {custom_knn_time:.2f} seconds")
print(f"Original KNN CPU Usage: {original_knn_cpu:.2f}%")
print(f"Enhanced Custom KNN CPU Usage: {custom_knn_cpu:.2f}%")