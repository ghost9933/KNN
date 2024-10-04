from collections import Counter
import math

class CustomKNN:
    def __init__(self, n_neighbors=5, metric='euclidean', task='classification'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.task = task
        self.X_train = None
        self.y_train = None

    # Function to fit the training data
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Function to calculate distances
    def _calculate_distance(self, x1, x2):
        if self.metric == 'euclidean':
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))
        elif self.metric == 'manhattan':
            return sum(abs(a - b) for a, b in zip(x1, x2))
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    # Function to predict the value for a single data point
    def _predict_single(self, x):
        distances = []

        # Calculate distances from the point to all other training points
        for i in range(len(self.X_train)):
            distance = self._calculate_distance(self.X_train[i], x)
            distances.append((distance, self.y_train[i]))

        # Sort distances to find the closest neighbors
        sorted_distances = sorted(distances, key=lambda x: x[0])
        nearest_neighbors = sorted_distances[:self.n_neighbors]

        # Make prediction based on the task type
        if self.task == 'classification':
            labels = [neighbor[1] for neighbor in nearest_neighbors]
            most_common_label = Counter(labels).most_common(1)[0][0]
            return most_common_label
        elif self.task == 'regression':
            values = [neighbor[1] for neighbor in nearest_neighbors]
            return sum(values) / len(values)
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    # Function to predict for an entire dataset
    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return predictions

    # Function to score the model based on the testing data
    def score(self, X, y):
        predictions = self.predict(X)
        if self.task == 'classification':
            correct = sum(1 for pred, actual in zip(predictions, y) if pred == actual)
            return correct / len(y)
        elif self.task == 'regression':
            ss_total = sum((actual - sum(y) / len(y)) ** 2 for actual in y)
            ss_residual = sum((actual - pred) ** 2 for pred, actual in zip(predictions, y))
            return 1 - (ss_residual / ss_total)
        else:
            raise ValueError(f"Unsupported task: {self.task}")

# Example Usage
if __name__ == "__main__":
    # Dummy data
    X_train = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
    y_train = [0, 1, 1, 0, 1]
    X_test = [[2, 3], [6, 7]]
    y_test = [1, 0]

    # Classification Example
    knn = CustomKNN(n_neighbors=3, task='classification')
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracy = knn.score(X_test, y_test)

    print(f"Predictions: {predictions}")
    print(f"Accuracy: {accuracy:.2f}")
