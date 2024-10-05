from collections import Counter
import math

class CustomKNN:
    valid_metrics={'euclidean','chebyshev','manhattan','hamming','cosine'}
    def __init__(self, n=5, metric='euclidean', task='classification', weights='uniform'):
        self.n_neighbors = n
        self.metric = metric
        if self.metric not in CustomKNN.valid_metrics:
            print("metric has to be one of :", self.valid_metrics)
        self.task = task
        self.weights = weights
        self.X_train = None
        self.y_train = None

    # Function to fit the training data
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Function to calculate distances
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

    # Function to predict the value for a single data point
    def _predict_single(self, x):
        distances = []

        # Calculate distances from the point to all other training points
        for i in range(len(self.X_train)):
            distance = self.calculateDistance(self.X_train[i], x)
            distances.append((distance, self.y_train[i]))

        # Sort distances to find the closest neighbors
        sorted_distances = sorted(distances, key=lambda x: x[0])
        nearest_neighbors = sorted_distances[:self.n_neighbors]

        # Weighting logic
        if self.weights == 'distance':
            # Use inverse of the distance as weight, avoid division by zero
            weighted_votes = {}
            for distance, label in nearest_neighbors:
                weight = 1 / (distance + 1e-5) if distance != 0 else 1.0
                if label in weighted_votes:
                    weighted_votes[label] += weight
                else:
                    weighted_votes[label] = weight
            return max(weighted_votes, key=weighted_votes.get)
        else:
            # Uniform weighting
            labels = [neighbor[1] for neighbor in nearest_neighbors]
            most_common_label = Counter(labels).most_common(1)[0][0]
            return most_common_label

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

# # Example Usage
# if __name__ == "__main__":
#     # Dummy data
#     X_train = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
#     y_train = [0, 1, 1, 0, 1]
#     X_test = [[2, 3], [6, 7]]
#     y_test = [1, 0]

#     # Classification Example
#     knn = CustomKNN(n_neighbors=3, task='classification', weights='distance')
#     knn.fit(X_train, y_train)
#     predictions = knn.predict(X_test)
#     accuracy = knn.score(X_test, y_test)

#     print(f"Predictions: {predictions}")
#     print(f"Accuracy: {accuracy:.2f}")