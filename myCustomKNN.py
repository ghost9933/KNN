from collections import Counter
import math

class CustomKNN:
    valid_metrics={'euclidean','manhattan','cosine'}
    def __init__(self, n=5, metric='euclidean',weights='uniform'):
        if not isinstance(n, int):
            raise ValueError("n_neighbors should be an integer")
        self.neighbors = n
        self.metric = metric.lower()
        if self.metric not in CustomKNN.valid_metrics and not self.metric.startswith('minkowski'):
            raise ValueError("Invalid metric. Metric should be one of 'euclidean', 'manhattan', 'chebyshev', 'hamming', 'cosine'")
        self.X_train = None
        self.y_train = None
        self.weights = weights

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def calculateDistance(self, x1, x2):
        try:
            if self.metric == 'euclidean':
                return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))
            elif self.metric == 'manhattan':
                return sum(abs(a - b) for a, b in zip(x1, x2))
            elif self.metric == 'cosine':
                dot_product = sum(a * b for a, b in zip(x1, x2))
                magnitude_x1 = math.sqrt(sum(a ** 2 for a in x1))
                magnitude_x2 = math.sqrt(sum(b ** 2 for b in x2))
                return 1 - (dot_product / (magnitude_x1 * magnitude_x2))
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
        except Exception as e:
            print(f"An error occurred in calculateDistance: {e}")
            return None
    def predictSingle(self, x):
        try:
            # Calculate distances to all training samples
            distances = []
            for i in range(len(self.X_train)):
                xi = self.X_train[i]
                yi = self.y_train[i]
                distance = self.calculateDistance(x, xi)
                distances.append((distance, yi))
            
            # Sort distances in ascending order
            distances.sort()  # Sorts based on the first element of the tuple (distance)
            
            # Select the top k nearest neighbors
            top_neighbors = distances[:self.neighbors]
            
            if self.weights == 'distance':
                # Weighted voting based on inverse distance
                weighted_votes = {}
                for neighbor in top_neighbors:
                    distance = neighbor[0]
                    label = neighbor[1]
                    if distance != 0:
                        weight = 1 / (distance + 1e-5)
                    else:
                        weight = 1.0  # Handle zero distance
                    if label in weighted_votes:
                        weighted_votes[label] += weight
                    else:
                        weighted_votes[label] = weight
                # Find the label with the highest total weight
                max_weight = -1
                predicted_label = None
                for label in weighted_votes:
                    if weighted_votes[label] > max_weight:
                        max_weight = weighted_votes[label]
                        predicted_label = label
                return predicted_label
            else:
                # Uniform voting
                label_counts = {}
                for neighbor in top_neighbors:
                    label = neighbor[1]
                    if label in label_counts:
                        label_counts[label] += 1
                    else:
                        label_counts[label] = 1
                # Find the label with the highest count
                max_count = -1
                predicted_label = None
                for label in label_counts:
                    if label_counts[label] > max_count:
                        max_count = label_counts[label]
                        predicted_label = label
                return predicted_label
        except Exception as e:
            print(f"An error occurred in predictSingle: {e}")
            return None

        

    def predict(self, X):
        try:
            predictions = [self.predictSingle(x) for x in X]
            return predictions
        except Exception as e:
            print(f"An error occurred in predict: {e}")
            return None

    def score(self, X, y):
        try:
            predictions = self.predict(X)
        
            correct = sum(1 for pred, actual in zip(predictions, y) if pred == actual)
            return correct / len(y)
        except Exception as e:
            print(f"An error occurred in score: {e}")
            return None


