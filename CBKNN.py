from collections import Counter
import math

class cbKNN:
    valid_metrics={'euclidean','manhattan','cosine'}
    def __init__(self, n=5, metric='euclidean',weights='uniform'):
        if not isinstance(n, int):
            raise ValueError("n_neighbors should be an integer")
        self.neighbors = n
        self.metric = metric.lower()
        if self.metric not in cbKNN.valid_metrics and not self.metric.startswith('minkowski'):
            raise ValueError("Invalid metric. Metric should be one of 'euclidean', 'manhattan', 'chebyshev', 'hamming', 'cosine'")
        self.X_train = None
        self.y_train = None
        self.weights = weights
        self.classes = None
        self.max_k = n  

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes = list(set(y))

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
        # For each class, find the nearest neighbors and compute harmonic mean of distances
        class_harmonic_means = {}
        
        for cls in self.classes:
            # Get all training samples of this class
            class_samples = [self.X_train[i] for i in range(len(self.X_train)) if self.y_train[i] == cls]
            # Calculate distances to all samples of this class
            distances = [self.calculateDistance(x, sample) for sample in class_samples]
            distances.sort()
            
            # Determine the optimal k for this class (up to self.max_k or the number of available samples)
            k = min(self.max_k, len(distances))
            if k == 0:
                continue  # Skip this class if there are no samples
            top_k_distances = distances[:k]
            
            # Calculate harmonic mean of the distances
            # Harmonic mean formula: n / sum(1 / x_i)
            denominator = sum(1 / (d + 1e-5) for d in top_k_distances)  # Add epsilon to avoid division by zero
            harmonic_mean = k / denominator if denominator != 0 else float('inf')
            
            class_harmonic_means[cls] = harmonic_mean
        
        # Choose the class with the lowest harmonic mean
        if not class_harmonic_means:
            return None  # Unable to classify
        predicted_class = min(class_harmonic_means, key=class_harmonic_means.get)
        return predicted_class

        

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


