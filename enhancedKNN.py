from collections import Counter
import math

class CustomKNN:
    valid_metrics = {'euclidean', 'manhattan', 'chebyshev', 'cosine', 'polynomial', 'rbf', 'sigmoid'}
    
    def __init__(self, n=5, metric='euclidean', weights='uniform', task='classification', degree=3, gamma=0.1, alpha=0.1, beta=0.0):
        if not isinstance(n, int):
            raise ValueError("n_neighbors should be an integer")
        self.neighbors = n
        self.metric = metric.lower()
        if self.metric not in CustomKNN.valid_metrics and not self.metric.startswith('minkowski'):
            raise ValueError("Invalid metric. Metric should be one of 'euclidean', 'manhattan', 'chebyshev', 'cosine', 'polynomial', 'rbf', 'sigmoid'")
        if task not in {'classification', 'regression'}:
            raise ValueError("task should be either classification or regression")
        self.task = task
        self.X_train = None
        self.y_train = None
        self.weights = weights
        
        # Kernel-specific parameters
        self.degree = degree  # For polynomial kernel
        self.gamma = gamma  # For rbf kernel
        self.alpha = alpha  # For sigmoid kernel
        self.beta = beta  # For sigmoid kernel

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def calculateDistance(self, x1, x2):
        try:
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
            elif self.metric == 'polynomial':
                dot_product = sum(a * b for a, b in zip(x1, x2))
                return (1 + dot_product) ** self.degree
            elif self.metric == 'rbf':
                diff = sum((a - b) ** 2 for a, b in zip(x1, x2))
                return math.exp(-self.gamma * diff)
            elif self.metric == 'sigmoid':
                dot_product = sum(a * b for a, b in zip(x1, x2))
                return math.tanh(self.alpha * dot_product + self.beta)
            elif self.metric.startswith('minkowski'):
                p = float(self.metric.split('_')[1])
                return sum(abs(a - b) ** p for a, b in zip(x1, x2)) ** (1 / p)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
        except Exception as e:
            print(f"An error occurred in calculateDistance: {e}")
            return None


    def predictSingle(self, x):
        try:
            distances = []
            for i in range((len(self.X_train))):
                distances.append((self.calculateDistance(x, self.X_train[i]), self.y_train[i]))
            distances = sorted(distances, key=lambda x: x[0])[:self.neighbors]
    
            if self.weights == 'distance':
                weighted_votes = {}
                for distance, label in distances:
                    weight = 1 / (distance + 1e-5) if distance != 0 else 1.0
                    if label in weighted_votes:
                        weighted_votes[label] += weight
                    else:
                        weighted_votes[label] = weight
                return max(weighted_votes, key=weighted_votes.get)
            else:
                labels = [neighbor[1] for neighbor in distances]
                most_common_label = Counter(labels).most_common(1)[0][0]
                return most_common_label
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
            if self.task == 'classification':
                correct = sum(1 for pred, actual in zip(predictions, y) if pred == actual)
                return correct / len(y)
            elif self.task == 'regression':
                ss_total = sum((actual - sum(y) / len(y)) ** 2 for actual in y)
                ss_residual = sum((actual - pred) ** 2 for pred, actual in zip(predictions, y))
                return 1 - (ss_residual / ss_total)
            else:
                raise ValueError(f"Unsupported task: {self.task}")
        except Exception as e:
            print(f"An error occurred in score: {e}")
            return None