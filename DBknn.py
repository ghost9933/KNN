from collections import Counter
import math

class CustomKNN:
    valid_metrics={'euclidean','chebyshev','manhattan','hamming','cosine'}
    def __init__(self, n=5, metric='euclidean',weights='uniform', task='classification'):
        if not isinstance(n, int):
            raise ValueError("n_neighbors should be an integer")
        self.neighbors = n
        self.metric = metric.lower()
        if self.metric not in CustomKNN.valid_metrics and not self.metric.startswith('minkowski'):
            raise ValueError("Invalid metric. Metric should be one of 'euclidean', 'manhattan', 'chebyshev', 'hamming', 'cosine'")
        if task not in {'classification','regression'}:
            raise ValueError("task should be either classification or regression")
        self.task = task
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
            elif self.metric == 'chebyshev':     
                return max(abs(a - b) for a, b in zip(x1, x2))
            elif self.metric == 'hamming':
                return sum(a != b for a, b in zip(x1, x2))
            elif self.metric == 'cosine':
                dot_product = sum(a * b for a, b in zip(x1, x2))
                magnitude_x1 = math.sqrt(sum(a ** 2 for a in x1))
                magnitude_x2 = math.sqrt(sum(b ** 2 for b in x2))
                return 1 - (dot_product / (magnitude_x1 * magnitude_x2))
            elif self.metric.startswith('minkowski'):
                p = float(self.metric.split('_')[1])
                return sum(abs(a - b) ** p for a, b in zip(x1, x2)) ** (1/p)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
        except Exception as e:
            print(f"An error occurred in calculateDistance: {e}")
            return None

    def predictSingle(self, x):
        try:
            distances = []
            for i in range(len(self.X_train)):
                distances.append((self.calculateDistance(x, self.X_train[i]), self.y_train[i]))
            distances = sorted(distances, key=lambda x: x[0])[:self.neighbors]

            if self.weights == 'density':
                # Compute density based weights
                weighted_votes = {}
                total_distance = sum(d for d, _ in distances)
                for distance, label in distances:
                    if total_distance == 0:
                        weight = 1.0
                    else:
                        density = 1 - (distance / total_distance)  # Density is higher when distance is lower
                        weight = density / (distance + 1e-5) if distance != 0 else 1.0
                    
                    if label in weighted_votes:
                        weighted_votes[label] += weight
                    else:
                        weighted_votes[label] = weight
                return max(weighted_votes, key=weighted_votes.get)

            elif self.weights == 'distance':
                # Already implemented distance-based weighting
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


from collections import Counter
import math

class dbKNN:
    valid_metrics = {'euclidean', 'chebyshev', 'manhattan', 'hamming', 'cosine'}

    def __init__(self, n=5, metric='euclidean', weights='uniform', task='classification'):
        if not isinstance(n, int):
            raise ValueError("n_neighbors should be an integer")
        self.neighbors = n
        self.metric = metric.lower()
        if self.metric not in dbKNN.valid_metrics and not self.metric.startswith('minkowski'):
            raise ValueError("Invalid metric. Metric should be one of 'euclidean', 'manhattan', 'chebyshev', 'hamming', 'cosine'")
        if task not in {'classification', 'regression'}:
            raise ValueError("task should be either classification or regression")
        self.task = task
        self.X_train = None
        self.y_train = None
        self.weights = weights
        self.alpha = None  # Stores the weight of this classifier in the final ensemble
        self.learners = []  # List to store weak learners

    def fit(self, X, y, sample_weights=None):
        self.X_train = X
        self.y_train = y
        if sample_weights is None:
            sample_weights = [1 / len(X)] * len(X)  # Initialize weights uniformly if not provided

        errors = [0] * len(X)  # Track misclassifications or errors

        for _ in range(5):  # Example: Train 5 weak learners for demonstration
            learner = CustomKNN(n=self.neighbors, metric=self.metric, weights=self.weights, task=self.task)  # Initialize CustomKNN
            learner.fit(X, y)

            predictions = learner.predict(X)
            incorrect = [1 if pred != actual else 0 for pred, actual in zip(predictions, y)]  # 1 for misclassified samples
            error = sum(sample_weights[i] * incorrect[i] for i in range(len(X)))  # Weighted error rate

            error /= sum(sample_weights)

            epsilon = 1e-10
            error = max(min(error, 1 - epsilon), epsilon)
            alpha = 0.5 * math.log((1 - error) / error)

            self.alpha = alpha
            errors = [errors[i] + incorrect[i] for i in range(len(X))]  # Increase the weights of the misclassified samples
            sample_weights = [sample_weights[i] * math.exp(alpha * incorrect[i]) for i in range(len(X))]
            sum_weights = sum(sample_weights)
            sample_weights = [weight / sum_weights for weight in sample_weights]

            self.learners.append((learner, alpha))  # Store the learner and its weight
            if error == epsilon:
                print("Perfect classifier found. Stopping early.")
                break


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
            weighted_votes = Counter()

            for learner, alpha in self.learners:
                prediction = learner.predictSingle(x)
                weighted_votes[prediction] += alpha

            return weighted_votes.most_common(1)[0][0]

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


