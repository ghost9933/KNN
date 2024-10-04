from ucimlrepo import fetch_ucirepo
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
import math


print("breast_cancer")
# fetch dataset 
breast_cancer = fetch_ucirepo(id=14) 
  
X = breast_cancer.data.features.to_numpy()
y = breast_cancer.data.targets.to_numpy().ravel() 
y = breast_cancer.data.targets 
  
# metadata 
print(breast_cancer.metadata) 
  
# variable information 
print(breast_cancer.variables) 



for feature in X:
    print(feature,dict(Counter(X[feature])))

print('Target:',dict(Counter(y['Class'])))


# distacne cal culations

def euclidean_distance(x1:float, x2:float):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))

def manhattan_distance(x1:float, x2:float):
    return sum(abs(a - b) for a, b in zip(x1, x2))

def minkowski_distance(x1:float, x2:float, p=3):
    return sum(abs(a - b) ** p for a, b in zip(x1, x2)) ** (1/p)

def hamming_distance(x1:float, x2:float):
    return sum(a != b for a, b in zip(x1, x2)) / len(x1)



class KNN:
    def __init__(self, k=3, distance='euclidean', p=3):
        self.k = k
        self.distance = distance
        self.p = p

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        if self.distance == 'euclidean':
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.distance == 'manhattan':
            distances = [manhattan_distance(x, x_train) for x_train in self.X_train]
        elif self.distance == 'minkowski':
            distances = [minkowski_distance(x, x_train, self.p) for x_train in self.X_train]
        elif self.distance == 'hamming':
            distances = [hamming_distance(x, x_train) for x_train in self.X_train]
        else:
            raise ValueError("Unsupported distance metric")
        
 
        k_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        

        label_counts = {}
        for label in k_nearest_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        

        most_common_label = max(label_counts, key=label_counts.get)
        return most_common_label
    
import random

def accuracy_score(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

def k_fold_cross_validation(model, X, y, k=10):
    fold_size = len(X) // k
    scores = []
    indices = list(range(len(X)))
    random.shuffle(indices)

    for fold in range(k):
        start, end = fold * fold_size, (fold + 1) * fold_size if fold < k-1 else len(X)
        test_indices = indices[start:end]
        train_indices = indices[:start] + indices[end:]

        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)
        scores.append(score)

    return sum(scores) / k, scores


knn_euclidean = KNN(k=3, distance='euclidean')
knn_manhattan = KNN(k=3, distance='manhattan')
knn_minkowski = KNN(k=3, distance='minkowski', p=3) 
knn_hamming = KNN(k=3, distance='hamming')
mean_accuracy1, accuracies1 = k_fold_cross_validation(knn_euclidean, X, y, k=10)
mean_accuracy2, accuracies2 = k_fold_cross_validation(knn_manhattan, X, y, k=10)
mean_accuracy3, accuracies3 = k_fold_cross_validation(knn_minkowski, X, y, k=10)
mean_accuracy4, accuracies4 = k_fold_cross_validation(knn_hamming, X, y, k=10)


print(f"KNN Mean Accuracies:")
print(f"Using Euclidean distance :{mean_accuracy1*100}%")
print(f"Using Manhattan distance :{mean_accuracy2*100}%")
print(f"Using Minkowski distance :{mean_accuracy3*100}%")
print(f"Using Hamming distance :{mean_accuracy4*100}%")




from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


knn1 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn2 = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
knn3 = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=3)
knn4 = KNeighborsClassifier(n_neighbors=3, metric='hamming', algorithm='brute')

cv_scores1 = cross_val_score(knn1, X, y, cv=10)
cv_scores2 = cross_val_score(knn2, X, y, cv=10)
cv_scores3 = cross_val_score(knn3, X, y, cv=10)
cv_scores4 = cross_val_score(knn4, X, y, cv=10)


mean_accuracy1 = sum(cv_scores1) / len(cv_scores1)
mean_accuracy2 = sum(cv_scores2) / len(cv_scores2)
mean_accuracy3 = sum(cv_scores3) / len(cv_scores3)
mean_accuracy4 = sum(cv_scores4) / len(cv_scores4)


print(f"Sk-learn KNN Euclidean Mean Accuracy: {mean_accuracy1*100}%")
print(f"Sk-learn KNN Manhattan Mean Accuracy: {mean_accuracy2*100}%")
print(f"Sk-learn KNN Minkowski Mean Accuracy: {mean_accuracy3*100}%")
print(f"Sk-learn KNN Hamming Mean Accuracy: {mean_accuracy4*100}%")



from scipy.stats import ttest_rel

t_stat1, p_val1 = ttest_rel(accuracies1, cv_scores1) 
t_stat2, p_val2 = ttest_rel(accuracies2, cv_scores2) 
t_stat3, p_val3 = ttest_rel(accuracies3, cv_scores3) 
t_stat4, p_val4 = ttest_rel(accuracies4, cv_scores4)

alpha = 0.05
    
print(f"T-test results for KNN using Euclidean distance function:")
print(f"T-statistic : {t_stat1}, P_value : {p_val1}")
if p_val1 > alpha:
    print(f"The difference betwwen Custom KNN model and Sklearn's KNN model is not statistically significant. We reject Null hypothesis.")
else:
    print(f"The difference is statistically different, we accept alternate hypothesis")
print(f"\nT-test results for KNN using Manhattan distance function:")
print(f"T-statistic : {t_stat2}, P_value : {p_val2}")
if p_val2 > alpha:
    print(f"The difference betwwen Custom KNN model and Sklearn's KNN model is not statistically significant. We reject Null hypothesis.")
else:
    print(f"The difference is statistically different, we accept alternate hypothesis")
print(f"\nT-test results for KNN using Minkowski distance function:")
print(f"T-statistic : {t_stat3}, P_value : {p_val3}")
if p_val3 > alpha:
    print(f"The difference betwwen Custom KNN model and Sklearn's KNN model is not statistically significant. We reject Null hypothesis.")
else:
    print(f"The difference is statistically different, we accept alternate hypothesis")
print(f"\nT-test results for KNN using Hamming distance function:")
print(f"T-statistic : {t_stat4}, P_value : {p_val4}")
if p_val4 > alpha:
    print(f"The difference betwwen Custom KNN model and Sklearn's KNN model is not statistically significant. We reject Null hypothesis.")
else:
    print(f"The difference is statistically different, we accept alternate hypothesis.")