from ucimlrepo import fetch_ucirepo
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
import math
import csv

file_path = "./breast_cancer/breast-cancer.data"
X = []
y = []

label_mapping = {}
with open(file_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        label = row[0]
        if label not in label_mapping:
            label_mapping[label] = len(label_mapping)
        y.append(label_mapping[label])
        features = []
        for feature in row[1:]:
            if feature not in label_mapping:
                label_mapping[feature] = len(label_mapping)
            features.append(label_mapping[feature])
        X.append(features)


print(X)

# for feature in X:
#     print(feature,dict(Counter(X[feature])))

print('Target:',dict(Counter(y)))

