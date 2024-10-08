# -*- coding: utf-8 -*-
"""ML EDA.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14lrUh1n4KDo2iMuP4SOuSvWr-GexmxSo
"""

!pip install ucimlrepo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
hayes_roth = fetch_ucirepo(id=44)

X = pd.DataFrame(hayes_roth.data.features)
y = pd.DataFrame(hayes_roth.data.targets)
print(hayes_roth.metadata)
print(hayes_roth.variables)
print("Missing values in X:", X.isnull().sum().sum())
print("Missing values in y:", y.isnull().sum().sum())
print("X summary statistics:")
print(X.describe())
print("y summary statistics:")
print(y.describe())
plt.figure(figsize=(10, 6))
for i in range(X.shape[1]):
    plt.subplot(2, 3, i+1)
    plt.hist(X.iloc[:, i], bins=50)
    plt.title(f"Feature {i+1} Distribution")
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
plt.hist(y.iloc[:, 0], bins=50)
plt.title("Target Variable Distribution")
plt.show()

from ucimlrepo import fetch_ucirepo

car_evaluation = fetch_ucirepo(id=19)
X = car_evaluation.data.features
y = car_evaluation.data.targets
print(car_evaluation.metadata)
print(car_evaluation.variables)


print("Missing values in X:", X.isnull().sum().sum())
print("Missing values in y:", y.isnull().sum().sum())
print("X summary statistics:")
print(X.describe())
print("y summary statistics:")
print(y.describe())
plt.figure(figsize=(10, 6))
for i in range(X.shape[1]):
    plt.subplot(2, 3, i+1)
    plt.hist(X.iloc[:, i], bins=50)
    plt.title(f"Feature {i+1} Distribution")
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
plt.hist(y.iloc[:, 0], bins=50)
plt.title("Target Variable Distribution")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

breast_cancer = fetch_ucirepo(id=14)
X = pd.DataFrame(breast_cancer.data.features)
y = pd.DataFrame(breast_cancer.data.targets)
print(breast_cancer.metadata)
print(breast_cancer.variables)
print("Missing values in X:", X.isnull().sum().sum())
print("Missing values in y:", y.isnull().sum().sum())
print("X summary statistics:")
print(X.describe())
print("y summary statistics:")
print(y.describe())
numerical_features = X.select_dtypes(include=['int64', 'float64'])
categorical_features = X.select_dtypes(include=['object'])
plt.figure(figsize=(10, 6))
for i in range(numerical_features.shape[1]):
    plt.subplot(2, 3, i+1)
    plt.hist(numerical_features.iloc[:, i], bins=50)
    plt.title(f"Feature {i+1} Distribution")
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
for i in range(categorical_features.shape[1]):
    categorical_features.iloc[:, i].value_counts().plot(kind='bar')
    plt.title(f"Feature {i+1} Distribution")
    plt.show()