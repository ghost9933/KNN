from ucimlrepo import fetch_ucirepo
from collections import Counter
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import csv

# Fetch Dataset
breast_cancer = fetch_ucirepo(id=14)
X = breast_cancer.data.features
y = breast_cancer.data.targets

# Read the .data File
file_path = "./breast_cancer/breast-cancer.data"
column_names = [
    "recurrence_status", "age", "menopause", "tumor_size", "inv_nodes",
    "node_caps", "deg_malig", "breast", "breast_quad", "irradiation"
]
data = []
with open(file_path, "r") as file:
    csv_reader = csv.reader(file, delimiter=",")
    for row in csv_reader:
        if row:
            data.append(row)

# Encode Data
encodings = {
    "recurrence_status": {"no-recurrence-events": 0, "recurrence-events": 1},
    "age": {"10-19": 0, "20-29": 1, "30-39": 2, "40-49": 3, "50-59": 4, "60-69": 5, "70-79": 6},
    "menopause": {"lt40": 0, "ge40": 1, "premeno": 2},
    "tumor_size": {"0-4": 0, "5-9": 1, "10-14": 2, "15-19": 3, "20-24": 4, "25-29": 5, "30-34": 6, "35-39": 7, "40-44": 8, "45-49": 9, "50-54": 10},
    "inv_nodes": {"0-2": 0, "3-5": 1, "6-8": 2, "9-11": 3, "12-14": 4, "15-17": 5, "18-20": 6, "21-23": 7, "24-26": 8},
    "node_caps": {"no": 0, "yes": 1, "?": np.nan},
    "deg_malig": {"1": 1, "2": 2, "3": 3},
    "breast": {"left": 0, "right": 1},
    "breast_quad": {"left_up": 0, "left_low": 1, "right_up": 2, "right_low": 3, "central": 4, "?": np.nan},
    "irradiation": {"no": 0, "yes": 1}
}
for row in data:
    for key, value in zip(column_names, row):
        row[column_names.index(key)] = encodings[key][value.strip()]

# Convert to Lists of Features and Labels
X = [row[1:] for row in data]
y = [row[0] for row in data]

# Step 1: Handle Missing Values Using Imputer
# Ensure no NaN values exist before scaling or applying PCA
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = imputer.fit_transform(X)

# Step 2: Scale Features Using Standard Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Step 3: Dimensionality Reduction with PCA
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_pca = pca.fit_transform(X_scaled)

# Step 4: Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Step 5: Hyperparameter Tuning for KNN
param_grid = {
    'n_neighbors': list(range(1, 21)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev']
}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_knn = grid_search.best_estimator_

# Step 6: Train the Best Model and Evaluate
y_pred = best_knn.predict(X_test)

# Step 7: Evaluate the Performance of the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Accuracy of Best KNN model: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Compare Performance Metrics
print("\nComparison of Performance Metrics:")
print(f"{'Metric':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
for i, params in enumerate(grid_search.cv_results_['params']):
    metric = params['metric']
    accuracy = grid_search.cv_results_['mean_test_score'][i]
    print(f"{metric:<15} {accuracy:<10.2f}")

