import csv
import math
import random
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from decimal import Decimal

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to integer using label encoding
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Calculate Minkowski distance
def getMinkowskiDistance(vector1, vector2, k):
    distance = 0.0
    for i in range(1, len(vector1)):
        distance += pow(abs(vector1[i] - vector2[i]), k)

    root = 1 / float(k)
    return round(Decimal(distance) ** Decimal(root), 5)

# Calculate the Euclidean distance between two vectors
def euclidean_distance(vector1, vector2):
    distance = 0.0
    for i in range(1, len(vector1)):  # Exclude the label column
        distance += (vector1[i] - vector2[i]) ** 2
    return math.sqrt(distance)

# KNN Algorithm with 10-fold cross-validation
def customKNNAlgorithm(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return predictions

def scikitKNN(train_set, test_set, num_neighbors_sklearn):
    # Train and evaluate scikit-learn k-NN
    X = [[row[i] for i in range(1, len(row))] for row in train_set]
    y = [row[0] for row in train_set]

    # Initialize k-NN classifier for scikit-learn
    knn_classifier = KNeighborsClassifier(n_neighbors=num_neighbors_sklearn)
    knn_classifier.fit(X, y)
    X_test_sklearn = [[row[i] for i in range(1, len(row))] for row in test_set]
    predictions_sklearn = knn_classifier.predict(X_test_sklearn)
    return predictions_sklearn

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = [(train_row, getMinkowskiDistance(test_row, train_row, 2)) for train_row in train]
    distances.sort(key=lambda x: x[1])
    neighbors = [row for row, _ in distances[:num_neighbors]]
    return neighbors

# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[0] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def getAccuracy(actual, predicted):
    n_correct_predictions = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            n_correct_predictions += 1
    return (n_correct_predictions * 100.0) / float(len(actual))

# Main function
if __name__ == "__main__":

    # Load the dataset from a .data file using pandas
    filename = './breast_cancer/breast-cancer.data'
    # Number of neighbours for knn
    n_neighbours = 5
    
    # Number of splits for k-fold cross-validation
    n_folds = 10

    # Type of distance used
    distance = euclidean_distance

    dataset_name = (filename.split("/")[-1]).split(".")[0]
    print("---------------------------------------------------------------------------------")
    print(dataset_name.upper())
    print("---------------------------------------------------------------------------------")
    
    dataset = load_csv(filename)

    # Convert string columns to integers using label encoding
    for i in range(1, len(dataset[0])):
        str_column_to_int(dataset, i)
    
    print("Number of neighbours in KNN : ", n_neighbours)
    print("Number of k folds for K fold cross validation : ", n_folds)

    accuracy_scores_custom = []
    accuracy_scores_sklearn = []
    
    # Initialize k-fold cross-validation
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)

        # Get results from Custom KNN algorithm
        predicted_custom = customKNNAlgorithm(train_set, test_set, n_neighbours)

        # Get results from Scikit-learn's KNN algorithm
        predicted_sk = scikitKNN(train_set, test_set, n_neighbours)

        actual = [row[0] for row in fold]
        accuracy_custom = getAccuracy(actual, predicted_custom)
        accuracy_sk = getAccuracy(actual, predicted_sk)
        accuracy_scores_custom.append(accuracy_custom)
        accuracy_scores_sklearn.append(accuracy_sk)
    
    # Calculate and print the average accuracy across folds for custom k-NN
    average_accuracy_custom = sum(accuracy_scores_custom) / len(accuracy_scores_custom)
    print("\nCustom k-NN Accuracy: ")
    print(accuracy_scores_custom)
    print("Custom k-NN Average Accuracy: {:.2f}\n".format(average_accuracy_custom))
    
    # Calculate and print the average accuracy across folds for scikit-learn k-NN
    average_accuracy_sklearn = sum(accuracy_scores_sklearn) / len(accuracy_scores_sklearn)
    print("Scikit-learn k-NN Accuracy: ")
    print(accuracy_scores_sklearn)
    print("Scikit-learn k-NN Average Accuracy: {:.2f}\n".format(average_accuracy_sklearn))
    print("---------------------------------------------------------------------------------")
    print("HYPOTHESIS TESTING")

    # Perform a paired t-test
    t_val, p_val = stats.ttest_rel(accuracy_scores_custom, accuracy_scores_sklearn)
    print("T-value : ", t_val)
    print("P-value : ", p_val)
    print("alpha : 0.05")
    print("---------------------------------------------------------------------------------")

        # Set the significance level (alpha) for your test
    alpha = 0.05

    # Check if the p-value is less than alpha
    if p_val < alpha:
        print('Null hypothesis REJECTED: There is a significant difference.')
    else:
        print('Null hypothesis ACCEPTED: There is no significant difference.')

  