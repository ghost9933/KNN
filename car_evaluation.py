import csv

# Importing my code 
from Scalers import *
from cleanHelpers import *
from mainDriver import main

def process_dataset(file_path, column_names, encodings):
    data = []
    with open(file_path, "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        for row in csv_reader:
            if row:
                data.append(row)

    for row in data:
        for key, value in zip(column_names, row):
            value = value.strip()
            if value in encodings[key]:
                row[column_names.index(key)] = encodings[key][value]
            else:
                row[column_names.index(key)] = None
    X = [row[:-1] for row in data]  # Features
    y = [row[-1] for row in data]   # Labels

    return X, y


# Running the script
if __name__ == "__main__":
    file_path = "./car_evaluation/car.data"
    column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

    encodings = {
        'buying': {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0},
        'maint': {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0},
        'doors': {'2': 0, '3': 1, '4': 2, '5more': 3},
        'persons': {'2': 0, '4': 1, 'more': 2},
        'lug_boot': {'small': 0, 'med': 1, 'big': 2},
        'safety': {'low': 0, 'med': 1, 'high': 2},
        'class': {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
    }

    # Data Preprocessing
    X, y = process_dataset(file_path, column_names, encodings)
    X_cleaned = replace_nan_and_question(X)
    X_imputed = replace_none_with_most_frequent(X_cleaned)
    X_imputed = [[float(value) for value in row] for row in X_imputed]

    # Standardize the features
    mean, std = standard_scaler_fit(X_imputed)
    X_scaled = standard_scaler_transform(X_imputed, mean, std)

    # Run the main function
    main(X_scaled, y, k=10)
