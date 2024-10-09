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
            row[column_names.index(key)] = encodings[key][value.strip()]

    X = [row[1:] for row in data]
    y = [row[0] for row in data]

    return X, y


if __name__ == "__main__":
    file_path = "./breast_cancer/breast-cancer.data"

    column_names = [
        "recurrence_status", "age", "menopause", "tumor_size", "inv_nodes",
        "node_caps", "deg_malig", "breast", "breast_quad", "irradiation"
    ]
    # mapping for coverting the categorical to numerical 
    encodings = {
        "recurrence_status": {"no-recurrence-events": 0, "recurrence-events": 1},
        "age": {"10-19": 0, "20-29": 1, "30-39": 2, "40-49": 3, "50-59": 4, "60-69": 5, "70-79": 6},
        "menopause": {"lt40": 0, "ge40": 1, "premeno": 2},
        "tumor_size": {"0-4": 0, "5-9": 1, "10-14": 2, "15-19": 3, "20-24": 4, "25-29": 5, "30-34": 6, "35-39": 7, "40-44": 8, "45-49": 9, "50-54": 10},
        "inv_nodes": {"0-2": 0, "3-5": 1, "6-8": 2, "9-11": 3, "12-14": 4, "15-17": 5, "18-20": 6, "21-23": 7, "24-26": 8},
        "node_caps": {"no": 0, "yes": 1, "?": None},
        "deg_malig": {"1": 1, "2": 2, "3": 3},
        "breast": {"left": 0, "right": 1},
        "breast_quad": {"left_up": 0, "left_low": 1, "right_up": 2, "right_low": 3, "central": 4, "?": None},
        "irradiation": {"no": 0, "yes": 1}
    }

    X, y = process_dataset(file_path, column_names, encodings)
    X_cleaned = replace_nan_and_question(X)
    X_imputed = replace_none_with_most_frequent(X_cleaned)

    mean, std = standard_scaler_fit(X_imputed)
    X_scaled = standard_scaler_transform(X_imputed, mean, std)

    main(X_scaled,y, k=10)
