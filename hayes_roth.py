import csv

# Importing custom code
from Scalers import *
from cleanHelpers import *
from mainDriver import main

# Function to process the dataset and return X (features) and y (labels)
def process_dataset(file_path, column_names):
    data = []
    with open(file_path, "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header if present
        for row in csv_reader:
            if row:
                data.append(row)
    X = [row[1:-1] for row in data]  # Exclude 'name' (ID) and 'class' columns
    y = [row[-1] for row in data]    # Target is the 'class' column
    return X, y

# Running the script
if __name__ == "__main__":

    file_path = "hayes_roth/hayes-roth.data"  # Replace with your dataset file path
    column_names = ['name', 'hobby', 'age', 'educational_level', 'marital_status', 'class']

    # Data Preprocessing
    X, y = process_dataset(file_path, column_names)
    X, y = replace_nan_and_question(X, y)
    X = replace_none_with_most_frequent(X)
    y = replace_none_with_mode(y)

    # Convert features and labels to numeric
    X = [[int(value) for value in row] for row in X]
    y = [int(float(value)) for value in y]

    # Standardize the features
    mean, std = standard_scaler_fit(X)
    X_scaled = standard_scaler_transform(X, mean, std)

    # Run the main function
    main(X_scaled, y, k=10)
