
# K-Nearest Neighbors Classifier Comparison Project

This project compares the performance of a custom K-Nearest Neighbors (KNN) implementation (`CustomKNN.py`) with the standard KNN from Scikit-learn, across different datasets. It includes hyperparameter tuning, cross-validation, and hypothesis testing to evaluate the performance.

## Project Structure

- **`breast_cancer/`**: Contains the dataset for the breast cancer classification problem.
  - `breast-cancer.data`: The dataset file.
  - `breast-cancer.names`: Description of the dataset.

- **`car_evaluation/`**: Directory for car evaluation dataset.

- **`hayes_roth/`**: Directory for the Hayes-Roth dataset.
  - `hayes-roth.data`: The dataset file.
  - `hayes-roth.names`: Dataset description.
  - `hayes-roth.test`: Test file for evaluation.

- **`breast_cancer.py`**: Script to perform classification on the breast cancer dataset, including hyperparameter tuning, cross-validation, and hypothesis testing.

- **`car_evaluation.py`**: Script to run classification tasks on the car evaluation dataset using both custom and Scikit-learn KNN models.

- **`hayes_roth.py`**: Script for classifying the Hayes-Roth dataset using KNN models.

- **`CustomKNN.py`**: Implementation of a custom KNN algorithm from scratch.

- **`enhancedKNN.py`**: Enhanced version of the KNN algorithm (if applicable).

- **`hyperParamTune.py`**: Hyperparameter tuning logic for the custom KNN model.

- **`kFold.py`**: Implements K-Fold cross-validation logic to test model accuracy.

- **`cleanHelpers.py`**: Helper functions for data preprocessing (e.g., replacing missing values).

- **`Scalers.py`**: Code for scaling and standardizing datasets.

- **`hypoTesting.py`**: Hypothesis testing functions to compare the performance of CustomKNN vs. Scikit-learn KNN.

- **`splitdata.py`**: Function to perform custom train-test splits (stratified).

## How to Run

1. Select a dataset (e.g., `breast_cancer`, `car_evaluation`, or `hayes_roth`).
2. Run the corresponding Python script (e.g., `breast_cancer.py`).
3. The script will:
   - Perform hyperparameter tuning using grid search.
   - Compare CustomKNN with Scikit-learn's KNN.
   - Run K-Fold cross-validation.
   - Perform hypothesis testing to check if there is a significant performance difference.

## Requirements

- Python 3.x
- Scikit-learn
- NumPy

## Contact

For any inquiries, please contact the author.

---

This README provides a brief overview of your project, folder structure, and how to use it.