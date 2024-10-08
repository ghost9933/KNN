# K-Nearest Neighbors Classifier Comparison Project

This project focuses on comparing the performance of a **Custom K-Nearest Neighbors (KNN)** algorithm implemented from scratch (`myCustomKNN.py`) with the standard **Scikit-learn KNN** across multiple datasets. The codebase includes features such as hyperparameter tuning, cross-validation, and hypothesis testing to analyze and compare the performance of the two KNN models.

## Project Structure

### Datasets
- **`breast_cancer/`**: Contains the dataset for the breast cancer classification problem.
  - `breast-cancer.data`: The dataset file.
  - `breast-cancer.names`: Description of the dataset features.

- **`car_evaluation/`**: Directory for the car evaluation dataset.

- **`hayes_roth/`**: Contains the Hayes-Roth dataset for classification.
  - `hayes-roth.data`: The dataset file.
  - `hayes-roth.names`: Description of the dataset features.
  - `hayes-roth.test`: Test dataset for evaluation.

### Core Scripts
- **`breast_cancer.py`**: Script to run classification tasks on the breast cancer dataset using both the Custom KNN and Scikit-learn KNN models.
- **`car_evaluation.py`**: Script to run classification tasks on the car evaluation dataset.
- **`hayes_roth.py`**: Script for classifying the Hayes-Roth dataset using KNN models.
- **`mainDriver.py`**: Central script for executing tasks across multiple datasets and testing different models.

### Custom Code and Utilities
- **`myCustomKNN.py`**: The core implementation of a custom K-Nearest Neighbors algorithm from scratch.
- **`DBknn.py`**: Modular working version of a potential enhanced KNN (possibly integrating density-based logic).
- **`hyperParamTune.py`**: Handles hyperparameter tuning for the custom KNN model.
- **`kFold.py`**: Implements K-Fold cross-validation to evaluate the accuracy of both KNN models.
- **`cleanHelpers.py`**: Helper functions for data preprocessing (e.g., handling missing values, replacing invalid entries).
- **`Scalers.py`**: Functions to scale and standardize datasets.
- **`hypoTesting.py`**: Functions for performing hypothesis testing to compare the statistical significance of the accuracy differences between Custom KNN and Scikit-learn KNN.


### Visualizations
- **`visuals.py`**: Code to generate visual representations (plots, charts) of the KNN performance or dataset characteristics.


## How to Run

### Prerequisites
- **Python 3.x**
- **Required Libraries**: Install the following libraries before running the scripts:
  
  ```bash
  pip install scikit-learn numpy
  ```

### Running the Scripts

1. **Choose the Dataset**: Each dataset has its own Python script:
   - For **breast cancer**, run `breast_cancer.py`.
   - For **car evaluation**, run `car_evaluation.py`.
   - For **Hayes-Roth**, run `hayes_roth.py`.

2. **Execute the Script**:
   
   ```bash
   python breast_cancer.py
   ```

   The script will:
   - Preprocess the dataset (e.g., handling missing values, scaling).
   - Perform hyperparameter tuning on the **Custom KNN** model.
   - Compare the performance of **Custom KNN** with **Scikit-learn KNN** using K-Fold cross-validation.
   - Conduct hypothesis testing to evaluate the significance of the differences in accuracy between the models.

### Example Commands

```bash
# Run the script for breast cancer dataset
python breast_cancer.py

# Run the script for car evaluation dataset
python car_evaluation.py

# Run the script for Hayes-Roth dataset
python hayes_roth.py
```

### Key Features

- **Hyperparameter Tuning**: Utilizes grid search to find the optimal values for `n_neighbors`, `weights`, and `metric` for the custom KNN.
- **K-Fold Cross-Validation**: Evaluates model performance using K-Fold cross-validation to get robust accuracy estimates.
- **Hypothesis Testing**: Performs a statistical test (e.g., paired T-test) to check if the difference between the performance of **Custom KNN** and **Scikit-learn KNN** is statistically significant.

## Modifying the Code

- **Dataset**: To use a different dataset, modify the `process_dataset` function in each script to load and preprocess your data.
- **Distance Metrics**: You can add more distance metrics to the custom KNN by extending the `calculateDistance` method in `myCustomKNN.py`.
- **Hypothesis Testing**: The logic for hypothesis testing is in `hypoTesting.py`. You can modify the significance level or add different types of statistical tests.

## Testing

To verify that everything is working, follow these steps:

1. **Run the script for any dataset** (e.g., `python breast_cancer.py`).
2. **Compare the accuracy** of both **Custom KNN** and **Scikit-learn KNN** after hyperparameter tuning.
3. **Perform hypothesis testing** to check if the accuracy difference between the two models is significant.

The output should display:
- The best hyperparameters for **Custom KNN**.
- The accuracy comparison of **Custom KNN** vs **Scikit-learn KNN** after K-fold cross-validation.
- The T-value and P-value from hypothesis testing.




# KNN Model Performance Comparison - Datasets Summary

This document summarizes the performance comparison of different KNN models (CustomKNN, DB_KNN, and Scikit-learn KNN) for three datasets: **Hayes-Roth**, **Car Evaluation**, and **Breast Cancer**. The results include hyperparameter tuning, K-Fold cross-validation, and hypothesis testing for each dataset.

## Hayes-Roth Dataset

### Hyperparameter Tuning Results
- **CustomKNN Best Parameters**: `{ 'n_neighbors': 1, 'weights': 'uniform', 'metric': 'euclidean' }`
- **Accuracy on 70-25 split**: **0.81**

- **DB_KNN Best Parameters**: `{ 'n_neighbors': 1, 'weights': 'uniform', 'metric': 'euclidean' }`
- **Accuracy on 70-25 split**: **0.81**

### K-Fold Cross-Validation
| Fold | CustomKNN Accuracy | DB_KNN Accuracy | Scikit-learn KNN Accuracy |
|------|--------------------|-----------------|---------------------------|
| 1    | 1.00               | 1.00            | 1.00                      |
| 2    | 0.62               | 0.62            | 0.69                      |
| 3    | 0.77               | 0.92            | 0.62                      |
| 4    | 0.69               | 0.92            | 0.85                      |
| 5    | 0.54               | 0.77            | 0.46                      |
| 6    | 0.85               | 1.00            | 0.77                      |
| 7    | 0.54               | 0.85            | 0.62                      |
| 8    | 0.69               | 0.92            | 0.69                      |
| 9    | 0.54               | 0.62            | 0.54                      |
| 10   | 0.69               | 0.85            | 0.69                      |

- **CustomKNN Mean Accuracy**: **0.69**
- **DB_KNN Mean Accuracy**: **0.85**
- **Scikit-learn KNN Mean Accuracy**: **0.69**

### Hypothesis Testing Results
- **CustomKNN Accuracies**: `[1.0, 0.615, 0.923, 0.923, 0.769, 1.0, 0.846, 0.923, 0.615, 0.846]`
- **Scikit-learn KNN Accuracies**: `[1.0, 0.692, 0.615, 0.846, 0.462, 0.769, 0.615, 0.692, 0.538, 0.692]`
- **T-value**: **3.72**
- **P-value**: **1.17**
- **Alpha**: **0.05**
- **Result**: The **Null hypothesis is accepted**, indicating no significant difference between the models.

## Car Evaluation Dataset

### Hyperparameter Tuning Results
- **CustomKNN Best Parameters**: `{ 'n_neighbors': 4, 'weights': 'uniform', 'metric': 'manhattan' }`
- **Accuracy on 70-25 split**: **0.73**

- **DB_KNN Best Parameters**: `{ 'n_neighbors': 4, 'weights': 'uniform', 'metric': 'manhattan' }`
- **Accuracy on 70-25 split**: **0.73**

### K-Fold Cross-Validation
| Fold | CustomKNN Accuracy | DB_KNN Accuracy | Scikit-learn KNN Accuracy |
|------|--------------------|-----------------|---------------------------|
| 1    | 0.61               | 0.61            | 0.71                      |
| 2    | 0.82               | 0.82            | 0.82                      |
| 3    | 0.71               | 0.96            | 0.71                      |
| 4    | 0.75               | 1.00            | 0.79                      |
| 5    | 0.57               | 0.86            | 0.57                      |
| 6    | 0.71               | 0.89            | 0.68                      |
| 7    | 0.61               | 0.82            | 0.68                      |
| 8    | 0.75               | 0.93            | 0.86                      |
| 9    | 0.71               | 0.93            | 0.71                      |
| 10   | 0.64               | 0.89            | 0.61                      |

- **CustomKNN Mean Accuracy**: **0.69**
- **DB_KNN Mean Accuracy**: **0.87**
- **Scikit-learn KNN Mean Accuracy**: **0.71**

### Hypothesis Testing Results
- **CustomKNN Accuracies**: `[0.607, 0.821, 0.964, 1.0, 0.857, 0.893, 0.821, 0.929, 0.929, 0.893]`
- **Scikit-learn KNN Accuracies**: `[0.714, 0.821, 0.714, 0.786, 0.571, 0.679, 0.679, 0.857, 0.714, 0.607]`
- **T-value**: **3.80**
- **P-value**: **1.15**
- **Alpha**: **0.05**
- **Result**: **Null hypothesis is accepted**, indicating no significant difference between the models.

## Breast Cancer Dataset

### Hyperparameter Tuning Results
- **CustomKNN Best Parameters**: `{ 'n_neighbors': 5, 'weights': 'distance', 'metric': 'euclidean' }`
- **Accuracy on 70-25 split**: **0.97**

- **DB_KNN Best Parameters**: `{ 'n_neighbors': 5, 'weights': 'distance', 'metric': 'euclidean' }`
- **Accuracy on 70-25 split**: **0.97**

### K-Fold Cross-Validation
| Fold | CustomKNN Accuracy | DB_KNN Accuracy | Scikit-learn KNN Accuracy |
|------|--------------------|-----------------|---------------------------|
| 1    | 0.97               | 0.97            | 0.97                      |
| 2    | 0.96               | 0.96            | 0.94                      |
| 3    | 0.98               | 0.99            | 0.97                      |
| 4    | 0.98               | 0.99            | 0.97                      |
| 5    | 0.97               | 0.98            | 0.98                      |
| 6    | 0.98               | 0.99            | 0.97                      |
| 7    | 0.95               | 0.99            | 0.97                      |
| 8    | 0.96               | 0.99            | 0.96                      |
| 9    | 0.98               | 1.00            | 0.98                      |
| 10   | 0.95               | 0.99            | 0.95                      |

- **CustomKNN Mean Accuracy**: **0.97**
- **DB_KNN Mean Accuracy**: **0.99**
- **Scikit-learn KNN Mean Accuracy**: **0.96**

### Hypothesis Testing Results
- **CustomKNN Accuracies**: `[0.971, 0.959, 0.994, 0.994, 0.977, 0.994, 0.994, 0.988, 1.0, 0.994]`
- **Scikit-learn KNN Accuracies**: `[0.965, 0.936, 0.971, 0.971, 0.977, 0.971, 0.965, 0.959, 0.977, 0.953]`
- **T-value**: **6.04**
- **P-value**: **0.657**
- **Alpha**: **0.05**
- **Result**: The **Null hypothesis is accepted**, indicating no significant difference between the models.

---

### Summary

The **DB_KNN** model consistently outperformed both **CustomKNN** and **Scikit-learn KNN** in terms of accuracy in all three datasets. The highest accuracy was achieved for the **Breast Cancer Dataset**, where **DB_KNN** reached a mean accuracy of **0.99**. Despite the higher mean accuracy values for **DB_KNN**, the hypothesis testing results indicate no statistically significant difference compared to **Scikit-learn KNN** across all datasets.