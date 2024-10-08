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
