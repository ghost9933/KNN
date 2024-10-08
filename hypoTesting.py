import math

def compute_p_value(t_stat, n):
    if n <= 1:
        raise ValueError("Degrees of freedom must be greater than 1.")
    return 2 * (1 - (abs(t_stat) / (n - 1)))

def paired_t_test(sample1, sample2):
    if len(sample1) != len(sample2):
        raise ValueError("The two samples must have the same length.")
    differences = [x1 - x2 for x1, x2 in zip(sample1, sample2)]
    n = len(differences)
    mean_diff = sum(differences) / n
    variance = sum((d - mean_diff) ** 2 for d in differences) / (n - 1)
    std_diff = math.sqrt(variance)
    if std_diff == 0:
        print("Standard deviation is zero, division by zero detected.")
        return float('nan'), float('nan')
    t_stat = mean_diff / (std_diff / math.sqrt(n))
    p_value = compute_p_value(t_stat, n)    
    return t_stat, p_value


def hypothesis_testing(accuracy_custom_kfold, accuracy_sklearn_kfold):
    print("\nHYPOTHESIS TESTING")
    print("CustomKNN accuracies:", accuracy_custom_kfold)
    print("Scikit-learn KNN accuracies:", accuracy_sklearn_kfold)
    t_val, p_val = paired_t_test(accuracy_custom_kfold, accuracy_sklearn_kfold)
    print("T-value : ", t_val)
    print("P-value : ", p_val)
    print("alpha : 0.05")
    print("----------------------------------------------------------")
    
    alpha = 0.05  # Corrected alpha value
    alpha_two_tailed = alpha / 2  # Divide alpha by 2 for two-tailed test
    
    # Check for significance in either direction
    if p_val < alpha_two_tailed:
        print('Null hypothesis REJECTED: There is a significant difference.')
    else:
        print('Null hypothesis ACCEPTED: There is no significant difference.')


