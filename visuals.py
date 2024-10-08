import matplotlib.pyplot as plt

# Function to plot comparison of accuracies
def showCmp(accuracy_custom_kfold, accuracy_sklearn_kfold):
    plt.figure(figsize=(8, 6))
    fold_indices = range(1, len(accuracy_custom_kfold) + 1)
    plt.plot(fold_indices, accuracy_custom_kfold, marker='o', linestyle='-', color='b', label='CustomKNN')
    plt.plot(fold_indices, accuracy_sklearn_kfold, marker='s', linestyle='--', color='r', label='Scikit-learn KNN')
    plt.xlabel('Fold Number')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Accuracy: CustomKNN vs Scikit-learn KNN')
    plt.xticks(fold_indices)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

