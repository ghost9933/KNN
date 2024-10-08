import random

def kFoldCV3(X, y, k, models, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)  # Set random seed for reproducibility

    fold_size = len(X) // k
    accuracies = {model_name: [] for model_name in models.keys()}

    fold_indices = list(range(len(X)))
    random.shuffle(fold_indices)  # Shuffle the indices to randomize the folds

    for i in range(k):
        # Create the test and train indices for this fold
        test_indices = fold_indices[i * fold_size:(i + 1) * fold_size]
        train_indices = fold_indices[:i * fold_size] + fold_indices[(i + 1) * fold_size:]

        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]

        print(f"Fold {i + 1}:")
        for model_name, model in models.items():
            # Train and evaluate model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = sum([1 if pred == actual else 0 for pred, actual in zip(y_pred, y_test)]) / len(y_test)
            accuracies[model_name].append(accuracy)
            print(f"  {model_name} Accuracy = {accuracy:.2f}")

    # Compute mean accuracies
    mean_accuracies = {model_name: sum(acc_list) / len(acc_list) for model_name, acc_list in accuracies.items()}

    return mean_accuracies, accuracies


