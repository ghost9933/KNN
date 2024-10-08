import random

def k_fold_cross_validation(X, y, k, model1, model2, random_seed=None):

    if random_seed is not None:
        random.seed(random_seed)  # Set random seed for reproducibility

    fold_size = len(X) // k
    accuracies_model1 = []
    accuracies_model2 = []
    
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

        # Train and evaluate model1 (e.g., CustomKNN)
        model1.fit(X_train, y_train)
        y_pred_model1 = model1.predict(X_test)
        accuracy_model1 = sum([1 if pred == actual else 0 for pred, actual in zip(y_pred_model1, y_test)]) / len(y_test)
        accuracies_model1.append(accuracy_model1)

        # Train and evaluate model2 (e.g., Scikit-learn KNN)
        model2.fit(X_train, y_train)
        y_pred_model2 = model2.predict(X_test)
        accuracy_model2 = sum([1 if pred == actual else 0 for pred, actual in zip(y_pred_model2, y_test)]) / len(y_test)
        accuracies_model2.append(accuracy_model2)

        print(f"Fold {i + 1}:")
        print(f"  CustomKNN Accuracy = {accuracy_model1:.2f}")
        print(f"  Scikit-learn KNN Accuracy = {accuracy_model2:.2f}")

    # Return mean accuracies and individual fold accuracies for both models
    return sum(accuracies_model1) / len(accuracies_model1), sum(accuracies_model2) / len(accuracies_model2), accuracies_model1, accuracies_model2



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


