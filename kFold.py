# KFold.py

def k_fold_cross_validation(X, y, k, model, custom=False):
    fold_size = len(X) // k
    accuracies = []
    fold_indices = list(range(len(X)))

    for i in range(k):
        test_indices = fold_indices[i * fold_size:(i + 1) * fold_size]
        train_indices = fold_indices[:i * fold_size] + fold_indices[(i + 1) * fold_size:]

        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]

        if custom:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        accuracy = sum([1 if pred == actual else 0 for pred, actual in zip(y_pred, y_test)]) / len(y_test)
        accuracies.append(accuracy)

        print(f"Fold {i + 1}: Accuracy = {accuracy:.2f}")

    return sum(accuracies) / len(accuracies)
